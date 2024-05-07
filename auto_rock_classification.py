############################################################################
#        AUTOMATIC ROCK CLASSIFICATION FROM CORE DATA TO WELL LOGS:        #
#                  USING MACHINE LEARNING TO ACCELERATE                    #
#                POTENTIAL CO2 STORAGE SITE CHARACTERIZATION               #
############################################################################
# Author: Misael M. Morales (github.com/misaelmmorales)                    #
# Co-Authors: Oriyomi Raheem, Dr. Michael Pyrcz, Dr. Carlos Torres-Verdin  #
# Co-Authors: Murray Christie, Vladimir Rabinovich - S&P Global            #
# Date: 2024-03                                                            #
############################################################################
# Copyright (c) 2024, Misael M. Morales                                    #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

import os, argparse, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

from cartopy import crs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import torch
import tensorflow as tf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, BisectingKMeans, Birch

def check_tf_gpu():
    sys_info = tf.sysconfig.get_build_info()
    version, cuda, cudnn = tf.__version__, sys_info["cuda_version"], sys_info["cudnn_version"]
    count = len(tf.config.experimental.list_physical_devices())
    name  = [device.name for device in tf.config.experimental.list_physical_devices('GPU')]
    print('-'*60)
    print('----------------------- VERSION INFO -----------------------')
    print('TF version: {} | # Device(s) available: {}'.format(version, count))
    print('TF Built with CUDA? {} | CUDA: {} | cuDNN: {}'.format(tf.test.is_built_with_cuda(), cuda, cudnn))
    print(tf.config.list_physical_devices()[0],'\n', tf.config.list_physical_devices()[1])
    print('-'*60+'\n')
    return None

def check_torch_gpu():
    torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
    count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
    print('-'*60)
    print('----------------------- VERSION INFO -----------------------')
    print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
    print('# Device(s) available: {}, Name(s): {}'.format(count, name))
    print('-'*60+'\n')
    return None

###########################################################################
################# AUTOMATIC CORE2LOG ROCK CLASSIFICATION ##################
###########################################################################
class RockClassification:
    def __init__(self, well_number:int=0, 
                       method=None,
                       n_classes:int=None,
                       cutoffs=None,
                       minpts:int=30,
                       birch_threshold=0.1,
                       random_state:int=2024, 
                       prop:str='PORO',
                       kexp=0.588, texp=0.732, pexp=0.864,
                       phimin=None, phimax=None, kmin=None, kmax=None,
                       folder:str='Data', 
                       subfolder:str='UT Export core classification', 
                       file:str='GULFCOAST & TX CORE.csv', 
                       outfile:str='GULFCOAST & TX CORE postprocess.csv',
                       s1=10, sw=80, s2=50, ms=30, alpha=0.25, alphag=0.33,
                       cmap0:str='plasma', cmap:str='jet', 
                       showfig:bool=True, figsize=(15,9), savefig:bool=True,
                       return_data:bool=False, verbose:bool=True,
                       ):
        
        self.show_id         = False
        self.folder          = folder
        self.subfolder       = subfolder
        self.file            = file
        self.outfile         = outfile
        self.minpts          = minpts
        self.random_state    = random_state
        self.well_number     = well_number
        self.prop            = prop
        self.n_classes       = n_classes
        self.method          = method
        self.birch_threshold = birch_threshold
        self.cutoffs         = cutoffs
        self.kexp            = kexp
        self.texp            = texp
        self.pexp            = pexp
        self.phimin          = phimin
        self.phimax          = phimax
        self.kmin            = kmin
        self.kmax            = kmax
        self.s1              = s1
        self.sw              = sw
        self.s2              = s2
        self.ms              = ms
        self.alpha           = alpha
        self.alphag          = alphag
        self.cmap0           = cmap0
        self.cmap            = cmap
        self.figsize         = figsize
        self.showfig         = showfig
        self.savefig         = savefig
        self.return_data     = return_data
        self.verbose         = verbose
        self.plate           = crs.PlateCarree()
        self.incols          = ['PORO', 'PERM', 'INTERVAL_DEPTH','SURFACE_LATITUDE','SURFACE_LONGITUDE']
        self.outcols         = ['UWI', 'INTERVAL_DEPTH', 'SURFACE_LATITUDE', 'SURFACE_LONGITUDE', 'PORO', 'PERM', 'CLASS']
        self.colors          = ['dodgerblue', 'seagreen', 'firebrick', 'gold', 'black']
        self.markers         = ['o', 's', 'D', '^', 'X']
        self.method_err_msg  = 'Invalid method. Choose between ("kmeans", "bisectkmeans", "gmm", "birch", "leverett", "winland", "lorenz")'
        self.ml_methods      = ['kmeans', 'gmm', 'birch', 'bisectkmeans']
        self.ph_methods      = ['leverett', 'winland', 'lorenz']
        self.all_methods     = self.ml_methods + self.ph_methods

    '''
    Main routines
    '''
    def run_dashboard(self):
        time0 = time.time()
        self.bigloader()
        self.preprocessing()
        self.calculate_method_clf()
        self.postprocessing()
        print('Elapsed time: {:.3f} seconds'.format(time.time()-time0)+'\n'+'-'*80)
        return None
    
    def run_processing(self):
        time0 = time.time()
        postprocess_dfs = []
        self.bigloader()
        print('-'*80+'\n'+' '*20+'Processing Core2Log Rock Classification'+'\n'+'-'*80)
        self.mthd = self.method.upper() if self.method in self.ml_methods else self.method.capitalize()
        print('Method: {} | Number of Classes: {} | Cutoffs: {}'.format(self.mthd, self.n_classes, self.cutoffs)+'\n'+'-'*80)
        for i in tqdm(range(len(self.well_core)), desc='Processing well(s)', unit=' well'):
            self.well_number = i
            self.preprocessing(header=False)
            self.calculate_method_clf()
            df = self.d.copy()
            df['UWI'] = self.uwi_clean[i]
            df['INTERVAL_DEPTH'] = self.d.index
            df.to_csv(os.path.join(self.folder, self.subfolder, '{}.csv'.format(self.uwi_clean[i])), index=False)
            postprocess_dfs.append(df)
        outname = os.path.join(self.folder, self.outfile)
        print('-'*80+'\n'+'Processing Done!'+'\n'+'Saving ({}) ...'.format(outname))
        postprocess_df = pd.concat(postprocess_dfs, ignore_index=True)
        postprocess_df = postprocess_df[self.outcols]
        postprocess_df.to_csv(outname, index=False)
        print('Elapsed time: {:.3f} seconds'.format(time.time()-time0)+'\n'+'-'*80)
        return None
       
    def run_comparison(self, figsize=(20,12), n_classes:int=3, leverett_cutoffs=[10,20,40], winland_cutoffs=[150,300,500], lorenz_cutoffs=[0.5,2,5]):
        print('-'*80+'\n'+' '*23+'Compare Rock Classification Methods'+'\n'+'-'*80)
        print('Number of Classes: {}'.format(n_classes))
        print('Leverett Cutoffs: {}\nWinland Cutoffs: {}\nLorenz Cutoffs: {}'.format(leverett_cutoffs, winland_cutoffs, lorenz_cutoffs))
        print('-'*80)
        len_leverett, len_winland, len_lorenz = len(leverett_cutoffs), len(winland_cutoffs), len(lorenz_cutoffs)
        assert len_leverett == len_winland == len_lorenz  == n_classes, 'Number of cutoffs and classes must be the same for all methods'
        time0 = time.time()
        self.bigloader()
        self.comp_classes = n_classes
        self.all_classes, self.all_labels = {}, []
        lati, longi = self.all_data['SURFACE_LATITUDE'][self.well_number], self.all_data['SURFACE_LONGITUDE'][self.well_number]
        wid = self.uwi_clean[self.well_number]
        if self.show_id:
            print('-'*80+'\n'+'Well #{} | UWI: {} | LAT: {} | LONG: {}'.format(self.well_number, wid, lati, longi))
        print('Well shape: {}'.format(self.well_core[self.uwi_clean[self.well_number]].shape))
        print('-'*80)
        self.calc_comparisons(n_classes, leverett_cutoffs, winland_cutoffs, lorenz_cutoffs)
        self.plot_comparison(figsize)
        print('Elapsed time: {:.3f} seconds'.format(time.time()-time0)+'\n'+'-'*80)
        return None
    
    def run_spatial_map(self, figsize=(10,10),  npts:int=100, interp:str='linear', fill:bool=False, 
                        cmap:str='turbo', vmin=0, vmax=0.5, shrink=0.33):
        time0 = time.time()
        self.method='kmeans' if self.method is None else self.method
        self.n_classes=3 if self.n_classes is None else self.n_classes
        self.bigloader()
        self.preprocessing()
        self.plot_spatial_map(npts, interp, fill, cmap, vmin, vmax, shrink, figsize)
        print('Elapsed time: {:.3f} seconds'.format(time.time()-time0)+'\n'+'-'*80)
        return None
    
    '''
    Running commands
    '''
    def bigloader(self):
        self.load_data()
        self.process_data()
        return None

    def preprocessing(self, header:bool=True):
        self.check_nclass_cutoffs()
        self.calc_values()
        self.make_header() if header else None
        return None
    
    def postprocessing(self):
        self.make_class_array()
        self.make_dashboard()
        return None
    
    '''
    Auxiliary functions
    '''
    def load_data(self):
        self.all_data = pd.read_csv(os.path.join(self.folder, self.file), low_memory=False)
        self.all_data['PORO'] = self.all_data['POROSITY'].fillna(self.all_data['EFFECTIVE_POROSITY'])
        self.all_data['PERM'] = self.all_data['K90'].fillna(self.all_data['KMAX']).fillna(self.all_data['KVERT'])
        self.all_data = self.all_data.replace(0., np.nan, inplace=False)
        self.all_data = self.all_data.dropna(subset=['PORO', 'PERM'], inplace=False)
        if self.verbose:
            print('All data shape:', self.all_data.shape)
            self.all_data.head()
        return self.all_data if self.return_data else None
    
    def process_data(self):
        self.well_core = {}
        self.uwi_clean = []
        for u, data in self.all_data.groupby('UWI'):
            if data.shape[0] >= self.minpts:
                self.well_core[str(u)] = data[self.incols].set_index('INTERVAL_DEPTH').sort_index()
                self.uwi_clean.append(str(u))
        print('Total number of wells:', len(self.well_core)) if self.verbose else None
        if self.return_data:
            return self.well_core, self.uwi_clean
        
    def check_nclass_cutoffs(self):
        if self.method in self.ml_methods:
            assert self.n_classes is not None, 'Number of classes is required for ML methods'
            assert self.n_classes < 6, 'Maximum number of classes is 5'
            assert self.cutoffs is None, 'Cutoffs are not required for ML methods'
        elif self.method in self.ph_methods:
            assert self.cutoffs is not None, 'Cutoffs are required for physics-based methods'
            assert np.array_equal(self.cutoffs, np.sort(self.cutoffs)), 'Cutoffs must be in ascending order'
            assert self.n_classes is None, 'Number of classes is not required for physics-based methods'
            self.n_classes = len(self.cutoffs)
        else:
            raise ValueError(self.method_err_msg)    
        return None
    
    def calc_values(self):
        self.wid = self.uwi_clean[self.well_number]
        self.lati, self.longi = self.all_data['SURFACE_LATITUDE'], self.all_data['SURFACE_LONGITUDE']
        self.ymin, self.ymax  = self.lati.min()-0.5,  self.lati.max()+0.5
        self.xmin, self.xmax  = self.longi.min()-1.0, self.longi.max()+1.0
        self.d = self.well_core[self.wid]
        self.x, self.y = self.d['SURFACE_LONGITUDE'], self.d['SURFACE_LATITUDE']
        self.p, self.k, self.logk = self.d['PORO']/100, self.d['PERM'], np.log10(self.d['PERM'])
        self.X = pd.DataFrame({'PORO':self.p, 'PERM':self.logk})
        self.d.loc[:,'CLASS'] = np.zeros_like(self.p, dtype=int)
        
        if self.phimin is None:
            self.phimin = self.p.min()
        if self.phimax is None:
            self.phimax = self.p.max()
        if self.kmin is None:
            self.kmin = self.k.min()
        if self.kmax is None:
            self.kmax = self.k.max()
        
        self.lin_poro = np.linspace(0, self.phimax, 50)
        self.lin_perm_low, self.lin_perm_med, self.lin_perm_high = [], [], []
        self.lin_X = pd.DataFrame({'PORO':np.linspace(0, self.phimax, len(self.d)), 'PERM':np.linspace(self.kmin, self.kmax, len(self.d))})
        
        if self.prop == 'PORO':
            self.q = self.all_data['PORO']/100
        elif self.prop == 'PERM':
            self.q = np.log10(self.all_data['PERM'])
        else:
            raise ValueError('Invalid property to display. Choose between ("PORO", "PERM")')
        
        return None
    
    def calc_kmeans(self):
        self.lab = 'K-Means Class'
        self.clf = make_pipeline(MinMaxScaler(), KMeans(n_clusters=self.n_classes, random_state=self.random_state))
        self.clf.fit(self.X)
        sorted_centroids = np.argsort(self.clf.steps[-1][1].cluster_centers_.sum(axis=1))
        label_map = {sorted_centroids[i]:i for i in range(self.n_classes)}
        self.d['CLASS'] = np.array([label_map[i] for i in self.clf.predict(self.X)])+1
        self.v = np.array([label_map[i] for i in self.clf.predict(self.lin_X)])+1
        return None

    def calc_bisectkmeans(self):
        self.lab = 'Bisecting-K-Means Class'
        self.clf = make_pipeline(MinMaxScaler(), BisectingKMeans(n_clusters=self.n_classes, random_state=self.random_state))
        self.clf.fit(self.X)   
        sorted_centroids = np.argsort(self.clf.steps[-1][1].cluster_centers_.sum(axis=1))
        label_map = {sorted_centroids[i]:i for i in range(self.n_classes)}
        self.d['CLASS'] = np.array([label_map[i] for i in self.clf.predict(self.X)])+1
        self.v = np.array([label_map[i] for i in self.clf.predict(self.lin_X)])+1
        return None
    
    def calc_gmm(self):
        self.lab = 'GMM Class'
        self.clf = make_pipeline(MinMaxScaler(), GaussianMixture(n_components=self.n_classes, random_state=self.random_state))
        self.clf.fit(self.X)
        sorted_centroids = np.argsort(self.clf.steps[-1][1].means_.sum(axis=1))
        label_map = {sorted_centroids[i]:i for i in range(self.n_classes)}
        self.d['CLASS'] = np.array([label_map[i] for i in self.clf.predict(self.X)])+1
        self.v  = np.array([label_map[i] for i in self.clf.predict(self.lin_X)])+1
        return None
    
    def calc_birch(self):
        self.lab = 'Birch Class'
        self.clf = make_pipeline(MinMaxScaler(), Birch(n_clusters=self.n_classes, threshold=self.birch_threshold))
        self.clf.fit(self.X)
        sorted_centroids = np.argsort(self.clf.steps[-1][1].subcluster_centers_.sum(axis=1))
        sublabel_map = {sorted_centroids[i]:i for i in range(len(sorted_centroids))}
        subpreds = np.array([sublabel_map[i] for i in self.clf.predict(self.X)])
        label_map = {label:i for i, label in enumerate(np.unique(subpreds)[::-1])}
        self.d['CLASS'] = np.array([label_map[i] for i in subpreds])+1
        lin_subpreds = np.array([sublabel_map[i] for i in self.clf.predict(self.lin_X)])
        lin_label_map = {label:i for i, label in enumerate(np.unique(lin_subpreds)[::-1])}
        self.v = np.array([lin_label_map[i] for i in lin_subpreds])+1
        return None
    
    def calc_leverett(self):
        self.cutoffs = [0] + self.cutoffs
        self.lab     = 'Leverett $\sqrt{k/\phi}$'
        self.v       = np.sqrt(self.k/self.p)
        self.mask, self.color_centers = [], []
        def leverett_fun(w, l=self.lin_poro):
            return (w**2 * l)
        for i in range(len(self.cutoffs)-1):
            self.mask.append(np.logical_and(self.v>=self.cutoffs[i], self.v<=self.cutoffs[i+1]))
            self.color_centers.append(np.mean([self.cutoffs[i], self.cutoffs[i+1]]))
        for i in range(len(self.color_centers)):
            self.lin_perm_low.append(leverett_fun(self.cutoffs[i]))
            self.lin_perm_med.append(leverett_fun(self.color_centers[i]))
            self.lin_perm_high.append(leverett_fun(self.cutoffs[i+1]))
        for i, m in enumerate(self.mask):
            self.d.loc[m,'CLASS'] = int(i+1)
        return None
    
    def calc_winland(self):
        self.cutoffs = [0] + self.cutoffs
        self.lab     = 'Winland $R_{35}$'
        self.v       = self.k**self.kexp * 10**self.texp / self.p**self.pexp
        self.mask, self.color_centers = [], []
        def winland_fun(r35, l=self.lin_poro):
            return ((r35 * l**self.pexp) / 10**self.texp)**(1/self.kexp)
        for i in range(len(self.cutoffs)-1):
            self.mask.append(np.logical_and(self.v>=self.cutoffs[i], self.v<=self.cutoffs[i+1]))
            self.color_centers.append(np.mean([self.cutoffs[i], self.cutoffs[i+1]]))
        for i in range(len(self.color_centers)):
            self.lin_perm_low.append(winland_fun(self.cutoffs[i]))
            self.lin_perm_med.append(winland_fun(self.color_centers[i]))
            self.lin_perm_high.append(winland_fun(self.cutoffs[i+1]))
        for i, m in enumerate(self.mask):
            self.d.loc[m,'CLASS'] = int(i+1)
        return None
    
    def calc_lorenz(self):
        self.lab = 'Lorenz Slope'
        self.cp  = np.cumsum(self.p)/self.p.sum()
        self.ck  = np.cumsum(self.k)/self.k.sum()
        self.cv  = np.cumsum(np.sort(self.ck)) / np.cumsum(np.sort(self.cp)).max()
        self.v   = np.concatenate([[0], np.diff(self.ck)/np.diff(self.cp)])
        self.mask = []
        ct = [0] + self.cutoffs
        for i in range(len(ct)-1):
            self.mask.append(np.logical_and(self.v>=ct[i], self.v<=ct[i+1]))
        for i, m in enumerate(self.mask):
            self.d.loc[m,'CLASS'] = int(i+1)
        return None
    
    def calculate_method_clf(self):
        if self.method == 'kmeans':
            self.calc_kmeans()
        elif self.method == 'bisectkmeans':
            self.calc_bisectkmeans()
        elif self.method == 'gmm':
            self.calc_gmm()
        elif self.method == 'birch':
            self.calc_birch()
        elif self.method == 'leverett':
            self.calc_leverett()
        elif self.method == 'winland':
            self.calc_winland()
        elif self.method == 'lorenz':
            self.calc_lorenz()
        else:
            raise ValueError(self.method_err_msg)
        return None
    
    def make_class_array(self):
        self.z = np.linspace(self.d.index.min(), self.d.index.max(), len(self.d))
        self.t = np.zeros_like(self.z)
        self.class_values = self.d['CLASS'].values
        for i in range(len(self.t)):
            self.t[i] = self.class_values[np.argmin(np.abs(self.d.index.values - self.z[i]))]
            self.t[i] = self.t[i-1] if self.t[i]==0 else self.t[i]
        return None
    
    def make_header(self):
        print('-'*80+'\n'+' '*16+'Automatic Core2Log Rock Classification Dashboard'+'\n'+'-'*80)
        if self.show_id:
            print('Well #{} | UWI: {} | LAT: {} | LONG: {}'.format(self.well_number, self.wid, self.lati[self.well_number], self.longi[self.well_number]))
        self.mthd = self.method.upper() if self.method in self.ml_methods else self.method.capitalize()
        print('Method: {} | Number of Classes: {} | Cutoffs: {}'.format(self.mthd, self.n_classes, self.cutoffs))
        print('Well shape: {}'.format(self.d.shape))
        print('-'*80)
        return None  
        
    def make_dashboard(self):
        fig   = plt.figure(figsize=self.figsize)
        gs    = GridSpec(6, 6, figure=fig)
        ax1 = fig.add_subplot(gs[:3, :3], projection=self.plate)
        ax2 = fig.add_subplot(gs[3:, :3])
        ax3 = fig.add_subplot(gs[:, 3])
        ax4 = fig.add_subplot(gs[:, 4])
        ax5 = fig.add_subplot(gs[:, 5])
        ax4.sharey(ax3); ax5.sharey(ax3)
        axs = [ax1, ax2, ax3, ax4, ax5]

        if self.cutoffs is None:
            self.cutoffs = np.linspace(0.1, 1000, self.n_classes+1)

        # Spatial plot of core data
        ax1.scatter(self.x, self.y, marker='*', c='k', s=self.sw, edgecolor='k', lw=0.5)
        im1 = ax1.scatter(self.longi, self.lati, c=self.q, cmap=self.cmap0, s=self.s1, vmax=0.35, transform=self.plate, zorder=2)
        ax1.coastlines(resolution='50m', color='black', lw=2, zorder=1)
        gl = ax1.gridlines(draw_labels=True)
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        cb1 = plt.colorbar(im1, pad=0.04, fraction=0.046)
        cb1.set_label('Porosity [v/v]', rotation=270, labelpad=15)
        ax1.vlines(self.x, self.ymin, self.y, color='k', ls='--', alpha=self.alpha, lw=1)
        ax1.hlines(self.y, self.xmin, self.x, color='k', ls='--', alpha=self.alpha, lw=1)
        ax1.set(xlim=(self.xmin, self.xmax), ylim=(self.ymin, self.ymax), xlabel='Surface Longitude', ylabel='Surface Latitude')
        ax1.patch.set_facecolor('lightgrey')

        # Poro-vs-Perm with Classification Values
        if self.method=='leverett' or self.method=='winland':
            im2 = ax2.scatter(self.p, self.k, c=self.v, cmap=self.cmap, s=self.s2, edgecolor='k', lw=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046); cb.set_label(self.lab, rotation=270, labelpad=15)
            ax2.set(xlim=(0,self.phimax))
            for i in range(len(self.mask)):
                ax2.plot(self.lin_poro, self.lin_perm_med[i], c=self.colors[i], label='$C_{}$={:.2f}'.format(i, self.cutoffs[i+1]))
                ax2.legend(loc='upper center', fancybox=True, facecolor='lightgrey', edgecolor='k', ncol=len(self.cutoffs)-1)
                ax2.fill_between(self.lin_poro, self.lin_perm_low[i], self.lin_perm_high[i], color=self.colors[i], alpha=self.alphag)
        
        elif self.method=='lorenz':
            cmap2  = ListedColormap(self.colors[:len(self.cutoffs)])
            im2 = ax2.scatter(self.p, self.k, c=self.v, cmap=self.cmap, s=self.s2, edgecolor='k', lw=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046); cb.set_label(self.lab, rotation=270, labelpad=15)
            ax21 = ax2.twinx().twiny()
            ax21.scatter(np.sort(self.cp), np.sort(self.ck), c=self.v, cmap=cmap2, marker='>', s=self.s1)
            ax21.scatter(self.cv, np.sort(self.ck), c=self.cv, cmap=cmap2, marker='x', s=self.s1)
            handles = []
            for i in range(self.n_classes):
                handles.append(plt.Line2D([], [], label='$C_{}={:.2f}$'.format(i+1, self.cutoffs[i]), marker=self.markers[i],
                                           color=self.colors[i], markersize=self.s1, markeredgecolor='k', linestyle='None'))
            ax21.legend(handles=handles, loc='upper center', fancybox=True, facecolor='lightgrey', edgecolor='k', ncol=self.n_classes)
            ax21.axline([0,0],[1,1], c='k', ls='--')
            ax21.set(xlim=(-0.01,1.025), ylim=(-0.025,1.025), yticklabels=[], xlabel='Stratigraphic modified Lorenz coefficients')
            ax21.grid(True, which='both', alpha=self.alphag)
       
        else:
            cmap2  = ListedColormap(self.colors[:len(self.cutoffs)-1])
            for i in range(self.n_classes):
                p_ = self.p[self.d['CLASS']==i+1]
                k_ = self.k[self.d['CLASS']==i+1]
                ax2.scatter(p_, k_, c=self.colors[i], marker=self.markers[i], s=self.s2, edgecolor='k', lw=0.5, label='$C_{}$'.format(i+1))
            im2 = ax2.scatter(self.p, self.k, c=self.d['CLASS'], cmap=cmap2, marker=',', s=1e-3)
            ax2.legend(loc='upper center', fancybox=True, facecolor='lightgrey', edgecolor='k', ncol=self.n_classes)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046); cb.set_label(self.lab, rotation=270, labelpad=15)
            cb.set_ticks(np.arange(1,self.n_classes+1)); cb.set_ticklabels(np.arange(1,self.n_classes+1))
        ax2.set_yscale('log')
        ax2.set(xlabel='Porosity [v/v]', ylabel='Permeability [mD]')

        # Core porosity vs depth
        for i in range(self.n_classes):
            cl = self.d['CLASS']
            ax3.scatter(self.p[cl==i+1], self.d.index[cl==i+1], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
        ax3.set(title='Porosity [v/v]', ylabel='Depth [ft]')
        ax3.invert_yaxis()

        # Core permeability vs depth
        for i in range(self.n_classes):
            cl = self.d['CLASS']
            ax4.scatter(self.k[cl==i+1], self.d.index[cl==i+1], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
        ax4.set(title='Permeability [mD]', xscale='log')

        # Rock Class vs depth
        lab = self.lab.split(' ')[0]
        for i in range(self.n_classes):
            ax5.fill_betweenx(self.z, 0, self.t, where=(self.t==i+1), color=self.colors[i])
        ax5.set(title='Rock Class', xlim=(0.25, self.n_classes+0.25))
        ax5.set_xticks(np.arange(1,self.n_classes+1)); ax5.set_xticklabels(np.arange(1,self.n_classes+1))

        # plot settings
        if self.show_id:
            fig.suptitle('Automatic Core2Log Rock Classification | W#{} | UWI: {} | {} method'.format(self.well_number, self.wid, lab), weight='bold')
        [ax.grid(True, which='both', alpha=self.alphag) for ax in axs]
        plt.tight_layout()
        plt.savefig('figures/ARC_dashboard_{}_{}.png'.format(self.wid, self.method), dpi=300) if self.savefig else None
        plt.show() if self.showfig else None
        return None
    
    def calc_comparisons(self, n_classes:int, leverett_cutoffs, winland_cutoffs, lorenz_cutoffs):
        for _, m in enumerate(self.all_methods):
            self.method = m
            if m in self.ml_methods:
                self.n_classes = n_classes
                self.cutoffs = None
            elif m in self.ph_methods:
                self.n_classes = None
                if m == 'leverett':
                    self.cutoffs = leverett_cutoffs
                elif m == 'winland':
                    self.cutoffs = winland_cutoffs
                elif m == 'lorenz':
                    self.cutoffs = lorenz_cutoffs
            self.check_nclass_cutoffs()
            self.calc_values()
            self.calculate_method_clf()
            self.make_class_array()
            self.all_classes[m] = self.t
            self.all_labels.append(self.lab.split(' ')[0])
        self.mean_class = np.round(np.array(list(self.all_classes.values())).mean(axis=0))
        return None

    def plot_comparison(self, figsize=(20,12)):
        fig = plt.figure(figsize=figsize)
        gs  = GridSpec(3, 10, figure=fig)
        ax1  = fig.add_subplot(gs[:2, 0])
        ax2  = fig.add_subplot(gs[:2, 1])
        ax3  = fig.add_subplot(gs[:2, 2])
        ax4  = fig.add_subplot(gs[:2, 3])
        ax5  = fig.add_subplot(gs[:2, 4])
        ax6  = fig.add_subplot(gs[:2, 5])
        ax7  = fig.add_subplot(gs[:2, 6])
        ax8  = fig.add_subplot(gs[:2, 7])
        ax9  = fig.add_subplot(gs[:2, 8])
        ax10 = fig.add_subplot(gs[:2, 9])
        ax11 = fig.add_subplot(gs[2, :5])
        ax12 = fig.add_subplot(gs[2, 5:])
        dat_axs = [ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        top_axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
        all_axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
        [ax.invert_yaxis() for ax in top_axs]

        for ax, method in zip(dat_axs, self.all_classes.keys()):
            for i in range(self.comp_classes):
                ax.fill_betweenx(self.z, 0, self.all_classes[method], where=(self.all_classes[method]==i+1), color=self.colors[i])
            ax.set(title=self.all_labels.pop(0), xlim=(0.25, self.comp_classes+0.25), 
                   xticks=np.arange(1, self.comp_classes+1), xticklabels=np.arange(1, self.comp_classes+1))

        for i in range(self.n_classes):
            p_, k_, idz = self.p[self.mean_class==i+1], self.k[self.mean_class==i+1], self.d.index[self.mean_class==i+1]
            ax1.scatter(p_, idz, c=self.colors[i], marker=self.markers[i], s=self.ms)
            ax2.scatter(k_, idz, c=self.colors[i], marker=self.markers[i], s=self.ms)
            ax10.fill_betweenx(self.z, 0, self.mean_class, where=(self.mean_class==i+1), color=self.colors[i])
            ax12.scatter(p_, k_, c=self.colors[i], marker=self.markers[i], s=self.s2, edgecolor='k', lw=0.5, label='$C_{}$'.format(i+1))
        ax1.set(title='Porosity [v/v]', ylabel='Depth [ft]')
        ax2.set(title='Permeability [mD]', xscale='log')
        ax10.set(title='Mean Class', xlim=(0.25, self.comp_classes+0.25),
                 xticks=np.arange(1, self.comp_classes+1), xticklabels=np.arange(1, self.comp_classes+1))
        ax12.set(yscale='log', xlabel='Porosity [v/v]', ylabel='Permeability [mD]')
        ax12.legend(loc='upper center', fancybox=True, facecolor='lightgrey', edgecolor='k', ncol=self.n_classes)

        ax11.scatter(self.x, self.y, marker='*', c='k', s=self.sw, edgecolor='k', lw=0.5)
        im11 = ax11.scatter(self.longi, self.lati, c=self.q, cmap=self.cmap0, s=self.s1, vmax=0.35)
        cb1 = plt.colorbar(im11, pad=0.04, fraction=0.046)
        cb1.set_label('Porosity [v/v]', rotation=270, labelpad=15)
        ax11.vlines(self.x, self.ymin, self.y, color='k', ls='--', alpha=self.alpha, lw=1)
        ax11.hlines(self.y, self.xmin, self.x, color='k', ls='--', alpha=self.alpha, lw=1)
        ax11.set(xlim=(self.xmin, self.xmax), ylim=(self.ymin, self.ymax), xlabel='Surface Longitude', ylabel='Surface Latitude')
        if self.show_id:
            fig.suptitle('Automatic Core2Log Rock Classification | W#{} | UWI: {}'.format(self.well_number, self.wid), weight='bold')
        [ax.grid(True, which='both', alpha=self.alphag) for ax in all_axs]
        plt.tight_layout()
        plt.savefig('figures/Comparison_of_techniques_{}'.format(self.wid), dpi=300) if self.savefig else None
        plt.show() if self.showfig else None
        return None
    
    def plot_spatial_map(self, npts:int, interp:str, fill:bool, cmap, vmin, vmax, shrink, figsize=(10,10)):
        x, y, p = [], [], []
        for f in os.listdir(os.path.join(self.folder, self.subfolder)):
            d = pd.read_csv('{}/{}/{}'.format(self.folder, self.subfolder, f))
            c = d['CLASS']
            u = np.unique(c)
            x.append(d['SURFACE_LONGITUDE'].values[0])
            y.append(d['SURFACE_LATITUDE'].values[0])
            p.append(len(np.argwhere(c==u[-1]))/len(c))
        gx, gy = np.meshgrid(np.linspace(min(x), max(x), npts), np.linspace(min(y), max(y), npts))
        gp     = griddata((x, y), p, (gx, gy), method=interp)
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection=self.plate)
        if fill:
            ax.contourf(gx, gy, gp, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.contour(gx, gy, gp, cmap=cmap, vmin=vmin, vmax=vmax)
        im1 = ax.scatter(x, y, c=p, s=self.s1, cmap=cmap, lw=0.25, vmin=vmin, vmax=vmax)
        ax.coastlines(resolution='50m', color='black', lw=2, zorder=2)
        cb = plt.colorbar(im1, shrink=shrink)
        cb.set_label('Proportion of Sweet Spots', weight='bold', rotation=270, labelpad=15)
        gl = ax.gridlines(draw_labels=True)
        gl.right_labels = gl.top_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        plt.tight_layout()
        plt.savefig('figures/regional_sweetspots', dpi=300) if self.savefig else None
        plt.show() if self.showfig else None
        return None
    
    def spatial_depth_map(self):
        '''
        incomplete! needs work
        '''
        folder = 'Data/UT Export core classification'
        files  = os.listdir(folder)
        all_wells = {}
        all_uwi   = []
        for i, file in enumerate(files):
            df = pd.read_csv('{}/{}'.format(folder, file))
            uwi = df['UWI'].values[0]
            all_uwi.append(uwi)
            lat = df['SURFACE_LATITUDE'].values[0]
            lon = df['SURFACE_LONGITUDE'].values[0]
            d = df[['INTERVAL_DEPTH','CLASS']]
            x = {'LAT':lat, 'LON':lon, 'DAT':d}
            all_wells[uwi] = x
        n_classes = np.unique(all_wells[all_uwi[0]]['DAT']['CLASS'])
        colors = ['dimgrey', 'slategrey', 'firebrick']
        cmap2  = ListedColormap(colors[:len(n_classes)])
        #plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for well_data in all_wells.values():
            lat = well_data['LAT']
            lon = well_data['LON']
            d = well_data['DAT']
            z = np.linspace(d['INTERVAL_DEPTH'].min(), d['INTERVAL_DEPTH'].max(), len(d))
            t = np.zeros_like(z)
            class_values = d['CLASS'].values
            for i in range(len(t)):
                t[i] = class_values[np.argmin(np.abs(d['INTERVAL_DEPTH'].values[i] - z))]
                t[i] = t[i - 1] if t[i] == 0 else t[i]
            x = np.ones_like(z) * lon
            y = np.ones_like(z) * lat
            ax.scatter(x, y, z, c=t, marker='o', cmap=cmap2)
        ax.set_xlabel('Longitude', weight='bold')
        ax.set_ylabel('Latitude', weight='bold')
        ax.set_zlabel('Depth [ft]', weight='bold')
        ax.invert_zaxis()
        ax.grid(alpha=0.1)
        ax.view_init(azim=270, elev=90)
        plt.show()
        return None


###########################################################################
############################## MAIN ROUTINE ###############################
###########################################################################
def main(args):
    print('-'*80+'\n'+'Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Current Working Directory:", os.getcwd())
    RockClassification(**vars(args)).run_dashboard()
    RockClassification(**vars(args)).run_processing()
    RockClassification(**vars(args)).run_comparison()
    RockClassification(**vars(args)).run_spatial_map()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Rock Classification for Core2Log')
    parser.add_argument('--folder', type=str, default='Data', help='Folder with core data')
    parser.add_argument('--subfolder', type=str, default='UT Export core classification', help='Subfolder with core data')
    parser.add_argument('--file', type=str, default='GULFCOAST & TX CORE.csv', help='Core data file')
    parser.add_argument('--outfile', type=str, default='GULFCOAST & TX CORE postprocess.csv', help='Postprocessed core data file')
    parser.add_argument('--well_number', type=int, default=0, help='Well number to process')
    parser.add_argument('--n_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--method', type=str, default='leverett', help='Classification method')
    parser.add_argument('--birch_threshold', type=float, default=0.1, help='Threshold for Birch method')
    parser.add_argument('--cutoffs', type=list, default=None, help='Cutoffs for classification')
    parser.add_argument('--minpts', type=int, default=30, help='Minimum number of points per well')
    parser.add_argument('--random_state', type=int, default=2024, help='Random state for reproducibility')
    parser.add_argument('--prop', type=str, default='PORO', help='Property to classify')
    parser.add_argument('--kexp', type=float, default=0.588, help='Exponent for Winland R35')
    parser.add_argument('--texp', type=float, default=0.732, help='Exponent for Winland R35')
    parser.add_argument('--pexp', type=float, default=0.864, help='Exponent for Winland R35')
    parser.add_argument('--phimin', type=float, default=None, help='Minimum porosity')
    parser.add_argument('--phimax', type=float, default=None, help='Maximum porosity')
    parser.add_argument('--kmin', type=float, default=None, help='Minimum permeability')
    parser.add_argument('--kmax', type=float, default=None, help='Maximum permeability')
    parser.add_argument('--s1', type=int, default=10, help='Size for spatial plot')
    parser.add_argument('--sw', type=int, default=80, help='Size for well points')
    parser.add_argument('--s2', type=int, default=50, help='Size for poro-perm plot')
    parser.add_argument('--ms', type=int, default=30, help='Size for class plot')
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha for spatial plot')
    parser.add_argument('--alphag', type=float, default=0.33, help='Alpha for gridlines')
    parser.add_argument('--cmap0', type=str, default='plasma', help='Colormap for spatial plot')
    parser.add_argument('--cmap', type=str, default='jet', help='Colormap for poro-perm plot')
    parser.add_argument('--figsize', type=tuple, default=(15,9), help='Figure size')
    parser.add_argument('--showfig', type=bool, default=True, help='Show figure')
    parser.add_argument('--savefig', type=bool, default=True, help='Save figure')
    parser.add_argument('--return_data', type=bool, default=False, help='Return data')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose')
    args = parser.parse_args()
    main(args)
    
###########################################################################
################################## END ####################################
###########################################################################