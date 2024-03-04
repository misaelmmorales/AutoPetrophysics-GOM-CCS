############################################################################
#     AUTOMATIC CORE2LOG ROCK CLASSIFICATION FOR RAPID CHARACTERIZATION    #
# OF POTENTIAL CO2 STORAGE SITES IN THE GULF OF MEXICO USING DEEP LEARNING #
############################################################################
# Author: Misael M. Morales (github.com/misaelmmorales)                    #
# Co-Authors: Dr. Michael Pyrcz, Dr. Carlos Torres-Verdin - UT Austin      #
# Co-Authors: Murray Christie, Vladimir Rabinovich - S&P Global            #
# Date: 2024-03-01                                                         #
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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

from cartopy import crs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, BisectingKMeans, Birch

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
                       **kwargs
                       ):
        
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
        self.incols          = ['PORO', 'PERM', 'INTERVAL_DEPTH','SURFACE_LATITUDE','SURFACE_LONGITUDE']
        self.outcols         = ['UWI', 'INTERVAL_DEPTH', 'SURFACE_LATITUDE', 'SURFACE_LONGITUDE', 'PORO', 'PERM', 'CLASS']
        self.colors          = ['firebrick', 'dodgerblue', 'seagreen', 'gold', 'black']
        self.markers         = ['o', 's', 'D', '^', 'v']
        self.method_err_msg  = 'Invalid method. Choose between ("kmeans", "bisectkmeans", "gmm", "birch", "leverett", "winland", "lorenz")'
        self.ml_methods      = ['kmeans', 'gmm', 'birch', 'bisectkmeans']
        self.ph_methods      = ['leverett', 'winland', 'lorenz']
        self.all_methods     = self.ml_methods + self.ph_methods
        self.mthd = self.method.upper() if self.method in self.ml_methods else self.method.capitalize()

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
        postprocess_dfs = []
        self.bigloader()
        print('-'*80+'\n'+' '*20+'Processing Core2Log Rock Classification')
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
        print('Done!'+'\n'+'-'*80)
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
        self.lin_X = pd.DataFrame({'PORO':np.linspace(0, self.phimax, len(self.d)), 
                                   'PERM':np.linspace(self.kmin, self.kmax, len(self.d))})

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
        self.d['CLASS'] = self.clf.predict(self.X) + 1
        self.v = self.clf.predict(self.lin_X) + 1
        return None

    def calc_bisectkmeans(self):
        self.lab = 'Bisecting-K-Means Class'
        self.clf = make_pipeline(MinMaxScaler(), BisectingKMeans(n_clusters=self.n_classes, random_state=self.random_state))
        self.clf.fit(self.X)
        self.d['CLASS'] = self.clf.predict(self.X) + 1
        self.v = self.clf.predict(self.lin_X) + 1
        return None
    
    def calc_gmm(self):
        self.lab = 'GMM Class'
        self.clf = make_pipeline(MinMaxScaler(), GaussianMixture(n_components=self.n_classes, random_state=self.random_state))
        self.clf.fit(self.X)
        self.d['CLASS'] = self.clf.predict(self.X) + 1
        self.v = self.clf.predict(self.lin_X) + 1
        return None
    
    def calc_birch(self):
        self.lab = 'Birch Class'
        self.clf = make_pipeline(MinMaxScaler(), Birch(n_clusters=self.n_classes, threshold=self.birch_threshold))
        self.clf.fit(self.X)
        self.d['CLASS'] = self.clf.predict(self.X) + 1
        self.v = self.clf.predict(self.lin_X) + 1
        return None
    
    def calc_leverett(self):
        self.cutoffs       = [0] + self.cutoffs
        self.lab           = 'Leverett $\sqrt{k/\phi}$'
        self.mask          = []
        self.color_centers = []
        self.v             = np.sqrt(self.k/self.p)
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
        self.cutoffs       = [0] + self.cutoffs
        self.lab           = 'Winland $R_{35}$'
        self.mask          = []
        self.color_centers = []
        self.v = self.k**self.kexp * 10**self.texp / self.p**self.pexp
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
        print('Well #{} | UWI: {} | LAT: {} | LONG: {}'.format(self.well_number, self.wid, self.lati[self.well_number], self.longi[self.well_number]))
        print('Method: {} | Number of Classes: {} | Cutoffs: {}'.format(self.mthd, self.n_classes, self.cutoffs))
        print('Well shape: {}'.format(self.d.shape))
        print('-'*80)
        return None  
        
    def make_dashboard(self):
        fig   = plt.figure(figsize=self.figsize)
        gs    = GridSpec(6, 6, figure=fig)
        plate = crs.PlateCarree()

        ax1 = fig.add_subplot(gs[:3, :3], projection=plate)
        ax2 = fig.add_subplot(gs[3:, :3])
        ax3 = fig.add_subplot(gs[:, 3])
        ax4 = fig.add_subplot(gs[:, 4])
        ax5 = fig.add_subplot(gs[:, 5])
        ax4.sharey(ax3); ax5.sharey(ax3)
        axs = [ax1, ax2, ax3, ax4, ax5]

        if self.cutoffs is None:
            self.cutoffs = np.linspace(0.1, 1000, self.n_classes+1)

        # Spatial plot of core data
        ax1.scatter(self.x, self.y, marker='*', c='k', s=self.sw)
        im1 = ax1.scatter(self.longi, self.lati, c=self.q, cmap=self.cmap0, s=self.s1, vmax=0.35, transform=plate, zorder=2)
        ax1.coastlines(resolution='50m', color='black', linewidth=2, zorder=1)
        gl = ax1.gridlines(draw_labels=True)
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        cb1 = plt.colorbar(im1, pad=0.04, fraction=0.046)
        cb1.set_label('Porosity [v/v]', rotation=270, labelpad=15)
        ax1.vlines(self.x, self.ymin, self.y, color='k', ls='--', alpha=self.alpha)
        ax1.hlines(self.y, self.xmin, self.x, color='k', ls='--', alpha=self.alpha)
        ax1.set(xlim=(self.xmin, self.xmax), ylim=(self.ymin, self.ymax), xlabel='Surface Longitude', ylabel='Surface Latitude')
        ax1.patch.set_facecolor('lightgrey')

        # Poro-vs-Perm with Classification Values
        if self.method=='leverett' or self.method=='winland':
            im2 = ax2.scatter(self.p, self.k, c=self.v, cmap=self.cmap, s=self.s2, edgecolor='k', linewidth=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046)
            cb.set_label(self.lab, rotation=270, labelpad=15)
            ax2.set(xlim=(0,self.phimax))
            for i, m in enumerate(self.mask):
                ax2.plot(self.lin_poro, self.lin_perm_med[i], c=self.colors[i], label='$C_{}$={:.2f}'.format(i, self.cutoffs[i+1]))
                ax2.legend(loc='upper center', fancybox=True, facecolor='lightgrey', edgecolor='k', ncol=len(self.cutoffs)-1)
                ax2.fill_between(self.lin_poro, self.lin_perm_low[i], self.lin_perm_high[i], color=self.colors[i], alpha=self.alphag)
        elif self.method=='lorenz':
            cmap2  = ListedColormap(self.colors[:len(self.cutoffs)][::-1])
            im2 = ax2.scatter(self.p, self.k, c=self.v, cmap=self.cmap, s=self.s2, edgecolor='k', linewidth=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046)
            cb.set_label(self.lab, rotation=270, labelpad=15)
            ax21 = ax2.twinx().twiny()
            ax21.scatter(np.sort(self.cp), np.sort(self.ck), c=self.v, cmap=cmap2)
            ax21.axline([0,0],[1,1], c='k', ls='--')
            ax21.set(xlim=(-0.01,1.025), ylim=(-0.025,1.025))
        else:
            cmap2  = ListedColormap(self.colors[:len(self.cutoffs)-1])
            im2 = ax2.scatter(self.p, self.k, c=self.d['CLASS'], cmap=cmap2, s=self.s2, edgecolor='k', linewidth=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046)
            cb.set_label(self.lab, rotation=270, labelpad=15)
            cb.set_ticks(np.arange(1,self.n_classes+1))
            cb.set_ticklabels(np.arange(1,self.n_classes+1))
        ax2.set_yscale('log')
        ax2.set(xlabel='Porosity [v/v]', ylabel='Permeability [mD]')

        # Core porosity vs depth
        for i in range(self.n_classes):
            c = self.d['CLASS']
            if self.method != 'lorenz':
                ax3.scatter(self.p[c==i+1], self.d.index[c==i+1], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
            else:
                ax3.scatter(self.p[c==i], self.d.index[c==i], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
        ax3.set(xlabel='Porosity [v/v]', ylabel='Depth [ft]')
        ax3.invert_yaxis()

        # Core permeability vs depth
        for i in range(self.n_classes):
            c = self.d['CLASS']
            if self.method != 'lorenz':
                ax4.scatter(self.k[c==i+1], self.d.index[c==i+1], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
            else:
                ax4.scatter(self.k[c==i], self.d.index[c==i], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
        ax4.set(xlabel='Permeability [mD]', xscale='log')

        # Rock Class vs depth
        lab = self.lab.split(' ')[0]
        for i in range(self.n_classes):
            ax5.fill_betweenx(self.z, 0, self.t, where=(self.t==i+1), color=self.colors[i])
        ax5.set(xlabel='Rock Class', xlim=(0.25, self.n_classes+0.25))
        ax5.set_xticks(np.arange(1,self.n_classes+1)); ax5.set_xticklabels(np.arange(1,self.n_classes+1))

        # plot settings
        fig.suptitle('Automatic Core2Log Rock Classification | W#{} | UWI: {} | {} method'.format(self.well_number, self.wid, lab), weight='bold')
        for ax in axs:
            ax.grid(True, which='both', alpha=self.alphag)
        plt.tight_layout()
        plt.savefig('figures/ARC_dashboard_{}.png'.format(self.wid), dpi=300) if self.savefig else None
        plt.show() if self.showfig else None
        return None

###########################################################################
############################## MAIN ROUTINE ###############################
###########################################################################
    
def main(args):
    print('-'*80+'\n'+'Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Current Working Directory:", os.getcwd())
    RockClassification(**vars(args)).run_dashboard()
    RockClassification(**vars(args)).run_processing()

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