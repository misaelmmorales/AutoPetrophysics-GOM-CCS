############################################################################
#         AUTOMATIC ROCK CLASSIFICATION FOR RAPID CHARACTERIZATION         #
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
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from cartopy import crs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

class RockClassification:
    def __init__(self, well_number:int=0, method:str='leverett', n_classes:int=None, cutoffs=None, minpts:int=30,
                       random_state:int=2024, prop:str='PORO',
                       kexp=0.588, texp=0.732, pexp=0.864,
                       phimin=None, phimax=None, kmin=None, kmax=None,
                       folder:str='Data', file:str='GULFCOAST & TX CORE.csv', 
                       columns:list=['PORO', 'PERM', 'INTERVAL_DEPTH','SURFACE_LATITUDE','SURFACE_LONGITUDE'],
                       colors:list=['firebrick', 'dodgerblue', 'seagreen', 'gold', 'black'],
                       markers:list=['o', 's', 'D', '^', 'v'],
                       s1=10, sw=80, s2=50, ms=30, alpha=0.25, alphag=0.33,
                       cmap0:str='plasma', cmap:str='jet', figsize=(15,9), 
                       showfig:bool=True, savefig:bool=True, return_data:bool=False, verbose:bool=True,
                       **kwargs
                       ):
        
        self.folder       = folder
        self.file         = file
        self.mycols       = columns
        self.minpts       = minpts
        self.random_state = random_state
        self.colors       = colors
        self.markers      = markers
        self.well_number  = well_number
        self.prop         = prop
        self.method       = method
        self.n_classes    = n_classes
        self.cutoffs      = cutoffs
        self.kexp         = kexp
        self.texp         = texp
        self.pexp         = pexp
        self.phimin       = phimin
        self.phimax       = phimax
        self.kmin         = kmin
        self.kmax         = kmax
        self.s1           = s1
        self.sw           = sw
        self.s2           = s2
        self.ms           = ms
        self.alpha        = alpha
        self.alphag       = alphag
        self.cmap0        = cmap0
        self.cmap         = cmap
        self.figsize      = figsize
        self.showfig      = showfig
        self.savefig      = savefig
        self.return_data  = return_data
        self.verbose      = verbose

    def run_dashboard(self):
        time0 = time.time()
        self.load_data()
        self.process_data()
        self.make_dashboard()
        print('Elapsed time: {:.3f} seconds'.format(time.time()-time0)+'\n'+'-'*80)

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
                self.well_core[str(u)] = data[self.mycols].set_index('INTERVAL_DEPTH').sort_index()
                self.uwi_clean.append(str(u))
        print('Total number of wells:', len(self.well_core)) if self.verbose else None
        if self.return_data:
            return self.well_core, self.uwi_clean
        
    def check_nclass_cutoffs(self):
        if self.n_classes is None and self.cutoffs is not None:
            self.n_classes = len(self.cutoffs)
            assert len(self.cutoffs) < 6, 'Maximum number of classes is 5'
            if self.method=='kmeans' or self.method=='gmm':
                print('Warning: "kmeans" and "gmm" methods do not require cutoffs!')
                print('         Ignoring cutoffs. Calculating n_classes from cutoffs.')
        elif self.n_classes is not None and self.cutoffs is None:
            if self.method=='leverett':
                self.cutoffs = np.linspace(1, 100, self.n_classes+1)
            elif self.method=='winland':
                self.cutoffs = np.linspace(10, 1000, self.n_classes+1)
            elif self.method=='lorenz':
                self.cutoffs = np.linspace(0.01, 1, self.n_classes)
            else:
                self.cutoffs = np.linspace(0.1, 1000, self.n_classes+1)
            assert self.n_classes < 6, 'Maximum number of classes is 5'
        elif self.n_classes is None and self.cutoffs is None:
            self.n_classes = 3
            if self.method=='leverett':
                self.cutoffs = np.linspace(1, 100, self.n_classes+1)
            elif self.method=='winland':
                self.cutoffs = np.linspace(10, 1000, self.n_classes+1)
            elif self.method=='lorenz':
                self.cutoffs = np.linspace(0.01, 1, self.n_classes)
            else:
                self.cutoffs = np.linspace(0.1, 1000, self.n_classes+1)
        else:
            raise ValueError('Either n_classes or cutoffs must be provided, not both')

        if self.method not in ['kmeans', 'gmm', 'leverett', 'winland', 'lorenz']:
            raise ValueError('Invalid method. Choose between ("kmeans", "gmm", "leverett", "winland", "lorenz")')
        return None   
        
    def make_dashboard(self, **kwargs):
        self.check_nclass_cutoffs()
        
        self.wid    = self.uwi_clean[self.well_number]
        lati, longi = self.all_data['SURFACE_LATITUDE'], self.all_data['SURFACE_LONGITUDE']
        ymin, ymax  = lati.min()-0.5,  lati.max()+0.5
        xmin, xmax  = longi.min()-1.0, longi.max()+1.0

        if self.prop == 'PORO':
            q = self.all_data['PORO']/100
        elif self.prop == 'PERM':
            q = np.log10(self.all_data['PERM'])
        else:
            raise ValueError('Invalid property. Choose between ("PORO", "PERM")')

        d = self.well_core[self.wid]
        x, y = d['SURFACE_LONGITUDE'], d['SURFACE_LATITUDE']
        p, k, logk = d['PORO']/100, d['PERM'], np.log10(d['PERM'])
        X = pd.DataFrame({'PORO':p, 'PERM':logk})
        d.loc[:,'CLASS'] = np.zeros_like(p, dtype=int)

        if self.method == 'gmm' or self.method == 'kmeans':
            mthd = self.method.upper()
        else:
            mthd = self.method.capitalize()

        print('-'*80+'\n'+' '*16+'Automatic Core2Log Rock Classification Dashboard'+'\n'+'-'*80)
        print('Well #{} | UWI: {} | LAT: {} | LONG: {}'.format(self.well_number, self.wid, lati[self.well_number], longi[self.well_number]))
        print('Method: {} | Number of Classes: {} | Cutoffs: {}'.format(mthd, self.n_classes, self.cutoffs))
        print('Well shape: {}'.format(d.shape))
        print('-'*80)

        if self.phimin is None:
            self.phimin = p.min()
        if self.phimax is None:
            self.phimax = p.max()
        if self.kmin is None:
            self.kmin = k.min()
        if self.kmax is None:
            self.kmax = k.max()

        lin_poro = np.linspace(0, self.phimax, 50)
        lin_perm_low, lin_perm_med, lin_perm_high = [], [], []
        lin_X = pd.DataFrame({'PORO':np.linspace(0, self.phimax, len(d)), 'PERM':np.linspace(self.kmin, self.kmax, len(d))})

        if self.method == 'kmeans':
            lab = 'K-Means Class'
            clf = make_pipeline(MinMaxScaler(), KMeans(n_clusters=self.n_classes, random_state=self.random_state)).fit(X)
            d['CLASS'] = clf.predict(X) + 1
            v = clf.predict(lin_X) + 1
        
        elif self.method == 'gmm':
            lab = 'GMM Class'
            clf = make_pipeline(MinMaxScaler(), GaussianMixture(n_components=self.n_classes, random_state=self.random_state)).fit(X)
            d['CLASS'] = clf.predict(X) + 1
            v = clf.predict(lin_X) + 1

        elif self.method == 'leverett':
            self.cutoffs = [0] + self.cutoffs
            lab     = 'Leverett $\sqrt{k/\phi}$'
            mask, color_centers = [], []
            v = np.sqrt(k/p)
            def leverett_fun(w, l=lin_poro):
                return (w**2 * l)
            for i in range(len(self.cutoffs)-1):
                mask.append(np.logical_and(v>=self.cutoffs[i], v<=self.cutoffs[i+1]))
                color_centers.append(np.mean([self.cutoffs[i], self.cutoffs[i+1]]))
            for i in range(len(color_centers)):
                lin_perm_low.append(leverett_fun(self.cutoffs[i]))
                lin_perm_med.append(leverett_fun(color_centers[i]))
                lin_perm_high.append(leverett_fun(self.cutoffs[i+1]))
            for i, m in enumerate(mask):
                d.loc[m,'CLASS'] = int(i+1)
            
        elif self.method == 'winland':
            self.cutoffs = [0] + self.cutoffs
            lab     = 'Winland $R_{35}$'
            mask, color_centers = [], []
            v = k**self.kexp * 10**self.texp / p**self.pexp
            def winland_fun(r35, l=lin_poro):
                return ((r35 * l**self.pexp) / 10**self.texp)**(1/self.kexp)
            for i in range(len(self.cutoffs)-1):
                mask.append(np.logical_and(v>=self.cutoffs[i], v<=self.cutoffs[i+1]))
                color_centers.append(np.mean([self.cutoffs[i], self.cutoffs[i+1]]))
            for i in range(len(color_centers)):
                lin_perm_low.append(winland_fun(self.cutoffs[i]))
                lin_perm_med.append(winland_fun(color_centers[i]))
                lin_perm_high.append(winland_fun(self.cutoffs[i+1]))
            for i, m in enumerate(mask):
                d.loc[m,'CLASS'] = int(i+1)

        elif self.method == 'lorenz':
            lab = 'Lorenz Class'
            cp = np.cumsum(p)/p.sum()
            ck = np.cumsum(k)/k.sum()
            cv = np.cumsum(np.sort(ck)) / np.cumsum(np.sort(cp)).max()
            v  = np.concatenate([[0], np.diff(ck)/np.diff(cp)])
            mask = []
            for i in range(len(self.cutoffs)-1):
                mask.append((v>=self.cutoffs[i]) & (v<=self.cutoffs[i+1]))
            for i, m in enumerate(mask):
                d.loc[m,'CLASS'] = int(i+1)
        
        z = np.linspace(d.index.min(), d.index.max()+15, 100)
        t = np.zeros_like(z)
        class_values = d['CLASS'].values
        for i in range(len(t)):
            t[i] = class_values[np.argmin(np.abs(d.index.values - z[i]))]
            t[i] = t[i-1] if t[i] == 0 else t[i]

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

        # Spatial plot of core data
        ax1.scatter(x, y, marker='*', c='k', s=self.sw)
        im1 = ax1.scatter(longi, lati, c=q, cmap=self.cmap0, s=self.s1, vmax=0.35, transform=plate, zorder=2)
        ax1.coastlines(resolution='50m', color='black', linewidth=2, zorder=1)
        gl = ax1.gridlines(draw_labels=True)
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        cb1 = plt.colorbar(im1, pad=0.04, fraction=0.046); cb1.set_label('Porosity [v/v]', rotation=270, labelpad=15)
        ax1.vlines(x, ymin, y, color='k', ls='--', alpha=self.alpha)
        ax1.hlines(y, xmin, x, color='k', ls='--', alpha=self.alpha)
        ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel='Surface Longitude', ylabel='Surface Latitude')
        ax1.patch.set_facecolor('lightgrey')

        # Poro-vs-Perm with Classification Values
        if self.method=='leverett' or self.method=='winland':
            im2 = ax2.scatter(p, k, c=v, cmap=self.cmap, s=self.s2, edgecolor='k', linewidth=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046); cb.set_label(lab, rotation=270, labelpad=15)
            ax2.set(xlim=(0,self.phimax))
            for i, m in enumerate(mask):
                ax2.plot(lin_poro, lin_perm_med[i], c=self.colors[i])
                ax2.fill_between(lin_poro, lin_perm_low[i], lin_perm_high[i], color=self.colors[i], alpha=self.alphag)
        elif self.method=='lorenz':
            cmap2  = ListedColormap(self.colors[:len(self.cutoffs)])
            im2 = ax2.scatter(p, k, c=v, cmap=self.cmap, s=self.s2, edgecolor='k', linewidth=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046); cb.set_label(lab, rotation=270, labelpad=15)
            ax21 = ax2.twinx().twiny()
            ax21.scatter(np.sort(cp), np.sort(ck), c=v, cmap=cmap2)
            ax21.axline([0,0],[1,1], c='k', ls='--')
            ax21.set(xlim=(-0.01,1.025), ylim=(-0.025,1.025))
        else:
            cmap2  = ListedColormap(self.colors[:len(self.cutoffs)-1])
            im2 = ax2.scatter(p, k, c=d['CLASS'], cmap=cmap2, s=self.s2, edgecolor='k', linewidth=0.5)
            cb = plt.colorbar(im2, pad=0.04, fraction=0.046); cb.set_label(lab, rotation=270, labelpad=15)
            cb.set_ticks(np.arange(1,self.n_classes+1)); cb.set_ticklabels(np.arange(1,self.n_classes+1))
        ax2.set_yscale('log')
        ax2.set(xlabel='Porosity [v/v]', ylabel='Permeability [mD]')

        # Core porosity vs depth
        for i in range(self.n_classes):
            c = d['CLASS']
            if self.method != 'lorenz':
                ax3.scatter(p[c==i+1], d.index[c==i+1], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
            else:
                ax3.scatter(p[c==i], d.index[c==i], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
        ax3.set(xlabel='Porosity [v/v]', ylabel='Depth [ft]')
        ax3.invert_yaxis()

        # Core permeability vs depth
        for i in range(self.n_classes):
            c = d['CLASS']
            if self.method != 'lorenz':
                ax4.scatter(k[c==i+1], d.index[c==i+1], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
            else:
                ax4.scatter(k[c==i], d.index[c==i], c=self.colors[i], marker=self.markers[i], s=self.ms, edgecolor='gray', lw=0.5)
        ax4.set(xlabel='Permeability [mD]', xscale='log')

        # Rock Class vs depth
        for i in range(self.n_classes):
            if self.method != 'lorenz':
                ax5.fill_betweenx(z, 0, t, where=(t==i+1), color=self.colors[i])
            else:
                ax5.fill_betweenx(z, 0, t, where=(t==i), color=self.colors[i])
        ax5.set(xlabel='{} Rock Class'.format(mthd), xlim=(0.25, self.n_classes+0.25))
        ax5.set_xticks(np.arange(1,self.n_classes+1)); ax5.set_xticklabels(np.arange(1,self.n_classes+1))

        # plot settings
        fig.suptitle('Automatic Core2Log Rock Classification | W#{} | UWI: {} | {} method'.format(self.well_number, self.wid, mthd), weight='bold')
        for ax in axs:
            ax.grid(True, which='both', alpha=self.alphag)
        plt.tight_layout()
        plt.savefig('figures/ARC_dashboard_{}.png'.format(self.wid), dpi=300) if self.savefig else None
        plt.show() if self.showfig else None
        self.d = d
        return self.d if self.return_data else None

###########################################################################
############################## MAIN ROUTINE ###############################
###########################################################################
def main(args):
    print('-'*80+'\n'+'Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Current Working Directory:", os.getcwd())
    arc = RockClassification(**vars(args))
    arc.run_dashboard()   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Rock Classification for Core2Log')
    parser.add_argument('--folder', type=str, default='Data', help='Folder with core data')
    parser.add_argument('--file', type=str, default='GULFCOAST & TX CORE.csv', help='Core data file')
    parser.add_argument('--well_number', type=int, default=0, help='Well number to process')
    parser.add_argument('--method', type=str, default='leverett', help='Classification method')
    parser.add_argument('--n_classes', type=int, default=None, help='Number of classes')
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