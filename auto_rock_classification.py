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

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class RockClassification:
    def __init__(self):
        self.folder      = 'Data'
        self.file        = 'GULFCOAST & TX CORE.csv'
        self.savefig     = True
        self.return_data = False
        self.verbose     = True

    def load_data(self):
        df = pd.read_csv(os.path.join(self.folder, self.file), low_memory=False)
        df['PORO'] = df['POROSITY'].fillna(df['EFFECTIVE_POROSITY'])
        df['PERM'] = df['K90'].fillna(df['KMAX']).fillna(df['KVERT'])
        self.df = df.dropna(subset=['PERM','PORO'], inplace=False)
        print('Data loaded:', self.df.shape) if self.verbose else None
        return self.df if self.return_data else None

    def plot_xy(self, data=None, cmap='jet', s=5, alpha=0.8, alphag=0.2, figsize=(12,4), log:bool=True, showfig:bool=True):
        if data is None:
            data = self.df
        _, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        ax1, ax2 = axs
        im1 = ax1.scatter(data['SURFACE_LONGITUDE'], data['SURFACE_LATITUDE'], c=data['PORO'], cmap=cmap, s=s, alpha=alpha)
        plt.colorbar(im1, ax=ax1, label='Porosity [%]')
        k = np.log10(data['PERM']) if log else data['PERM']
        im2 = ax2.scatter(data['SURFACE_LONGITUDE'], data['SURFACE_LATITUDE'], c=k, cmap=cmap, s=s, alpha=alpha)
        plt.colorbar(im2, ax=ax2, label='Permeability [mD]')
        for ax in axs:
            ax.grid(True, which='both', alpha=alphag)
            ax.set(xlabel='Surface Longitude')
        ax1.set(ylabel='Surface Latitude')
        plt.tight_layout()
        plt.savefig('figures/xy_poroperm', dpi=300) if self.savefig else None
        plt.show() if showfig else None
        return None
    
    def plot_curve(self, ax, df, curve, lb=None, ub=None, color='k', pad=0, s=2, mult=1,
                   units:str=None, mask=None, offset:int=0, title:str=None, label:str=None,
                   semilog:bool=False, bar:bool=False, fill:bool=None, rightfill:bool=False,
                   marker=None, edgecolor=None, ls=None, alpha=None):
        if mask is None:
            x, y = -offset+mult*df[curve], df.index
        else:
            x, y = -offset+mult*df[curve][mask], df.index[mask]
        lb = x[~np.isnan(x)].min() if lb is None else lb
        ub = x[~np.isnan(x)].max() if ub is None else ub
        if semilog:
            ax.semilogx(x, y, c=color, label=curve, alpha=alpha,
                        marker=marker, markersize=s, markeredgecolor=edgecolor, linestyle=ls, linewidth=s)
        else:
            if bar:
                ax.barh(y, x, color=color, label=curve, alpha=alpha)
            else:
                ax.plot(x, y, c=color, label=curve, alpha=alpha,
                        marker=marker, markersize=s, markeredgecolor=edgecolor, linewidth=s, linestyle=ls)
        if fill:
            ax.fill_betweenx(y, x, ub, alpha=alpha, color=color) if rightfill else ax.fill_betweenx(y, lb, x, alpha=alpha, color=color)
        if units is None:
            if hasattr(df, 'curvesdict'):
                units = df.curvesdict[curve].unit
            else:
                units = ''
        ax.set_xlim(lb, ub)
        ax.grid(True, which='both')
        ax.set_title(title, weight='bold') if title != None else None
        xlab = label if label is not None else curve
        if offset != 0:
            ax.set_xlabel('{} [{}] with {} offset'.format(xlab, units, offset), color=color, weight='bold')
        else:
            ax.set_xlabel('{} [{}]'.format(xlab, units), color=color, weight='bold')
        ax.xaxis.set_label_position('top'); ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_tick_params(color=color, width=s)
        ax.spines['top'].set_position(('axes', 1+pad/100))
        ax.spines['top'].set_edgecolor(color); ax.spines['top'].set_linewidth(2)
        if ls is not None:
            ax.spines['top'].set_linestyle(ls)
        return None
    

###########################################################################
############################## MAIN ROUTINE ###############################
###########################################################################
if __name__ == '__main__':
    time0 = time.time()

    arc = RockClassification()
    
    print('-'*60,'\n','Elapsed time: {:.3f} seconds'.format(time.time()-time0))
###########################################################################
################################## END ####################################
###########################################################################