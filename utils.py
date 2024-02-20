import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lasio
from scipy import linalg, stats, optimize

from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from keras import Model
from keras import optimizers
from keras import backend as K
from keras.layers import Input, LSTM, ConvLSTM1D, Conv1D, Conv1DTranspose, BatchNormalization, LeakyReLU, ReLU, MaxPooling1D, UpSampling1D
from keras.layers import Concatenate, Add, Reshape, Flatten, Dense, Dropout, ZeroPadding1D, Cropping1D, GlobalAveragePooling1D

class SPLogAnalysis:
    def __init__(self):
        self.return_data = False
        self.verbose     = True
        self.check_tf_gpu()

    def check_tf_gpu(self):
        sys_info = tf.sysconfig.get_build_info()
        if self.verbose:
            print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
            print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
            print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
            print(tf.config.list_physical_devices()[0],'\n', tf.config.list_physical_devices()[1])
        return None
    
    def read_all_headers(self):
        self.headers = {}
        k = 0
        for root, _, files in os.walk('Data/UT Export 9-19'):
            for f in files:
                fname = os.path.join(root,f)
                df = lasio.read(fname).df()
                self.headers[k] = df.columns
                k += 1
        return self.headers if self.return_data else None

    ### PLOTTING ###

    def plot_curve(self, ax, df, curve, lb, ub, color='k', size=2, pad=1, mult=1,
                semilog=False, bar=False, units=None, alpha=None, 
                marker=None, linestyle=None, fill=None, rightfill=False):
            x, y = mult*df[curve], df['DEPT']
            if semilog:
                ax.semilogx(x, y, c=color, label=curve, alpha=alpha)
            else:
                if bar:
                    ax.barh(y, x, color=color, label=curve, alpha=alpha)
                else:
                    ax.plot(x, y, c=color, label=curve, marker=marker, linestyle=linestyle, alpha=alpha)
            if fill:
                if rightfill:
                    ax.fill_betweenx(y, x, ub, alpha=alpha, color=color)
                else:
                    ax.fill_betweenx(y, lb, x, alpha=alpha, color=color)
            if units==None:
                units = df.curvesdict[curve].unit
            ax.set_xlim(lb, ub)
            ax.grid(True, which='both')
            ax.set_xlabel('{} [{}]'.format(curve, units), color=color, weight='bold') 
            ax.xaxis.set_label_position('top'); ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_tick_params(color=color, width=size)
            ax.spines['top'].set_position(('axes', pad))
            ax.spines['top'].set_edgecolor(color); ax.spines['top'].set_linewidth(1.75)
            if linestyle != None:
                ax.spines['top'].set_linestyle(linestyle)
            return None

    def plot_ccs_sand_wells(self, df=None, figsize=(10,5), value='POROSITY', cmap='jet', showcols:bool=False):
        df = pd.read_csv('Data/CCS_Sand_wells1.csv') if df is None else df
        print('DF Columns:', df.columns.values) if showcols else None
        plt.figure(figsize=figsize)
        plt.scatter(df['LONG'], df['LAT'], s=5, c=df[value], cmap=cmap)
        plt.xlabel('X (Longitude)', weight='bold'); plt.ylabel('Y (Latitude)', weight='bold')
        plt.colorbar(pad=0.04, fraction=0.046, label='{}'.format(value))
        plt.gca().set_facecolor('lightgray')
        plt.grid(True, which='both'); plt.tight_layout(); plt.show()
        return None

    def plot_survey(self, survey, figsize=(10,5), showcols:bool=False):
        print('DF Columns:', survey.columns.values) if showcols else None
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d', elev=40, azim=-45, aspect='equal')
        ax.scatter(survey['X(FT)'], survey['Y(FT)'], survey['MD(FT)'], s=5)
        ax.set_xlabel('X (ft)', weight='bold'); ax.set_ylabel('Y (ft)', weight='bold'); ax.set_zlabel('MD (ft)', weight='bold')
        ax.set(xlim3d=(0,500), ylim3d=(-500,0), zlim3d=(0,7000))
        ax.invert_zaxis()
        plt.tight_layout(); plt.show()
        return None

    def plot_well(self, well_name:str, figsize=(10,8), fig2size=(10,3)):
        well_log = lasio.read('Data/UT Export 9-19/{}.las'.format(well_name))
        well_name, well_field = well_log.header['Well']['WELL'].value, well_log.header['Well']['FLD'].value
        print(well_log.curvesdict.keys()) if self.verbose else None
        fig, axs = plt.subplots(1, 5, figsize=figsize, sharey=True)
        fig.suptitle('{} | {}'.format(well_field, well_name), weight='bold')
        ax1, ax2, ax3, ax4, ax5 = axs.flatten()
        ax11, ax12 = ax1.twiny(), ax1.twiny()
        self.plot_curve(ax12, well_log, 'CALI', 0.1, 100, color='k', fill=True)
        self.plot_curve(ax1, well_log, 'GR', 0, 120, color='olive', pad=1.08)
        self.plot_curve(ax11, well_log, 'GR_NORM', 0, 120, color='darkgreen', pad=1.16)
        ax21 = ax2.twiny()
        self.plot_curve(ax2, well_log, 'SP', -120, 20, color='magenta')
        self.plot_curve(ax21, well_log, 'SP_NORM', -120, 20, color='darkmagenta', pad=1.08)
        ax31 = ax3.twiny()
        self.plot_curve(ax3, well_log, 'VSH_GR', -0.05, 1.05, color='green')
        self.plot_curve(ax31, well_log, 'VSH_SP', -0.05, 1.05, color='purple', alpha=0.7, pad=1.08)
        ax41 = ax4.twiny()
        self.plot_curve(ax4, well_log, 'ILD', 0.2, 20, color='r', semilog=True)
        self.plot_curve(ax41, well_log, 'ASN', 0.2, 20, color='b', semilog=True, pad=1.08)
        ax51, ax52 = ax5.twiny(), ax5.twiny()
        self.plot_curve(ax5, well_log, 'RHOB', 1.65, 2.65, color='tab:red')
        self.plot_curve(ax51, well_log, 'DRHO', -0.5, 0.5, color='k', linestyle='--', pad=1.08)
        self.plot_curve(ax52, well_log, 'DT', 50, 180, color='tab:blue', pad=1.16)
        ax1.set_ylabel('DEPTH [ft]', weight='bold')
        plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
        plt.figure(figsize=fig2size)
        pd.plotting.autocorrelation_plot(well_log['SP'])
        plt.title('Autocorrelation of SP')
        plt.tight_layout(); plt.show()
        return well_log if self.return_data else None
    
    def make_arima(self, well_log, order=(5,1,0), figsize=(10,4)):
        model = ARIMA(well_log['SP'], order=order)
        model_fit = model.fit()
        print(model_fit.summary()) if self.verbose else None
        _, ax = plt.subplots(1,1,figsize=figsize)
        mu, std = stats.norm.fit(model_fit.resid)
        x = np.linspace(-20,20,500)
        p = stats.norm.pdf(x, mu, std)
        ax2 = ax.twiny()
        ax.plot(model_fit.resid, c='tab:blue', label='Residuals')
        ax2.plot(p,x,c='tab:red', linewidth=3, label='PDF')
        ax2.set_xticks([])
        plt.title('ARIMA MODEL | Residuals', weight='bold')
        plt.tight_layout(); plt.show()
        return None
    
class BaselineCorrection:
    def __init__(self):
        self.return_data = False
        self.verbose     = True