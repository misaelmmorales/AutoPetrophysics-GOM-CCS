############################################################################
#         AUTOMATIC WELL-LOG BASELINE CORRECTION VIA DEEP LEARNING         #
#            FOR RAPID SCREENING OF POTENTIAL CO2 STORAGE SITES            #
############################################################################
# Author: Misael M. Morales (github.com/misaelmmorales)                    #
# Co-Authors: Dr. Michael Pyrcz, Dr. Carlos Torres-Verdin - UT Austin      #
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

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lasio
from tqdm import tqdm
from scipy import stats, signal, interpolate, ndimage
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from cartopy import crs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import keras
import tensorflow as tf
from keras import Model
from keras import layers, optimizers
from keras import backend as K

###########################################################################
###################### S&P GLOBAL LOG ANALYSIS TOOL #######################
###########################################################################
class SPLogAnalysis:
    def __init__(self):
        self.return_data = False
        self.verbose     = True
        self.save_fig    = True
        print('-'*30,' Log Analysis Tool ','-'*30)
    
    def read_all_headers(self, folder='Data/UT Export 9-19/'):
        '''
        Read all headers one-by-one for all logs in the folder to identify repeated
        and unique curves. This will help in identifying the most common curves and 
        fixing multiple mnemonics for the same curve.
        '''
        self.headers = {}
        k = 0
        for root, _, files in os.walk(folder):
            for f in files:
                fname = os.path.join(root,f)
                df = lasio.read(fname).df()
                self.headers[k] = df.columns
                k += 1
        return self.headers if self.return_data else None

    ### PLOTTING ###
    def plot_curve(self, ax, df, curve, lb=0, ub=1, color='k', size=2, pad=1, mult=1,
                semilog=False, bar=False, units=None, alpha=None, 
                marker=None, linestyle=None, fill=None, rightfill=False, **kwargs):
            '''
            subroutine to plot a curve on a given axis
            '''
            x, y = mult*df[curve], df.index
            if linestyle is None:
                linestyle = kwargs.get('ls', None)
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
        '''
        Plot the dataset CCS_Sand_wells1.csv to visualize the spatial distribution of a value (e.g., POROSITY)
        '''
        df = pd.read_csv('Data/CCS_Sand_wells1.csv') if df is None else df
        print('DF Columns:', df.columns.values) if showcols else None
        plt.figure(figsize=figsize)
        plt.scatter(df['LONG'], df['LAT'], s=5, c=df[value], cmap=cmap)
        plt.xlabel('X (Longitude)', weight='bold'); plt.ylabel('Y (Latitude)', weight='bold')
        plt.colorbar(pad=0.04, fraction=0.046, label='{}'.format(value))
        plt.gca().set_facecolor('lightgray')
        plt.grid(True, which='both'); plt.tight_layout()
        plt.savefig('figures/CCS_Sand_wells1.png', dpi=300) if self.save_fig else None
        plt.show()
        return None

    def plot_survey(self, fname=None, figsize=(10,5), showcols:bool=False, maketitle:bool=False):
        '''
        Plot the directional survey from 'DATA/UT dir surveys'
        '''
        folder = 'Data/UT dir surveys'
        if fname==None:
            fname = os.listdir(folder)[14]
        survey = pd.read_csv('{}/{}'.format(folder, fname), skiprows=3, sep='\s+')
        wname = fname.split('.')[0].split('_')[0]
        print('DF Columns:', survey.columns.values) if showcols else None
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d', elev=40, azim=-45, aspect='equal')
        ax.scatter(survey['X(FT)'], survey['Y(FT)'], survey['MD(FT)'], s=5)
        ax.set_xlabel('X (ft)', weight='bold'); ax.set_ylabel('Y (ft)', weight='bold'); ax.set_zlabel('MD (ft)', weight='bold')
        ax.set(xlim3d=(0,500), ylim3d=(-500,0), zlim3d=(0,7000))
        ax.invert_zaxis()
        ax.set_title('{}'.format(wname), weight='bold') if maketitle else None
        plt.tight_layout()
        plt.savefig('figures/Dir_Survey_{}.png'.format(wname), dpi=300) if self.save_fig else None
        plt.show()
        return None

    def plot_well(self, well_name:str=None, maketitle:bool=False, printkeys:bool=False,
                  figsize=(10,8), fig2size=(10,3), curve='SP', order=(5,1,0), fig3size=(10,4)):
        '''
        Full well log plot with tracks for each curve
        '''
        folder = 'Data/UT Export 9-19'
        if well_name is None:
            well_name = os.listdir(folder)[5]
        well_log = lasio.read('{}/{}'.format(folder, well_name))
        well_name, well_field = well_log.header['Well']['WELL'].value, well_log.header['Well']['FLD'].value
        print(well_log.curvesdict.keys()) if printkeys else None
        fig, axs = plt.subplots(1, 5, figsize=figsize, sharey=True)
        fig.suptitle('{} | {}'.format(well_field, well_name), weight='bold') if maketitle else None
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
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig('figures/well_{}.png'.format(well_name), dpi=300) if self.save_fig else None
        plt.show()
        # autocorrelation plot
        plt.figure(figsize=fig2size)
        pd.plotting.autocorrelation_plot(well_log['SP'])
        plt.title('Autocorrelation of SP')
        plt.tight_layout()
        plt.savefig('figures/autocorr_well_{}.png'.format(well_name), dpi=300) if self.save_fig else None
        plt.show()
        ### Calculate ARIMA model for a given well log curve
        model = ARIMA(well_log[curve], order=order)
        model_fit = model.fit()
        print(model_fit.summary()) if self.verbose else None
        _, ax = plt.subplots(1,1,figsize=fig3size)
        mu, std = stats.norm.fit(model_fit.resid)
        x = np.linspace(-20,20,500)
        p = stats.norm.pdf(x, mu, std)
        ax2 = ax.twiny()
        ax.plot(model_fit.resid, c='tab:blue', label='Residuals')
        ax2.plot(p,x,c='tab:red', linewidth=3, label='PDF')
        ax2.set_xticks([])
        plt.title('ARIMA MODEL | Residuals', weight='bold')
        plt.tight_layout()
        plt.savefig('figures/arima_{}'.format(well_name), dpi=300) if self.save_fig else None
        plt.show()
        return well_log if self.return_data else None    

###########################################################################
###################### AUTOMATIC BASELINE CORRECTION ######################
###########################################################################
class BaselineCorrection:
    def __init__(self):
        self.log_length      = 44055
        self.folder          = 'Data/UT Export 9-19/'
        self.scaler          = 'standard'
        self.bounds          = [10, 90]

        self.decimate        = False
        self.decimate_q      = 10
        self.dxdz            = True
        self.hilbert         = True
        self.detrend         = True
        self.fourier         = True
        self.fourier_window  = [1e-3,0.025]
        self.fourier_scale   = 1e3
        self.symiir          = True
        self.symiir_c0       = 0.5
        self.symiir_z1       = 0.1
        self.savgol          = True
        self.savgol_window   = 15
        self.savgol_order    = 2
        self.cspline         = True
        self.spline_lambda   = 0.0
        self.autocorr        = True
        self.autocorr_method = 'fft'
        self.autocorr_mode   = 'same'

        self.return_data     = False
        self.verbose         = True
        self.save_fig        = True
        print('\n','-'*30,' Baseline Correction Tool ','-'*30)
        self.check_tf_gpu()
    
    def check_tf_gpu(self):
        sys_info = tf.sysconfig.get_build_info()
        if self.verbose:
            print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
            print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
            print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
            print(tf.config.list_physical_devices()[0],'\n', tf.config.list_physical_devices()[1])
            print('-'*60) if self.verbose else None
        return None

    def load_logs(self, 
                    preload:bool     = False,   
                    preload_file:str = 'Data/log_data.npy',
                    folder           = None,
                    save_file:str    = 'Data/log_data.npy',
                    showfig          = True):
            '''
            Load all logs. 
                If preload=False: 
                    This function will read over each LAS file in the folder and extract the 
                    SP and SP_NORM curves, and then save then into a single NumPy array, along with a 
                    clean version that removes NaNs.
                If preload=True: 
                    This function will load the logs from the saved NumPy file and creates a clean version too.
            '''
            if folder==None:
                folder = self.folder
            if preload:
                self.logs = np.load(preload_file)
                print('Well logs raw:', self.logs.shape) if self.verbose else None
            else:
                files = os.listdir(folder)
                logs_list, k = {}, 0
                for file in tqdm(files, desc='Processing Files', unit=' file(s)'):
                    log = lasio.read('{}/{}'.format(folder,file))
                    if 'SP' in log.curvesdict.keys() and 'SP_NORM' in log.curvesdict.keys():
                        logs_list[k] = pd.DataFrame({'DEPT': log['DEPT'], 'SP': log['SP'], 'SP_NORM': log['SP_NORM']})
                        k += 1
                logs = np.zeros((len(logs_list),self.log_length,3))
                for i in range(len(logs_list)):
                    logs[i,logs_list[i].index,:] = logs_list[i].values
                self.logs = np.where(logs==0, np.nan, logs)
                np.save(save_file, self.logs)
                print('Well logs raw:', self.logs.shape) if self.verbose else None
            self.logs_clean = np.nan_to_num(self.logs, nan=0)        
            self.calc_features()
            self.plot_SP_and_NORM(data=self.logs, short_title='raw') if showfig else None
            self.plot_SP_and_NORM(data=self.logs_clean, short_title='clean') if showfig else None
            print('-'*60) if self.verbose else None
            if self.return_data:
                return self.logs, self.logs_clean
   
    def scale_and_random_split(self, scaler=None, test_size=0.2, showfig=True):
        self.datascaler(scaler=scaler)
        self.plot_SP_and_NORM(self.logs_norm, short_title='normalized', xlim=(-5,5)) if showfig else None
        x = np.delete(self.logs_norm, 2, axis=-1)
        y = np.expand_dims(self.logs_norm[...,2], -1)
        self.train_idx = np.random.choice(range(x.shape[0]), size=int(x.shape[0]*(1-test_size)), replace=False)
        self.test_idx  = np.array([i for i in range(x.shape[0]) if i not in self.train_idx])
        self.X_train, self.X_test = x[self.train_idx], x[self.test_idx]
        self.y_train, self.y_test = y[self.train_idx], y[self.test_idx]
        if self.verbose:
            print('X_train: {} | X_test: {}'.format(self.X_train.shape, self.X_test.shape))
            print('y_train: {} | y_test: {}'.format(self.y_train.shape, self.y_test.shape))
        self.train_test_data = {'X_train':self.X_train, 'X_test':self.X_test, 
                                'y_train':self.y_train, 'y_test':self.y_test}
        self.plot_features(train_or_test='train') if showfig else None
        self.plot_features(train_or_test='test') if showfig else None
        print('-'*60) if self.verbose else None
        return self.train_test_data if self.return_data else None
    
    def make_model(self, pretrained=None, show_summary:bool=False,
                   kernel_size=15, dropout=0.2, depths=[16,32,64],
                   optimizer='adam', lr=1e-3, loss='mse', metrics=['mse'], 
                   epochs=100, batch_size=30, valid_split=0.25, verbose=True,
                   save_name='baseline_correction_model', figsize=(10,5)):
        if pretrained != None:
            self.model = keras.models.load_model(pretrained)
            self.encoder = Model(inputs=self.model.input, outputs=self.model.layers[15].output)
            self.model.summary() if show_summary else None
            print('-'*50,'\n','# Parameters: {:,}'.format(self.model.count_params())) if self.verbose else None
            print('-'*60) if self.verbose else None
            if self.return_data:
                return self.model, self.encoder
        elif pretrained == None:
            self.make_nn(kernel_size=kernel_size, drop=dropout, depths=depths, in_channels=self.X_train.shape[-1])
            print('-'*50,'\n','# Parameters: {:,}'.format(self.model.count_params())) if self.verbose else None
            self.model.summary() if show_summary else None
            self.train_model(optimizer=optimizer, lr=lr, loss=loss, metrics=metrics, 
                             epochs=epochs, batch_size=batch_size, valid_split=valid_split, 
                             verbose=verbose, save_name=save_name)
            self.plot_loss(figsize=figsize)
            self.encoder = Model(inputs=self.model.input, outputs=self.model.layers[15].output)
            print('-'*60) if self.verbose else None
            if self.return_data:
                return self.model, self.encoder, self.fit
        else:
            raise ValueError('pretrained must be either: "None" to make and train a model, or a valid path to a .keras model')

    def make_predictions(self, showfig=True, xlim=(-5,5)):
        self.y_train_pred = self.model.predict(self.X_train).squeeze().astype('float32')
        self.y_test_pred  = self.model.predict(self.X_test).squeeze().astype('float32')
        train_mse = mean_squared_error(self.y_train.squeeze().astype('float32'), self.y_train_pred)
        test_mse  = mean_squared_error(self.y_test.squeeze().astype('float32'), self.y_test_pred)
        if self.verbose:
            print('-'*50)
            print('X_train: {}  | y_train: {}'.format(self.X_train.shape, self.y_train.shape))
            print('X_test:  {}   | y_test:  {}'.format(self.X_test.shape, self.y_test.shape))   
            print('y_train_pred: {} | y_test_pred: {}'.format(self.y_train_pred.shape, self.y_test_pred.shape))
            print('-'*50)
            print('Train MSE: {:.4f} | Test MSE: {:.4f}'.format(train_mse, test_mse))
        if showfig:
            self.plot_predictions(train_or_test='train', xlim=xlim)
            self.plot_predictions(train_or_test='test', xlim=xlim)
            self.calc_ensemble_uq(showfig=showfig); self.plot_csh_pred('train'); self.plot_csh_pred('test')
        else:
            self.calc_ensemble_uq(showfig=showfig); self.predict_all(); self.calc_csh()
        print('-'*60) if self.verbose else None
        return None    

    '''
    Auxiliary functions
    '''
    def calc_dxdz(self, l):
        dxdz = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            dxdz[i,:] = np.gradient(l[i,:,1])
        return np.expand_dims(dxdz, axis=-1)
    
    def calc_autocorr(self, l, autocorr_mode, autocorr_method):
        ac = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            ac[i,:] = signal.correlate(l[i,:,1], l[i,:,1], mode=autocorr_mode, method=autocorr_method)
        return np.expand_dims(ac, axis=-1)
    
    def calc_detrend(self, l):
        dt = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            dt[i,:] = signal.detrend(l[i,:,1])
        return np.expand_dims(dt, axis=-1)
    
    def calc_fourier(self, l, fourier_window, fourier_scale):
        zfft = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            z = signal.zoom_fft(l[i,:,1], fourier_window)/fourier_scale
            zfft[i] = np.real(z) + np.imag(z)
        return np.expand_dims(zfft, axis=-1)

    def calc_hilbert(self, l):
        hilb = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            hilb[i,:] = np.abs(signal.hilbert(l[i,:,1]))
        return np.expand_dims(hilb, axis=-1)
    
    def calc_symiir(self, l, symiir_c0, symiir_z1):
        symiir = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            symiir[i,:] = signal.symiirorder1(l[i,:,1], symiir_c0, symiir_z1)
        return np.expand_dims(symiir, axis=-1)
    
    def calc_savgol(self, l, savgol_window, savgol_order):
        savgol = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            savgol[i,:] = signal.savgol_filter(l[i,:,1], savgol_window, savgol_order)
        return np.expand_dims(savgol, axis=-1)
    
    def calc_cspline(self, l, spline_lambda):
        spline = np.zeros((l.shape[0], l.shape[1]))
        for i in range(l.shape[0]):
            spline[i,:] = signal.cspline1d(l[i,:,1], lamb=spline_lambda)
        return np.expand_dims(spline, axis=-1)

    def calc_features(self):
        if self.dxdz:
            logs_dxdz = self.calc_dxdz(self.logs)
            self.logs = np.concatenate((self.logs, logs_dxdz), axis=-1)
            print('Well logs with Depth Derivative:', self.logs.shape) if self.verbose else None
        if self.autocorr:
            logs_ac   = self.calc_autocorr(self.logs_clean, self.autocorr_mode, self.autocorr_method)
            self.logs = np.concatenate((self.logs, logs_ac), axis=-1)
            print('Well logs with Autocorrelation:', self.logs.shape) if self.verbose else None
        if self.detrend:
            logs_detrend = self.calc_detrend(self.logs_clean)
            self.logs = np.concatenate((self.logs, logs_detrend), axis=-1)
            print('Well logs with Detrend Filter:', self.logs.shape) if self.verbose else None
        if self.fourier:
            logs_fourier = self.calc_fourier(self.logs_clean, self.fourier_window, self.fourier_scale)
            self.logs = np.concatenate((self.logs, logs_fourier), axis=-1)
            print('Well logs with Fourier Transform:', self.logs.shape) if self.verbose else None
        if self.hilbert:
            logs_hilb = self.calc_hilbert(self.logs_clean)
            self.logs = np.concatenate((self.logs, logs_hilb), axis=-1)
            print('Well logs with Hilbert Transform:', self.logs.shape) if self.verbose else None
        if self.symiir:
            logs_symiir = self.calc_symiir(self.logs_clean, self.symiir_c0, self.symiir_z1)
            self.logs = np.concatenate((self.logs, logs_symiir), axis=-1)
            print('Well logs with Symmetric IIR Filter:', self.logs.shape) if self.verbose else None
        if self.savgol:
            logs_savgol = self.calc_savgol(self.logs_clean, self.savgol_window, self.savgol_order)
            self.logs = np.concatenate((self.logs, logs_savgol), axis=-1)
            print('Well logs with Savitzky-Golay Filter:', self.logs.shape) if self.verbose else None
        if self.cspline:
            logs_cspline = self.calc_cspline(self.logs_clean, self.spline_lambda)
            self.logs = np.concatenate((self.logs, logs_cspline), axis=-1)
            print('Well logs with Cubic Spline:', self.logs.shape) if self.verbose else None
        if self.decimate:
            self.logs_clean = signal.decimate(self.logs_clean, q=self.decimate_q, axis=1)
            print('Well logs Decimated {}x: {}'.format(self.decimate_q, self.logs_clean.shape)) if self.verbose else None
        self.logs_clean = np.nan_to_num(self.logs, nan=0)
        return self.logs if self.return_data else None

    def datascaler(self, scaler=None, data=None):
        if scaler==None:
            scaler = self.scaler
        if data==None:
            data = self.logs_clean
        self.logs_norm = np.zeros_like(data)
        sd, mu, minvalue, maxvalue = {}, {}, {}, {}
        if scaler=='standard':
            for k in range(data.shape[-1]):
                df = data[...,k]
                sd[k] = np.nanstd(df)
                mu[k] = np.nanmean(df)
                self.logs_norm[...,k] = (df - mu[k]) / sd[k]
        elif scaler=='minmax':
            for k in range(data.shape[-1]):
                df = data[...,k]
                minvalue[k] = np.nanmin(df)
                maxvalue[k] = np.nanmax(df)
                self.logs_norm[...,k] = (df - minvalue[k]) / (maxvalue[k] - minvalue[k])
        elif scaler=='none':
            self.logs_norm = data
        else:
            raise ValueError('Invalid scaler. Choose a scaler from ("standard", "minmax" or "none")')
        self.scaler_values = {'sd':sd, 'mu':mu, 'min':minvalue, 'max':maxvalue}
        return self.logs_norm if self.return_data else None

    def make_nn(self, kernel_size:int=15, drop=0.2, depths=[16,32,64], in_channels:int=10):
        K.clear_session()
        def enc_layer(inp, units):
            _ = layers.Conv1D(units, kernel_size, padding='same')(inp)
            _ = layers.BatchNormalization()(_)
            _ = layers.ReLU()(_)
            _ = layers.Dropout(drop)(_)
            _ = layers.MaxPooling1D(2)(_)
            return _
        def dec_layer(inp, units):
            _ = layers.Conv1D(units, kernel_size, padding='same')(inp)
            _ = layers.BatchNormalization()(_)
            _ = layers.ReLU()(_)
            _ = layers.Dropout(drop)(_)
            _ = layers.UpSampling1D(2)(_)
            return _
        def residual_cat(in1, in2):
            _ = layers.ZeroPadding1D((0,1))(in2) #(1,0) 
            _ = layers.Concatenate()([in1, _])
            return _
        def out_layer(inp, units):
            _ = dec_layer(inp, units)
            _ = layers.Conv1D(1, kernel_size, padding='same', activation='linear')(_)
            _ = layers.ZeroPadding1D((0,1))(_) #(1,0)
            return _
        inputs  = layers.Input(shape=(None, in_channels))
        enc1    = enc_layer(inputs, depths[0])
        enc2    = enc_layer(enc1, depths[1])
        zlatent = enc_layer(enc2, depths[2])
        dec3    = residual_cat(enc2, dec_layer(zlatent, depths[1]))
        dec2    = residual_cat(enc1, dec_layer(dec3, depths[0]))
        outputs = out_layer(dec2, 4)
        self.model = Model(inputs, outputs, name='baseline_correction')
        return self.model if self.return_data else None

    def train_model(self, optimizer:str='adam', lr=1e-3, wd=1e-5, loss='mse', metrics=['mse'],
                    epochs:int=100, batch_size:int=32, valid_split=0.25, verbose:bool=True,
                    save_name:str='baseline_correction_model'):
        if optimizer=='adam':
            opt = optimizers.Adam(learning_rate=lr)
        elif optimizer=='adamw':
            opt = optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        elif optimizer=='sgd':
            opt = optimizers.SGD(learning_rate=lr)
        else:
            raise ValueError('optimizer must be "adam", "adamw" or "sgd"')
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        self.fit = self.model.fit(self.X_train, self.y_train,
                                epochs           = epochs,
                                batch_size       = batch_size,
                                validation_split = valid_split,
                                shuffle          = True,
                                verbose          = verbose)
        self.model.save('{}.keras'.format(save_name))
        return self.model, self.fit if self.return_data else None
    
    def plot_loss(self, figsize=(5,4)):
        plt.figure(figsize=figsize)
        plt.plot(self.fit.history['loss'], label='Training')
        plt.plot(self.fit.history['val_loss'], label='Validation')
        plt.title('Training Performance', weight='bold')
        plt.xlabel('Epochs', weight='bold'); plt.ylabel('Loss', weight='bold')
        plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
        plt.savefig('figures/Training_Performance.png', dpi=300) if self.save_fig else None
        plt.show()
        return None

    def plot_features(self, train_or_test:str='train', nrows:int=5, ncols:int=10, mult:int=1, figsize=(20,12),
                    feature_names=['Depth','SP','dxdz','AutoCorrelation','Detrend','FFT',
                                   'Hilbert','SymIIR','Savitzky-Golay','Cubic Splines'],
                    colors=['tab:gray','tab:red','tab:orange','tab:olive','tab:green',
                            'tab:cyan','tab:blue','tab:pink','magenta','tab:purple','tab:brown']):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        if train_or_test == 'train':
            x = self.X_train
            idx = x[...,0]
        elif train_or_test == 'test':
            x = self.X_test
            idx = x[...,0]
        else:
            raise ValueError('train_or_test must be "train" or "test"')
        for i in range(nrows):
            for j in range(ncols):
                k = i*mult
                axs[i,j].plot(x[k,:,j], idx[k], color=colors[j])
                axs[0,j].set_title(feature_names[j])
                axs[i,0].set_ylabel('{} {}'.format(train_or_test, k))
                axs[i,j].grid(True, which='both')
        fig.suptitle('{} Features'.format(train_or_test), weight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures/{}_features.png'.format(train_or_test), dpi=300) if self.save_fig else None
        plt.show()
        return None

    def plot_predictions(self, train_or_test:str='train', xlim=(-200,50), 
                         nrows:int=3, ncols:int=8, mult:int=1, figsize=(20,12)):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        if train_or_test=='train':
            x, y, yh = self.X_train, self.y_train, self.y_train_pred
        elif train_or_test=='test':
            x, y, yh = self.X_test, self.y_test, self.y_test_pred
        else:
            raise ValueError('train_or_test must be "train" or "test"')
        k = 0
        for i in range(nrows):
            for j in range(ncols):
                mask = ~np.isnan(x[k,:,0])
                idx,    xvalue  = x[k,:,0][mask], x[k,:,1][mask]
                yvalue, yhvalue = y[k][mask],     yh[k][mask]
                axs[i,j].plot(xvalue, idx, c='tab:purple', label='SP')
                axs[i,j].plot(yvalue, idx, c='darkmagenta', label='SP_NORM')
                axs[i,j].plot(yhvalue, idx, c='black', label='SP_NORM_PRED')
                axs[i,0].set_ylabel('Depth [ft]', weight='bold')
                axs[-1,j].set_xlabel('SP [mV]', weight='bold')
                axs[i,j].set_xlim(xlim)
                axs[i,j].grid(True, which='both')
                axs[i,j].invert_yaxis()
                k += mult
        fig.suptitle('{} Predictions'.format(train_or_test), weight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures/{}_predictions.png'.format(train_or_test), dpi=300) if self.save_fig else None
        plt.show()
        return None

    def plot_SP_and_NORM(self, data=None, short_title:str='clean', wellname:bool=False,
                         nrows:int=3, ncols:int=10, mult:int=1, figsize=(20,12), xlim=(-200,50)):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
        k = 0
        d = self.logs_clean if data is None else data
        for i in range(nrows):
            for j in range(ncols):
                idx, sp, sp_norm = d[k,:,0], d[k,:,1], d[k,:,2]
                axs[i,j].plot(sp,  idx, c='tab:purple', label='SP')
                axs[i,j].plot(sp_norm, idx, c='darkmagenta', label='SP_NORM')
                axs[i,j].set_title(os.listdir(self.folder)[k].split('.')[0], weight='bold') if wellname else None
                axs[i,0].set_ylabel('DEPTH [ft]', weight='bold')
                axs[-1,j].set_xlabel('SP/SP_NORM', weight='bold')
                axs[i,j].set_xlim(xlim)
                axs[i,j].grid(True, which='both')
                k += mult
        axs[0,0].invert_yaxis()
        fig.suptitle('{} dataset'.format(short_title), weight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures/SP_and_NORM_{}.png'.format(short_title), dpi=300) if self.save_fig else None
        plt.show()
        return None
    
    def calc_ensemble_uq(self, data=None, sample_log:int=5,
                         showfig:bool=True, figsize=(5,7), colors=['darkred','red']):
        if data is None:
            data, sample, index = self.logs[...,2], self.logs[sample_log,:,2], self.logs[sample_log,:,0]
        else:
            data, sample, index = data[...,2], data[sample_log,:,2], data[sample_log,:,0]
        lb = np.nanpercentile(data, self.bounds[0], axis=0)
        mu = np.nanpercentile(data, 50, axis=0)
        ub = np.nanpercentile(data, self.bounds[1], axis=0)
        if showfig:
            plt.figure(figsize=figsize)
            plt.plot(sample, index, 'darkmagenta', label='Sample Log (#{})'.format(sample_log))
            plt.plot(lb,     index, color=colors[0], label='P{}'.format(self.bounds[0]))
            plt.plot(mu,     index, color=colors[1], label='P50')
            plt.plot(ub,     index, color=colors[0], label='P{}'.format(self.bounds[1]))
            plt.fill_betweenx(index, lb, ub, color=colors[0], alpha=0.5)
            plt.xlabel('SP [mV]', weight='bold'); plt.ylabel('Depth [ft]', weight='bold')
            plt.title('Ensemble UQ', weight='bold')
            plt.legend(facecolor='lightgrey', edgecolor='k', fancybox=False, shadow=True)
            plt.grid(True, which='both')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('figures/ensemble_uq.png', dpi=300) if self.save_fig else None
            plt.show()
        self.ens_uq = {'lb':lb, 'mu':mu, 'ub':ub}
        return self.ens_uq if self.return_data else None

    def moving_window_percentile(self, arr, window_size=1001, percentile=10, mode='edge'):
        pad_width  = (window_size - 1) // window_size
        arr_padded = np.pad(arr, pad_width, mode=mode)
        result     = ndimage.percentile_filter(arr_padded, percentile, size=window_size)
        return np.min([arr, result], axis=0)

    def predict_all(self):
        self.X_all  = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y_all  = np.concatenate((self.y_train, self.y_test), axis=0)
        self.y_pred = np.concatenate((self.y_train_pred, self.y_test_pred), axis=0)
        return self.X_all, self.y_all, self.y_pred if self.return_data else None

    def calc_csh(self, data=None, window_size=1001, percentile=10, mode='edge'):
        data = (self.X_all, self.y_all, self.y_pred) if data is None else data
        idx, yt, yh = data[0], data[1], data[2]
        self.csh_linear = np.zeros((yh.shape[0], yh.shape[1]))
        self.csh_uncert = np.zeros((yh.shape[0], yh.shape[1]))
        self.csh_smooth = np.zeros((yh.shape[0], yh.shape[1]))
        self.sand_bl    = np.zeros((yh.shape[0], yh.shape[1]))
        for i in range(yh.shape[0]):
            d  = yh[i]
            lb = np.percentile(d, self.bounds[0])
            ub = np.percentile(d, self.bounds[1])
            z  = (d - lb) / (ub - lb)
            self.csh_linear[i] = (d - d.min()) / (d.max() - d.min())
            self.csh_uncert[i] = (z - z.min()) / (z.max() - z.min())
            self.sand_bl[i] = self.moving_window_percentile(d, window_size=window_size, percentile=percentile, mode=mode)
            self.csh_smooth[i] = (d - self.sand_bl[i]) / (d.max() - self.sand_bl[i])
        return self.csh_linear, self.csh_uncert, self.csh_smooth if self.return_data else None
    
    def plot_csh_pred(self, train_or_test:str='train',
                      window_size=1001, percentile=0.10, mode='edge',
                      showfig:bool=True, nrows:int=3, ncols:int=10, mult:int=1, x2lim=None, 
                      colors=['darkmagenta','tab:blue','tab:green','tab:red'], figsize=(20,12)):
        if train_or_test=='train':
            yh, idx = self.y_train_pred, self.X_train[...,0]
        elif train_or_test=='test':
            yh, idx = self.y_test_pred, self.X_test[...,0]
        else:
            raise ValueError('train_or_test must be "train" or "test"')
        csh_linear = np.zeros((yh.shape[0], yh.shape[1]))
        csh_uncert = np.zeros((yh.shape[0], yh.shape[1]))
        csh_smooth = np.zeros((yh.shape[0], yh.shape[1]))
        for i in range(yh.shape[0]):
            d = yh[i]
            lb = np.percentile(d, self.bounds[0])
            ub = np.percentile(d, self.bounds[1])
            csh_linear[i] = (d - d.min()) / (d.max() - d.min())
            z = (d - lb) / (ub - lb)
            csh_uncert[i] = (z - z.min()) / (z.max() - z.min())
            sand_bl = self.moving_window_percentile(d, window_size=window_size, percentile=percentile, mode=mode)
            csh_smooth[i] = (d - sand_bl) / (d.max() - sand_bl)
        if showfig:
            k = 0
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            for i in range(nrows):
                for j in range(ncols):
                    ax, ax2 = axs[i,j], axs[i,j].twiny()
                    ax.plot(yh[k], idx[k], color=colors[0])
                    ax2.plot(csh_linear[k], idx[k], color=colors[1])
                    ax2.plot(csh_uncert[k], idx[k], color=colors[2])
                    ax2.plot(csh_smooth[k], idx[k], color=colors[3])
                    ax2.set_xlim(x2lim)
                    axs[i,0].set_ylabel('Depth [ft]'); axs[-1,j].set_xlabel('SP_pred')
                    ax.grid(True, which='both'); ax.invert_yaxis()
                    k += mult
            fig.suptitle('{} $Csh$ estimation'.format(train_or_test), weight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig('figures/csh_uq_{}.png'.format(train_or_test), dpi=300) if self.save_fig else None
            plt.show()
        return self.csh if self.return_data else None
    
###########################################################################
################## TRANSFER LEARNING BASELINE CORRECTION ##################
###########################################################################
class TransferLearning(BaselineCorrection):
    def __init__(self):
        super().__init__()
        self.in_folder  = 'Data/UT Export 9-19'
        self.out_folder = 'Data/UT Export postprocess'
        self.model      = keras.models.load_model('baseline_correction_model.keras')
        self.plate      = crs.PlateCarree()
    
    def make_transfer_prediction(self, csh_method:str='sand-corrected'):
        files = os.listdir(self.in_folder)
        for file in tqdm(files, desc='Transfer Learning predictions', unit=' file(s)'):
            log_las = lasio.read('{}/{}'.format(self.in_folder, file))
            if 'SP' in log_las.curvesdict.keys():
                self.log_df = pd.DataFrame({'DEPT': log_las['DEPT'], 'SP': log_las['SP']})
                self.log    = np.nan_to_num(np.array(self.log_df), nan=0)
                self.calc_transfer_features(verbose=False)
                self.transfer_scaler()
                d = np.expand_dims(self.log_norm, 0)
                size = d.shape[1]
                if size != 44055:
                    d = np.pad(d, ((0,0),(0,44055-size),(0,0)), mode='constant', constant_values=0.)
                    d = layers.Masking(mask_value=0.)(d)
                self.sp_pred  = self.model.predict(d, verbose=0).squeeze().astype('float32')
                if size != 44055:
                    self.sp_pred = self.sp_pred[:size]
                self.csh_pred = self.calc_csh(csh_type=csh_method)
                self.transfer_inverse_scaler()
                log_las.append_curve('SP_PRED', self.log_, unit='mV', descr='Predicted SP from baseline correction')
                log_las.append_curve('CSH_PRED', self.csh_pred, unit='%', descr='Estimated Csh from predicted SP')
                log_las.write('{}/{}'.format(self.out_folder, file), version=2.0)
        return None
    
    def plot_transfer_results(self, filenum:int=100, figsize=(10,8), showfig:bool=True, 
                              add_title:bool=False, semilog1:bool=False):
        f = os.listdir(self.out_folder)[filenum]
        l = lasio.read('{}/{}'.format(self.out_folder,f))
        d = l.df()
        fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
        ax1, ax2, ax3 = axs
        if 'GR' in d.columns:
            mnem1 = 'GR'
            lb, ub = 0, 200
        elif 'RHOB' in d.columns:
            mnem1 = 'RHOB'
            lb, ub = 1.65, 2.65
        elif 'RHOZ' in d.columns:
            mnem1 = 'RHOZ'
            lb, ub = 1.65, 2.65
        else:
            mnem1 = d.columns[2]
            lb, ub = d[mnem1].min(), d[mnem1].max()
        unit1 = l.curvesdict[mnem1].unit
        self.plot_curve(ax1, d, mnem1, lb, ub, 'g', units=unit1, semilog=semilog1)
        ax21, ax22 = ax2.twiny(), ax2.twiny()
        self.plot_curve(ax2, d, 'SP', -200, 50, 'magenta', units='mV')
        self.plot_curve(ax21, d, 'SP_NORM', -200, 50, 'darkmagenta', units='mV', pad=1.08)
        self.plot_curve(ax22, d, 'SP_PRED', -200, 50, 'k', units='mV', pad=1.16)
        if 'VHS_GR' in d.columns:
            ax31, ax32 = ax3.twiny(), ax3.twiny()
            self.plot_curve(ax3, d, 'VSH_GR', 0, 1, 'lightgreen', units='/')
            self.plot_curve(ax31, d, 'VSH_SP', 0, 1, 'purple', units='/', pad=1.08)
            self.plot_curve(ax32, d, 'CSH_PRED', 0, 1, 'k', ls='--', units='/', pad=1.16)
        else:
            ax31 = ax3.twiny()
            self.plot_curve(ax3, d, 'VSH_SP', 0, 1, 'purple', units='/')
            self.plot_curve(ax31, d, 'CSH_PRED', 0, 1, 'k', ls='--', units='/', pad=1.08)
        fig.suptitle('Estimation Results | {}'.format(f.split('.')[0]), weight='bold', fontsize=14) if add_title else None
        ax1.set_ylabel('DEPTH [ft]', weight='bold')
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig('figures/estimation_well_{}.png'.format(f.split('.')[0]), dpi=300) if self.save_fig else None
        plt.show() if showfig else None
        return None

    '''
    Auxiliary functions
    '''
    def moving_window_percentile(self, arr, window_size=1001, percentile=10, mode='edge'):
        pad_width  = (window_size - 1) // window_size
        arr_padded = np.pad(arr, pad_width, mode=mode)
        result     = ndimage.percentile_filter(arr_padded, percentile, size=window_size)
        return np.min([arr, result], axis=0)

    def calc_csh(self, csh_type='sand-corrected', d=None, window_size=1001, percentile=10, mode='edge'):
        if d==None:
            d = self.sp_pred
        assert csh_type in ['raw', 'percentile', 'sand-corrected'], 'csh_type must be "raw", "percentile" or "sand-corrected"'
        csh_linear = (d - d.min()) / (d.max() - d.min())
        lb = np.percentile(self.sp_pred, self.bounds[0])
        ub = np.percentile(self.sp_pred, self.bounds[1])
        z          = (d-lb) / (ub-lb)
        csh_uncert = (z-z.min()) / (z.max()-z.min())
        sand_bl    = self.moving_window_percentile(d, window_size=window_size, percentile=percentile, mode=mode)
        csh_smooth = (d - sand_bl) / (d.max() - sand_bl)
        if csh_type=='raw':
            return csh_linear
        elif csh_type=='percentile':
            return csh_uncert
        elif csh_type=='sand-corrected':
            return csh_smooth
        else:
            raise ValueError('csh_type must be "raw", "percentile" or "sand-corrected"') 
    
    def transfer_scaler(self):
        sd, mu, minvalue, maxvalue = {}, {}, {}, {}
        self.log_norm = np.zeros_like(self.log)
        if self.scaler=='standard':
            for k in range(self.log.shape[-1]):
                df = self.log[...,k]
                sd[k] = np.nanstd(df)
                mu[k] = np.nanmean(df)
                self.log_norm[...,k] = (df - mu[k]) / sd[k]
        elif self.scaler=='minmax':
            for k in range(self.log.shape[-1]):
                df = self.log[...,k]
                minvalue[k] = np.nanmin(df)
                maxvalue[k] = np.nanmax(df)
                self.log_norm[...,k] = (df - minvalue[k]) / (maxvalue[k] - minvalue[k])
        elif self.scaler=='none':
            self.log_norm = self.log
        else:
            raise ValueError('Invalid scaler. Choose a scaler from ("standard", "minmax" or "none")')
        self.scaler_values = {'sd':sd, 'mu':mu, 'min':minvalue, 'max':maxvalue}
        return self.log_norm if self.return_data else None
    
    def transfer_inverse_scaler(self, inv_data=None):
        if inv_data == None:
            inv_data = self.sp_pred
        sd, mu = self.scaler_values['sd'], self.scaler_values['mu']
        minvalue, maxvalue = self.scaler_values['min'], self.scaler_values['max']
        if self.scaler=='standard':
            self.log_ = inv_data * sd[1] #+ mu[1]
        elif self.scaler=='minmax':
            self.log_ = inv_data * (maxvalue[1] - minvalue[1]) + minvalue[1]
        elif self.scaler=='none':
            self.log_ = inv_data
        else:
            raise ValueError('Invalid scaler. Choose a scaler from ("standard", "minmax" or "none")')
        return self.log_backtransform if self.return_data else None
    
    def calc_transfer_features(self, verbose:bool=False):  
        d = self.log[:,1]              
        if self.dxdz:
            log_dxdz = np.gradient(d)
            self.log = np.concatenate([self.log, np.expand_dims(log_dxdz,-1)], axis=-1)
            print('Log with Depth Derivative:', self.log.shape) if verbose else None
        if self.autocorr:
            log_ac   = signal.correlate(d, d, mode=self.autocorr_mode, method=self.autocorr_method)
            self.log = np.concatenate([self.log, np.expand_dims(log_ac,-1)], axis=-1)
            print('Log with Autocorrelation:', self.log.shape) if verbose else None
        if self.detrend:
            log_detrend = signal.detrend(d)
            self.log = np.concatenate([self.log, np.expand_dims(log_detrend,-1)], axis=-1)
            print('Log with Detrend filter:', self.log.shape) if verbose else None
        if self.fourier:
            log_fft  = signal.zoom_fft(d, self.fourier_window)/self.fourier_scale
            self.log = np.concatenate([self.log, np.expand_dims(log_fft,-1)], axis=-1)
            print('Log with Fourier Transform:', self.log.shape) if verbose else None
        if self.hilbert:
            log_hilbert = np.abs(signal.hilbert(d))
            self.log = np.concatenate([self.log, np.expand_dims(log_hilbert,-1)], axis=-1)
            print('Log with Hilbert Transform:', self.log.shape) if verbose else None
        if self.symiir:
            log_symiir = signal.symiirorder1(d, self.symiir_c0, self.symiir_z1)
            self.log = np.concatenate([self.log, np.expand_dims(log_symiir,-1)], axis=-1)
            print('Log with Symmetric IIR filter:', self.log.shape) if verbose else None
        if self.savgol:
            log_savgol = signal.savgol_filter(d, self.savgol_window, self.savgol_order)
            self.log = np.concatenate([self.log, np.expand_dims(log_savgol,-1)], axis=-1)
            print('Log with Savitzky-Golay filter:', self.log.shape) if verbose else None
        if self.cspline:
            log_cspline = signal.cspline1d(d, lamb=self.spline_lambda)
            self.log = np.concatenate([self.log, np.expand_dims(log_cspline,-1)], axis=-1)
            print('Log with Cubic Spline:', self.log.shape) if verbose else None
        if self.decimate:
            self.log_decimate = signal.decimate(d, q=self.decimate_q, axis=1)
            print('Well log Decimated {}x: {}'.format(self.decimate_q, self.log_decimate.shape)) if verbose else None


    def plot_curve(self, ax, df, curve, lb=0, ub=1, color='k', size=2, pad=1, mult=1,
                semilog=False, bar=False, units=None, alpha=None,
                marker=None, linestyle=None, fill=None, rightfill=False, **kwargs):
            '''
            subroutine to plot a curve on a given axis
            '''
            x, y = mult*df[curve], df.index
            if linestyle is None:
                linestyle = kwargs.get('ls', None)
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
    
    def plot_spatial_map(self, cmap='jet', vmin=0.0, vmax=0.35, method='linear'):
        '''
        incomplete! Needs work
        '''
        folder = 'Data/UT Export postprocess'
        files  = os.listdir(folder)
        lat, lon, csh = [], [], []
        for i, f in enumerate(files):
            l = lasio.read('{}/{}'.format(folder, f))
            lat.append(l.header['Well']['LAT'].value)
            lon.append(l.header['Well']['LON'].value)
            csh.append(l['CSH_PRED'])

        sweet_ratio = []
        for i in range(len(csh)):
            m = csh[i] < 0.6
            r = np.sum(m) / len(m)
            sweet_ratio.append(r)

        gx, gy = np.meshgrid(np.linspace(-98, -90.75, 100), np.linspace(26.5, 30.25, 100))
        gp = interpolate.griddata((lon, lat), sweet_ratio, (gx, gy), method=method)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection=plate)
        im1 = ax.scatter(lon, lat, c=sweet_ratio, s=30,
                        cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='k', lw=0.5, transform=plate, zorder=1)
        ax.coastlines(resolution='50m', color='black', lw=2, zorder=2)
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        cb = plt.colorbar(im1, ax=ax, shrink=0.365)
        cb.set_label('Proportion of Sweet Spots', rotation=270, labelpad=15, weight='bold', fontsize=12)
        ax.contour(gx, gy, gp, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.patch.set_facecolor('lightgrey')
        plt.tight_layout()
        plt.show()
        return None

###########################################################################
############################## MAIN ROUTINE ###############################
###########################################################################
if __name__ == '__main__':
    time0 = time.time()

    ### Log Analysis
    spl = SPLogAnalysis()
    spl.__dict__
    spl.plot_ccs_sand_wells(figsize=(8,3), value='POROSITY', cmap='jet')
    spl.plot_survey(figsize=(10,3))
    spl.plot_well(figsize=(10,8), curve='SP', order=(5,1,0))

    ### Automatic Baseline Correction
    blc = BaselineCorrection()
    blc.__dict__
    blc.load_logs(preload      = True,
                  preload_file = 'Data/log_data.npy',
                  folder       = 'Data/UT Export 9-19',
                  save_file    = 'Data/log_data.npy',
                  showfig      = True,
                  )    
    
    blc.scale_and_random_split(scaler    = 'standard', 
                               test_size = 0.227, 
                               showfig   = True,
                               )
    
    blc.make_model(pretrained   = 'baseline_correction_model.keras',
                   show_summary = True, 
                   kernel_size  = 15, 
                   dropout      = 0.2,
                   depths       = [16,32,64], 
                   optimizer    = 'adam',
                   lr           = 1e-3,
                   loss         = 'mse',
                   metrics      = ['mse'],
                   epochs       = 100,
                   batch_size   = 30,
                   valid_split  = 0.25,
                   verbose      = True,
                   figsize      = (10,5),
                   )
    
    blc.make_predictions(showfig=True,
                         xlim=(-5,5),
                         )

    ### Transfer Learning Baseline Correction
    tlc = TransferLearning()
    tlc.__dict__
    tlc.make_transfer_prediction()
    tlc.plot_transfer_results(filenum=0, 
                              figsize=(10,8), 
                              showfig=True,
                              )

    ### exit
    print('-'*60,'\n','Elapsed time: {:.3f} seconds'.format(time.time()-time0))

###########################################################################
################################## END ####################################
###########################################################################