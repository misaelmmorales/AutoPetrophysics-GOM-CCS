import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lasio
from tqdm import tqdm
from scipy import stats, signal, linalg, fft
from statsmodels.tsa.arima.model import ARIMA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

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
    
    def read_all_headers(self):
        '''
        Read all headers one-by-one for all logs in the folder to identify repeated
        and unique curves. This will help in identifying the most common curves and 
        fixing multiple mnemonics for the same curve.
        '''
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
            '''
            subroutine to plot a curve on a given axis
            '''
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

    def plot_survey(self, survey=None, figsize=(10,5), showcols:bool=False):
        '''
        Plot the directional survey from 'DATA/UT dir surveys'
        '''
        if survey==None:
            fname = '427064023000_DIRSUR_NAD27(USFEET)US-SPC27-EXACT(TX-27SC).TXT'
            survey = pd.read_csv('Data/UT dir surveys/{}'.format(fname), skiprows=3, sep='\s+')
        wname = fname.split('.')[0].split('_')[0]
        print('DF Columns:', survey.columns.values) if showcols else None
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d', elev=40, azim=-45, aspect='equal')
        ax.scatter(survey['X(FT)'], survey['Y(FT)'], survey['MD(FT)'], s=5)
        ax.set_xlabel('X (ft)', weight='bold'); ax.set_ylabel('Y (ft)', weight='bold'); ax.set_zlabel('MD (ft)', weight='bold')
        ax.set(xlim3d=(0,500), ylim3d=(-500,0), zlim3d=(0,7000))
        ax.invert_zaxis()
        ax.set_title('{}'.format(wname), weight='bold')
        plt.tight_layout()
        plt.savefig('figures/Dir_Survey_{}.png'.format(wname), dpi=300) if self.save_fig else None
        plt.show()
        return None

    def plot_well(self, well_name:str, figsize=(10,8), fig2size=(10,3), curve='SP', order=(5,1,0), fig3size=(10,4)):
        '''
        Full well log plot with tracks for each curve
        '''
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
        self.folder = 'Data/UT Export 9-19/'
        self.return_data = False
        self.verbose     = True
        self.save_fig    = True
        self.check_tf_gpu()
    
    def check_tf_gpu(self):
        sys_info = tf.sysconfig.get_build_info()
        if self.verbose:
            print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
            print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
            print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
            print(tf.config.list_physical_devices()[0],'\n', tf.config.list_physical_devices()[1])
        return None

    def load_logs(self, folder='Data/UT Export 9-19/', preload:bool=True, showfig=True,
                  decimate:bool=False, decimate_q:int=10,
                  dxdz:bool=True,      hilbert:bool=True, detrend:bool=True,
                  fourier:bool=True,   fourier_window=[1e-3,0.025], fourier_scale=1e3,
                  symiir:bool=True,    symiir_c0=0.5, symiir_z1=0.1,
                  savgol:bool=True,    savgol_window=15, savgol_order=2,
                  cspline:bool=True,   spline_lambda=0.0,
                  autocorr:bool=True,  autocorr_method='fft', autocorr_mode='same'):
        '''
        Load all logs. 
            If preload=False: 
                This function will read over each LAS file in the folder and extract the 
                SP and SP_NORM curves, and then save then into a single NumPy array, along with a 
                clean version that removes NaNs.
            If preload=True: 
                This function will load the logs from the saved NumPy file and creates a clean version too.
        '''
        if preload:
            self.logs = np.load('Data/log_data.npy')
            print(self.logs.shape) if self.verbose else None
            logs_clean = np.nan_to_num(self.logs, nan=0)
        else:
            logs_list = {}
            files = os.listdir(folder)
            k = 0
            for file in tqdm(files, desc='Processing Files', unit='file'):
                log = lasio.read('{}/{}'.format(folder,file))
                if 'SP' in log.curvesdict.keys() and 'SP_NORM' in log.curvesdict.keys():
                    logs_list[k] = pd.DataFrame({'DEPT': log['DEPT'], 'SP': log['SP'], 'SP_NORM': log['SP_NORM']})
                    k += 1
            logs = np.zeros((len(logs_list),44055,3))
            for i in range(len(logs_list)):
                logs[i,logs_list[i].index,:] = logs_list[i].values
            self.logs = np.where(logs==0, np.nan, logs)
            np.save('Data/log_data.npy', self.logs)
            print(self.logs.shape) if self.verbose else None
            logs_clean = np.nan_to_num(self.logs, nan=0)
        if dxdz:
            def calc_dxdz(l):
                dxdz = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    dxdz[i,:] = np.gradient(l[i,:,1])
                return np.expand_dims(dxdz, axis=-1)
            logs_dxdz = calc_dxdz(self.logs)
            self.logs = np.concatenate((self.logs, logs_dxdz), axis=-1)
            print('Well logs with Depth Derivative:', self.logs.shape) if self.verbose else None
        if autocorr:
            def calc_autocorr(l):
                ac = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    ac[i,:] = signal.correlate(l[i,:,1], l[i,:,1], mode=autocorr_mode, method=autocorr_method)
                return np.expand_dims(ac, axis=-1)
            logs_ac = calc_autocorr(logs_clean)
            self.logs = np.concatenate((self.logs, logs_ac), axis=-1)
            print('Well logs with Autocorrelation:', self.logs.shape) if self.verbose else None
        if detrend:
            def calc_detrend(l):
                dt = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    dt[i,:] = signal.detrend(l[i,:,1])
                return np.expand_dims(dt, axis=-1)
            logs_detrend = calc_detrend(logs_clean)
            self.logs = np.concatenate((self.logs, logs_detrend), axis=-1)
            print('Well logs with Detrend Filter:', self.logs.shape) if self.verbose else None
        if fourier:
            def calc_fourier(l):
                zfft = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    z = signal.zoom_fft(l[i,:,1], fourier_window)/fourier_scale
                    zfft[i] = np.real(z) + np.imag(z)
                return np.expand_dims(zfft, axis=-1)
            logs_fourier = calc_fourier(logs_clean)
            self.logs = np.concatenate((self.logs, logs_fourier), axis=-1)
            print('Well logs with Fourier Transform:', self.logs.shape) if self.verbose else None
        if hilbert:
            def calc_hilbert(l):
                hilb = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    hilb[i,:] = np.abs(signal.hilbert(l[i,:,1]))
                return np.expand_dims(hilb, axis=-1)
            logs_hilb = calc_hilbert(logs_clean)
            self.logs = np.concatenate((self.logs, logs_hilb), axis=-1)
            print('Well logs with Hilbert Transform:', self.logs.shape) if self.verbose else None
        if symiir:
            def calc_symiir(l):
                symiir = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    symiir[i,:] = signal.symiirorder1(l[i,:,1], symiir_c0, symiir_z1)
                return np.expand_dims(symiir, axis=-1)
            logs_symiir = calc_symiir(logs_clean)
            self.logs = np.concatenate((self.logs, logs_symiir), axis=-1)
            print('Well logs with Symmetric IIR Filter:', self.logs.shape) if self.verbose else None
        if savgol:
            def calc_savgol(l):
                savgol = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    savgol[i,:] = signal.savgol_filter(l[i,:,1], savgol_window, savgol_order)
                return np.expand_dims(savgol, axis=-1)
            logs_savgol = calc_savgol(logs_clean)
            self.logs = np.concatenate((self.logs, logs_savgol), axis=-1)
            print('Well logs with Savitzky-Golay Filter:', self.logs.shape) if self.verbose else None
        if cspline:
            def calc_cspline(l):
                spline = np.zeros((l.shape[0], l.shape[1]))
                for i in range(l.shape[0]):
                    spline[i,:] = signal.cspline1d(l[i,:,1], lamb=spline_lambda)
                return np.expand_dims(spline, axis=-1)
            logs_cspline = calc_cspline(logs_clean)
            self.logs = np.concatenate((self.logs, logs_cspline), axis=-1)
            print('Well logs with Cubic Spline:', self.logs.shape) if self.verbose else None
        self.logs_clean = np.nan_to_num(self.logs, nan=0)
        self.plot_SP_and_NORM(self.logs, short_title='raw') if showfig else None
        self.plot_SP_and_NORM(self.logs_clean, short_title='clean') if showfig else None
        if decimate:
            self.logs_clean = signal.decimate(self.logs_clean, q=decimate_q, axis=1)
            print('Well logs decimated by a factor of {}: {}'.format(decimate_q, self.logs_clean.shape)) if self.verbose else None
        if self.return_data:
            return self.logs, self.logs_clean
   
    def scale_and_random_split(self, scaler:str='none', test_size=0.227, showfig=True):
        self.logs_norm = np.zeros_like(self.logs_clean)
        sd, mu, minvalue, maxvalue = {}, {}, {}, {}
        if scaler=='standard':
            for k in range(self.logs_clean.shape[-1]):
                df = self.logs_clean[...,k]
                sd[k] = np.nanstd(df)
                mu[k] = np.nanmean(df)
                self.logs_norm[...,k] = (df - mu[k]) / sd[k]
        elif scaler=='minmax':
            for k in range(self.logs_clean.shape[-1]):
                df = self.logs_clean[...,k]
                minvalue[k] = np.nanmin(df)
                maxvalue[k] = np.nanmax(df)
                self.logs_norm[...,k] = (df - minvalue[k]) / (maxvalue[k] - minvalue[k])
        elif scaler=='none':
            self.logs_norm = self.logs_clean
        else:
            raise ValueError('scaler must be "standard", "minmax" or "none"')
        self.plot_SP_and_NORM(self.logs_norm, short_title='normalized', xlim=(-5,5)) if showfig else None
        self.scaler_values = {'sd':sd, 'mu':mu, 'min':minvalue, 'max':maxvalue}
        x = np.delete(self.logs_norm, 2, axis=-1)
        y = np.expand_dims(self.logs_norm[...,2], -1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size)
        if self.verbose:
            print('X_train: {} | X_test: {}'.format(self.X_train.shape, self.X_test.shape))
            print('y_train: {} | y_test: {}'.format(self.y_train.shape, self.y_test.shape))
        self.train_test_data = {'X_train':self.X_train, 'X_test':self.X_test, 
                                'y_train':self.y_train, 'y_test':self.y_test}
        return self.train_test_data if self.return_data else None
    
    def make_model(self, pretrained=None, show_summary:bool=False,
                   kernel_size=15, dropout=0.2, depths=[16,32,64],
                   optimizer='adam', lr=1e-3, loss='mse', metrics='mse', 
                   epochs=100, batch_size=32, valid_split=0.24, 
                   save_name='baseline_correction_model', figsize=(10,5)):
        if pretrained != None:
            self.model = keras.models.load_model(pretrained)
            self.encoder = Model(inputs=self.model.input, outputs=self.model.layers[15].output)
            self.model.summary() if show_summary else None
            print('-'*50,'\n','# Parameters: {:,}'.format(self.model.count_params())) if self.verbose else None
            if self.return_data:
                return self.model, self.encoder
        elif pretrained == None:
            self.make_nn(kernel_size=kernel_size, drop=dropout, depths=depths)
            print('-'*50,'\n','# Parameters: {:,}'.format(self.model.count_params())) if self.verbose else None
            self.model.summary() if show_summary else None
            self.train_model(optimizer=optimizer, lr=lr, loss=loss, metrics=metrics, 
                             epochs=epochs, batch_size=batch_size, valid_split=valid_split, 
                             save_name=save_name)
            self.plot_loss(figsize=figsize)
            self.encoder = Model(inputs=self.model.input, outputs=self.model.layers[15].output)
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
            self.calc_ensemble_uq(); self.plot_csh_pred('train'); self.plot_csh_pred('test')
        return None
        
    '''
    Auxiliary functions
    '''
    def make_nn(self, kernel_size=15, drop=0.2, depths=[16,32,64]):
        K.clear_session()
        ndata, nchannels = self.X_train.shape[1], self.X_train.shape[2]
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
            _ = layers.ZeroPadding1D((1,0))(in2)
            _ = layers.Concatenate()([in1, _])
            return _
        def out_layer(inp, units):
            _ = dec_layer(inp, units)
            _ = layers.ZeroPadding1D((1,0))(_)
            _ = layers.Conv1D(1, kernel_size, padding='same', activation='linear')(_)
            return _
        inputs  = layers.Input(shape=(ndata, nchannels))
        enc1    = enc_layer(inputs, depths[0])
        enc2    = enc_layer(enc1,   depths[1])
        zlatent = enc_layer(enc2,   depths[2])
        dec3    = residual_cat(enc2, dec_layer(zlatent, depths[1]))
        dec2    = residual_cat(enc1, dec_layer(dec3,    depths[0]))
        outputs = out_layer(dec2, 4)
        self.model = Model(inputs, outputs)
        return self.model if self.return_data else None
    
    def train_model(self, optimizer='adam', lr=1e-3, loss='mse', metrics='mse',
                    epochs=100, batch_size=32, valid_split=0.25, 
                    save_name='baseline_correction_model'):
        if optimizer=='adam':
            opt = optimizers.Adam(learning_rate=lr)
        elif optimizer=='adamw':
            opt = optimizers.AdamW(learning_rate=lr)
        elif optimizer=='sgd':
            opt = optimizers.SGD(learning_rate=lr)
        self.model.compile(optimizer=opt, loss=loss, metrics=[metrics])
        self.fit = self.model.fit(self.X_train, self.y_train,
                                epochs           = epochs,
                                batch_size       = batch_size,
                                validation_split = valid_split,
                                shuffle          = True,
                                verbose          = False)
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

    def plot_predictions(self, train_or_test:str='train', xlim=(-200,50), nrows=3, ncols=8, mult=1, figsize=(20,12)):
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

    def plot_SP_and_NORM(self, data=None, nrows=3, ncols=10, mult=1, figsize=(20,12), xlim=(-200,50), short_title:str='clean'):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
        k = 0
        d = self.logs_clean if data is None else data
        for i in range(nrows):
            for j in range(ncols):
                idx, sp, sp_norm = d[k,:,0], d[k,:,1], d[k,:,2]
                axs[i,j].plot(sp,  idx, c='tab:purple', label='SP')
                axs[i,j].plot(sp_norm, idx, c='darkmagenta', label='SP_NORM')
                axs[i,j].set_title(os.listdir(self.folder)[k].split('.')[0], weight='bold')
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
    
    def calc_ensemble_uq(self, data=None, bounds=[10,90], showfig:bool=True, sample_log:int=5, figsize=(5,7), colors=['darkred','red']):
        if data is None:
            data, sample, index = self.logs[...,2], self.logs[sample_log,:,2], self.logs[sample_log,:,0]
        else:
            data, sample, index = data[...,2], data[sample_log,:,2], data[sample_log,:,0]
        lb = np.nanpercentile(data, bounds[0], axis=0)
        mu = np.nanpercentile(data, 50, axis=0)
        ub = np.nanpercentile(data, bounds[1], axis=0)
        if showfig:
            plt.figure(figsize=figsize)
            plt.plot(sample, index, 'darkmagenta', label='Sample Log (#{})'.format(sample_log))
            plt.plot(lb,     index, color=colors[0], label='P{}'.format(bounds[0]))
            plt.plot(mu,     index, color=colors[1], label='P50')
            plt.plot(ub,     index, color=colors[0], label='P{}'.format(bounds[1]))
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
    
    def plot_csh_pred(self, train_or_test:str='train', bounds=[10,90],
                      showfig:bool=True, nrows=3, ncols=10, mult:int=1, x2lim=None, 
                      colors=['darkmagenta','tab:blue','tab:green'], figsize=(20,12)):
        if train_or_test=='train':
            yh, idx = self.y_train_pred, self.X_train[...,0]
        elif train_or_test=='test':
            yh, idx = self.y_test_pred, self.X_test[...,0]
        else:
            raise ValueError('train_or_test must be "train" or "test"')
        csh_linear = np.zeros((yh.shape[0], yh.shape[1]))
        csh_uncert = np.zeros((yh.shape[0], yh.shape[1]))
        for i in range(yh.shape[0]):
            d = yh[i]
            lb = np.percentile(d, bounds[0])
            ub = np.percentile(d, bounds[1])
            csh_linear[i] = (d - d.min()) / (d.max() - d.min())
            z = (d - lb) / (ub - lb)
            csh_uncert[i] = (z - z.min()) / (z.max() - z.min())
        if showfig:
            k = 0
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            for i in range(nrows):
                for j in range(ncols):
                    ax, ax2 = axs[i,j], axs[i,j].twiny()
                    ax.plot(yh[k], idx[k], color=colors[0])
                    ax2.plot(csh_linear[k], idx[k], color=colors[1])
                    ax2.plot(csh_uncert[k], idx[k], color=colors[2])
                    ax2.set_xlim(x2lim)
                    axs[i,0].set_ylabel('Depth [ft]'); axs[-1,j].set_xlabel('SP_pred')
                    ax.grid(True, which='both'); ax.invert_yaxis()
                    k += mult
            fig.suptitle('{} $C_{sh}$ estimation'.format(train_or_test), weight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig('figures/csh_uq_{}.png'.format(train_or_test), dpi=300) if self.save_fig else None
            plt.show()
        return self.csh if self.return_data else None

###########################################################################
############################## MAIN ROUTINE ###############################
###########################################################################

if __name__ == '__main__':

    ### Log analysis
    spl = SPLogAnalysis()
    spl.plot_ccs_sand_wells()
    spl.plot_survey()
    spl.plot_well('17700004060000')

    ### Baseline Correction
    blc = BaselineCorrection()
    blc.load_logs(preload=True)
    blc.scale_and_random_split(scaler='standard')
    blc.make_model(pretrained=None)
    blc.make_predictions()

###########################################################################
################################## END ####################################
###########################################################################