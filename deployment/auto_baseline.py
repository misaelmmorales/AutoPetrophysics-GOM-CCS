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

import dask
import dask.delayed
import dask.array as da

import keras
import tensorflow as tf
from keras import Model
from keras import layers, optimizers
from keras import backend as K

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

def load_train_logs(folder:str=None, padded_length:int=75000):
    files = os.listdir(folder)
    logs_list, k = {}, 0
    for file in tqdm(files, desc='Processing Files', unit=' file(s)'):
        log = lasio.read('{}/{}'.format(folder,file))
        if 'SP_NORM' not in log.keys() or 'SP' not in log.keys():
            continue
        logs_list[k] = pd.DataFrame({'DEPT': log['DEPT'], 'SP': log['SP'], 'SP_NORM': log['SP_NORM']})
        k += 1
    logs = np.zeros((len(logs_list), padded_length, 3))
    for i in range(len(logs_list)):
        logs[i,logs_list[i].index,:] = logs_list[i].values
    logs = np.where(logs==0, np.nan, logs)
    logs_clean = np.nan_to_num(logs, nan=0)
    logs, logs_clean = calc_features(logs, logs_clean)
    print('Training well logs:', logs_clean.shape)
    return logs, logs_clean

def load_train_logs_parallel(folder:str=None, padded_length:int=75000):
    files = os.listdir(folder)
    logs_list = []
    @dask.delayed
    def load_file(file):
        log = lasio.read(os.path.join(folder, file))
        if 'SP_NORM' in log.keys() and 'SP' in log.keys():
            logs_list.append(pd.DataFrame({'DEPT': log['DEPT'], 'SP': log['SP'], 'SP_NORM': log['SP_NORM']}))
    load_tasks = [load_file(file) for file in files]
    dask.compute(*load_tasks)    
    logs = da.zeros((len(logs_list), padded_length, 3), chunks=(1, padded_length, 3))
    for i, log_df in enumerate(logs_list):
        logs[i, log_df.index, :] = log_df.values
    logs = da.where(logs == 0, da.nan, logs)
    logs_clean = da.nan_to_num(logs, nan=0)
    logs, logs_clean = calc_features(logs, logs_clean)
    print('Training well logs:', logs_clean.shape)
    return logs, logs_clean

def calc_dxdz(l):
    dxdz = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        dxdz[i,:] = np.gradient(l[i,:,1])
    return np.expand_dims(dxdz, axis=-1)

def calc_autocorr(l, autocorr_mode, autocorr_method):
    ac = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        ac[i,:] = signal.correlate(l[i,:,1], l[i,:,1], mode=autocorr_mode, method=autocorr_method)
    return np.expand_dims(ac, axis=-1)

def calc_detrend(l):
    dt = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        dt[i,:] = signal.detrend(l[i,:,1])
    return np.expand_dims(dt, axis=-1)

def calc_fourier(l, fourier_window, fourier_scale):
    zfft = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        z = signal.zoom_fft(l[i,:,1], fourier_window)/fourier_scale
        zfft[i] = np.real(z) + np.imag(z)
    return np.expand_dims(zfft, axis=-1)

def calc_hilbert(l):
    hilb = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        hilb[i,:] = np.abs(signal.hilbert(l[i,:,1]))
    return np.expand_dims(hilb, axis=-1)

def calc_symiir(l, symiir_c0, symiir_z1):
    symiir = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        symiir[i,:] = signal.symiirorder1(l[i,:,1], symiir_c0, symiir_z1)
    return np.expand_dims(symiir, axis=-1)

def calc_savgol(l, savgol_window, savgol_order):
    savgol = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        savgol[i,:] = signal.savgol_filter(l[i,:,1], savgol_window, savgol_order)
    return np.expand_dims(savgol, axis=-1)

def calc_cspline(l, spline_lambda):
    spline = np.zeros((l.shape[0], l.shape[1]))
    for i in range(l.shape[0]):
        spline[i,:] = signal.cspline1d(l[i,:,1], lamb=spline_lambda)
    return np.expand_dims(spline, axis=-1)

def calc_features(logs, logs_clean):
    logs_dxdz = calc_dxdz(logs)
    logs = np.concatenate((logs, logs_dxdz), axis=-1)
    logs_ac = calc_autocorr(logs_clean, 'same', 'fft')
    logs = np.concatenate((logs, logs_ac), axis=-1)
    logs_detrend = calc_detrend(logs_clean)
    logs = np.concatenate((logs, logs_detrend), axis=-1)
    logs_fourier = calc_fourier(logs_clean, [1e-3,0.025], 1e3)
    logs = np.concatenate((logs, logs_fourier), axis=-1)
    logs_hilb = calc_hilbert(logs_clean)
    logs = np.concatenate((logs, logs_hilb), axis=-1)
    logs_symiir = calc_symiir(logs_clean, 0.5, 0.1)
    logs = np.concatenate((logs, logs_symiir), axis=-1)
    logs_savgol = calc_savgol(logs_clean, 15, 2)
    logs = np.concatenate((logs, logs_savgol), axis=-1)
    logs_cspline = calc_cspline(logs_clean, 0.0)
    logs = np.concatenate((logs, logs_cspline), axis=-1)
    logs_clean = np.nan_to_num(logs, nan=0)
    return logs, logs_clean

def datascaler(data):
    logs_norm = np.zeros_like(data)
    sd, mu = {}, {}
    for k in range(data.shape[-1]):
        df = data[...,k]
        sd[k] = np.nanstd(df)
        mu[k] = np.nanmean(df)
        logs_norm[...,k] = (df - mu[k]) / sd[k]
    scaler_values = {'sd': sd, 'mu': mu}
    return logs_norm, scaler_values

def make_nn(kernel_size:int=15, drop=0.2, depths=[16,32,64], in_channels:int=10):
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
        _ = layers.Concatenate()([in1, in2])
        return _
    def out_layer(inp, units):
        _ = dec_layer(inp, units)
        _ = layers.Conv1D(1, kernel_size, padding='same', activation='linear')(_)
        return _
    inputs  = layers.Input(shape=(None, in_channels))
    enc1    = enc_layer(inputs, depths[0])
    enc2    = enc_layer(enc1, depths[1])
    zlatent = enc_layer(enc2, depths[2])
    dec3    = residual_cat(enc2, dec_layer(zlatent, depths[1]))
    dec2    = residual_cat(enc1, dec_layer(dec3, depths[0]))
    outputs = out_layer(dec2, 4)
    return Model(inputs, outputs, name='baseline_correction_bigpad')

def plot_loss(fit, figsize=(5,4), savefig=False):
    plt.figure(figsize=figsize)
    plt.plot(fit.history['loss'], label='Training')
    plt.plot(fit.history['val_loss'], label='Validation')
    plt.title('Training Performance', weight='bold')
    plt.xlabel('Epochs', weight='bold'); plt.ylabel('Loss', weight='bold')
    plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
    plt.savefig('figures/Training_Performance.png', dpi=300) if savefig else None
    plt.show()
    return None