import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lasio
import os

from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from scipy.linalg import svd as SVD
from pywt import dwt
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.python.client import device_lib

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    cuda_version, cudnn_version = sys_info['cuda_version'], sys_info['cudnn_version']
    num_gpu_avail = len(tf.config.experimental.list_physical_devices('GPU'))
    gpu_name = device_lib.list_local_devices()[1].physical_device_desc[17:40]
    print('... Checking Tensorflow Version ...')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print("TF: {} | CUDA: {} | cuDNN: {}".format(tf.__version__, cuda_version, cudnn_version))
    print('# GPU available: {} ({})'.format(num_gpu_avail, gpu_name))
    #print(tf.config.list_physical_devices())
    return None

def load_data(LAS_folder, runs_file):
    ### Load Well Logs (LAS)
    df = {}
    count = 0
    for file in os.listdir(LAS_folder):
        if file.endswith('.las') or file.endswith('.LAS'):
            df[count] = lasio.read(os.path.join('log_data',file))
            count += 1
    ### Load all Mnemonics and Well Titles     
    mnemonics, well_title = {}, {}
    for i in range(len(df)):
        mnemonics[i] = list(df[i].curvesdict.keys())
        well_title[i] = list(df[i].well)
    ### Load Log Runs data
    well_runs_all = pd.read_excel(runs_file)
    well_runs = {}
    for i in range(len(df)):
        well_runs[i] = well_runs_all[well_runs_all['FILENAME'].astype('str')==df[i].header['Well']['API'].value+'00']
    ### Make DataFrames for each Well
    wdata = {}
    for i in range(len(df)):
        wdata[i] = pd.DataFrame(columns=mnemonics[i])
        for k in range(len(mnemonics[i])):
            wdata[i][mnemonics[i][k]] = np.nan_to_num(df[i][mnemonics[i][k]])
    return df, mnemonics, well_title, well_runs, wdata

def plot_k_suites(data, mnemo, k_start, k_end, figsize=(40,5), plot_mean=False):
    for k in range(k_start,k_end+1):
        plt.figure(figsize=figsize, facecolor='white')
        for p in range(1,len(mnemo[k])):
            plt.suptitle(str(k)+' | '+'Field='+data[k].well['FLD'].value+' | Name='+data[k].well['Well'].value+' | API='+data[k].well['API'].value)
            plt.subplot(1, len(mnemo[k]), p+1)
            plt.plot(data[k][mnemo[k][p]], data[k]['DEPT'], '.', markersize=2)
            plt.title(data[k].curvesdict[mnemo[k][p]].descr)
            plt.xlabel('{} [{}]'.format(data[k].curvesdict[mnemo[k][p]].mnemonic, data[k].curvesdict[mnemo[k][p]].unit))
            if plot_mean:
                plt.vlines(data[k][mnemo[k][p]][~np.isnan(data[k][mnemo[k][p]])].mean(), data[k]['DEPT'].min(), data[k]['DEPT'].max(), color='r')
            plt.gca().invert_yaxis(); plt.grid('on')
        plt.show()

def plot_well_and_runs(choose_well, df, well_runs, top_bottom=['TOPL_Fix','BOTL_Fix'], window=500, figsize=(10,5), bbox=(1.1,0.4)):
    mnem, mnem_keys = {}, df[choose_well].curvesdict.keys()
    for i in range(len(mnem_keys)):
        if list(mnem_keys)[i].startswith('GR'):
            mnem[i] = list(mnem_keys)[i]
    field = df[choose_well].header['Well']['FLD'].value
    well  = df[choose_well].header['Well']['WELL'].value.split()[0]
    api   = df[choose_well].header['Well']['API'].value
    roll_mean = pd.DataFrame(df[choose_well][list(mnem.values())[0]]).rolling(window, center=True).mean()
    plt.figure(figsize=figsize)
    for i in range(len(mnem)):
        plt.plot(df[choose_well]['DEPT'], df[choose_well][list(mnem.values())[i]], alpha=0.8, label=[list(mnem.values())[i]])
    plt.plot(df[choose_well]['DEPT'], roll_mean, c='k', label='Rolling Mean')
    for i in range(len(np.asarray(well_runs[choose_well][top_bottom[0]]).reshape(-1,1))):
        if not np.isnan(np.asarray(well_runs[choose_well][top_bottom[0]]).reshape(-1,1)[i]):
            plt.vlines(well_runs[choose_well][top_bottom[0]], 20, 100, colors='g', label='Run Top {}'.format(i))
            plt.vlines(well_runs[choose_well][top_bottom[1]], 20, 100, colors='r', label='Run Bot {}'.format(i))  
    plt.xlabel('DEPT'); plt.ylabel(list(mnem.values())[0][:2])
    plt.title('k: {} | Field: {} | Well: {} | API: {} | window: {}'.format(choose_well,field,well,api,window))
    plt.legend(loc='lower center', bbox_to_anchor=bbox); plt.grid('on')
    plt.show()

def plot_latent_projections(data, choose_well, zsvd, zpca, zdwt, figsize=(25,4)):
    field = data[choose_well].well['FLD'].value
    wname = data[choose_well].well['Well'].value.split()[0]
    api   = data[choose_well].well['API'].value
    plt.figure(figsize=figsize)
    plt.suptitle('#{} | {} , {} | API: {}'.format(choose_well, field, wname, api))
    plt.subplot(141)
    plt.plot(data[choose_well]['DEPT'], data[choose_well]['GR'])
    plt.xlabel('DEPT'); plt.ylabel('GR'); plt.grid(); plt.title('GR vs. Depth')
    plt.subplot(142)
    plt.scatter(zsvd[:,0], zsvd[:,1], s=1, alpha=0.5)
    plt.xlabel('$SVD_1$'); plt.ylabel('$SVD_2$'); plt.grid(); plt.title('SVD Projection')
    plt.subplot(143)
    plt.scatter(zpca[:,0], zpca[:,1], s=1, alpha=0.5)
    plt.xlabel('$PCA_1$'); plt.ylabel('$PCA_2$'); plt.grid(); plt.title('PCA Projection')
    plt.subplot(144)
    plt.scatter(zdwt[:,0], zdwt[:,1], s=1, alpha=0.5)
    plt.xlabel('$DWT_1$'); plt.ylabel('$DWTD_2$'); plt.grid(); plt.title('DWT Projection')
    plt.show()

def plot_normalized_latent_projections(data, choose_well, xdata, mnem, zsvd_n, zpca_n, zdwt_n, figsize=(25,4)):
    field = data[choose_well].well['FLD'].value
    wname = data[choose_well].well['Well'].value.split()[0]
    api   = data[choose_well].well['API'].value
    plt.figure(figsize=figsize)
    plt.suptitle('#{} | {} , {} | API: {}'.format(choose_well, field, wname, api))
    plt.subplot(141)
    plt.plot(xdata['DEPT'], xdata[mnem])
    plt.xlabel('DEPT'); plt.ylabel('GR'); plt.grid(); plt.title('GR vs. Depth')
    plt.subplot(142)
    plt.scatter(zsvd_n[:,0], zsvd_n[:,1], s=1, alpha=0.5)
    plt.xlabel('$SVD_1$'); plt.ylabel('$SVD_2$'); plt.grid(); plt.title('Normalized SVD Projection')
    plt.subplot(143)
    plt.scatter(zpca_n[:,0], zpca_n[:,1], s=1, alpha=0.5)
    plt.xlabel('$PCA_1$'); plt.ylabel('$PCA_2$'); plt.grid(); plt.title('Normalized PCA Projection')
    plt.subplot(144)
    plt.scatter(zdwt_n[:,0], zdwt_n[:,1], s=1, alpha=0.5)
    plt.xlabel('$DWT_1$'); plt.ylabel('$DWTD_2$'); plt.grid(); plt.title('normalized DWT Projection')
    plt.show()

def compute_latent_space(data, choose_well, wavelet='db1', dwtmode='per', normalize=False):
    if normalize:
        data_norm = pd.DataFrame(MinMaxScaler().fit_transform(data[choose_well]), columns=data[choose_well].columns)
        u0, s0, vt0 = SVD(data_norm, full_matrices=False)
        z_svd = u0[:,:2]@np.diag(s0)[:2,:2]@vt0[:2,:2]
        z_pca = PCA(2).fit_transform(data_norm)
        (cA, cD) = dwt(data_norm, wavelet=wavelet, mode=dwtmode)
        z_dwt = ((cA+cD)/2)[:,:2]
        return data_norm, z_svd, z_pca, z_dwt
    u0, s0, vt0 = SVD(data[choose_well], full_matrices=False)
    z_svd = u0[:,:2]@np.diag(s0)[:2,:2]@vt0[:2,:2]
    z_pca = PCA(2).fit_transform(data[choose_well])
    (cA,cD) = dwt(data[choose_well], wavelet=wavelet, mode=dwtmode)
    z_dwt   = ((cA+cD)/2)[:,:2]
    return z_svd, z_pca, z_dwt