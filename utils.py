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

def plot_curve(ax, df, curve, lb, ub, color='k', size=2, pad=1, mult=1,
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

def plot_ccs_sand_wells(df=None, figsize=(10,5), value='POROSITY', cmap='jet', showcols:bool=False):
    df = pd.read_csv('Data/CCS_Sand_wells1.csv') if df is None else df
    print('DF Columns:', df.columns.values) if showcols else None
    plt.figure(figsize=figsize)
    plt.scatter(df['LONG'], df['LAT'], s=5, c=df[value], cmap=cmap)
    plt.xlabel('X (Longitude)', weight='bold'); plt.ylabel('Y (Latitude)', weight='bold')
    plt.colorbar(pad=0.04, fraction=0.046, label='{}'.format(value))
    plt.gca().set_facecolor('lightgray')
    plt.grid(True, which='both'); plt.tight_layout(); plt.show()
    return None

def plot_survey(survey, figsize=(10,5), showcols:bool=False):
    print('DF Columns:', survey.columns.values) if showcols else None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d', elev=40, azim=-45, aspect='equal')
    ax.scatter(survey['X(FT)'], survey['Y(FT)'], survey['MD(FT)'], s=5)
    ax.set_xlabel('X (ft)', weight='bold'); ax.set_ylabel('Y (ft)', weight='bold'); ax.set_zlabel('MD (ft)', weight='bold')
    ax.set(xlim3d=(0,500), ylim3d=(-500,0), zlim3d=(0,7000))
    ax.invert_zaxis()
    plt.tight_layout(); plt.show()
    return None
