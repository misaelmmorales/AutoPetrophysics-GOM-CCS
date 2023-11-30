import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lasio
from scipy import linalg, stats, optimize

from statsmodels.tsa.arima.model import ARIMA

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