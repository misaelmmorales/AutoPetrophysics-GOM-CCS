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