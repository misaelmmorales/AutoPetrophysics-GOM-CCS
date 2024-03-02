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
from matplotlib.colors import ListedColormap

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