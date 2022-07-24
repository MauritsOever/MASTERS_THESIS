# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
# os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')

from models.VAE import VAE

from models.MGARCH import DCC_garch, robust_garch_torch
from data.datafuncs import GetData, GenerateAllDataSets
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.decomposition import PCA

dim_Z           = 12
correlated_dims = 4
epochs = 2500
amount_of_runs = 1

X = GetData('normal', correlated_dims=correlated_dims, rho=0.75)

RE1 = 0
RE2 = 0

for i in range(amount_of_runs):
    print(f'run {i+1} out of {amount_of_runs}')
    
    model1 = VAE(X, dim_Z=dim_Z, layers=2, dist='normal')
    model2 = VAE(X, dim_Z=dim_Z, layers=2, dist='t')
    
    model1.fit(epochs)
    model2.fit(epochs)
    
    RE1 += model1.REs.detach().numpy().mean()
    RE2 += model2.REs.detach().numpy().mean()
    print('')
    print(f'model gaussian = {RE1/(i+1)}')
    print(f'model t        = {RE2/(i+1)}')
    print('')


print('')
print('TOTAL:')
print(f'model gaussian = {RE1/amount_of_runs}')
print(f'model t        = {RE2/amount_of_runs}')


#%% 
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
from data.datafuncs import GetData, GenerateAllDataSets
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
# data = pd.read_csv(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets\real_sets\masterset_returns.csv').iloc[1:,:]
data = pd.read_csv(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets\real_sets\FuturesAndatmVola.csv')
# data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# data = data[['adjusted close', 'atmVola', 'numOptions', 'futTTM', 'opTTM']]

seriesvola = data['atmVola']
seriesnums = data['numOptions']

seriesvola = seriesvola[seriesvola.isna()]
seriesnums = seriesnums[seriesnums==0]

# data = data.ffill()
# data = data.backfill()
# data = (np.log(data) - np.log(data.shift(1))).iloc[1:,:]#.iloc[2611:,:]

sumstats = pd.DataFrame(index = ['Mean', 'Min', 'Max', '$ \sigma $', 'Skewness', 'Kurtosis'])
info     = pd.DataFrame(index = ['Obs', 'NaN count', 'Description'])

def sumarray(y):
    y = y.dropna()
    # y = (np.log(y) - np.log(y.shift())).iloc[1:]
    array = [np.mean(y), min(y), max(y), np.std(y), stats.skew(y, nan_policy = 'omit'), stats.kurtosis(y, nan_policy = 'omit')]
    return np.array(array)

def infoarray(y):
    array = [len(y), y.isna().sum(), '']
    return np.array(array)

for col in data.columns:
    # sumstats[col] = sumarray(data[col])
    info[col] = infoarray(data[col])
    
sumstats = sumstats.transpose()
info     = info.transpose()

# print(sumstats.style.format(precision=3, escape='latex').to_latex())
print(info.style.format(precision=3, escape='latex').to_latex())

# plt.plot(series)

plt.plot(data[data['atmVola'].isna()]['numOptions'])
plt.plot(data[data['atmVola'].isna() == False]['numOptions'])
