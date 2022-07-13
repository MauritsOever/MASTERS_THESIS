# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""


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


#%%
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
# os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')

from models.Gauss_VAE import GaussVAE
from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from models.MGARCH import DCC_garch, robust_garch_torch
from data.datafuncs import GetData, GenerateAllDataSets
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats

X = GetData('IV')

dates = ['2008-08-29', '2008-12-16', '2008-12-17', '2012-01-06', '2012-01-17', 
          '2017-05-24', '2017-06-05', '2018-11-28']

# dates with a LOT of nans
dates2 = ['2009-11-03', '2009-11-18', '2011-08-12', '2011-09-07', '2011-10-04', '2012-02-14']

dates = dates + dates2

columns = ['date', 'atmVola', 'adjusted close', 'numOptions', 'futTTM', 'opTTM']
data = pd.read_csv(os.getcwd()+'\\data\\datasets\\real_sets\\FuturesAndatmVola.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['date', 'opTTM'])[columns]

model = GaussVAE(X, dim_Z=3, layers=2, plot=False, batch_wise=True, standardize=True)
model.fit(epochs=2000)
print('')
print(f'model REs are {model.REs[:,0].mean()}')
print('')

counter = 1
for date in dates:
    old_curve = data[data['date'] == date][columns[1:]]
    
    curve_to_correct = old_curve.copy()
    curve_to_correct['atmVola'] = curve_to_correct['atmVola'].interpolate(method='linear')
    
    new_curve = model.forward(np.array(curve_to_correct))
    
    fig, axs = plt.subplots(1,2, figsize=(15, 5))
    axs[0].plot(np.array(old_curve['atmVola']))
    axs[0].set_title(date + ' - original curve')
    axs[0].set_ylim(bottom=0, top=1)
    axs[1].plot(new_curve[:,0])
    axs[1].set_title(date + ' - corrected curve')
    axs[1].set_ylim(bottom=0, top=1)
    plt.tight_layout()
    plt.savefig(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\\Documenten\\THESIS\\present\corrected curve ' + str(counter) +'.png')
    counter += 1
    plt.show()
    

