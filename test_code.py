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
from data.datafuncs import GetData, GenerateAllDataSets
from models.GARCH import univariate_garch

import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats
import scipy
from sklearn.decomposition import PCA

dim_Z = 3
dist = 'normal'
q = 0.05

X, _ = GetData('returns')
means = X.mean(axis=0)
stds  = X.std(axis=0)
X_standard = (X - means)/stds

weights = np.full((X.shape[0], X.shape[1]), 1.0/X.shape[1])

comp = PCA(n_components = dim_Z)
comp = comp.fit(X_standard)
transform = comp.transform(X_standard)
sigmas = univariate_garch(transform, dist).calibrate()

VaRs = np.zeros((len(sigmas), X.shape[1]))

for i in range(len(sigmas)):
    
    if dist == 'normal':
        sims = np.random.normal(loc=0, scale=1, size=(1000, X.shape[1])) # * sigmas[i,:]
    else:
        sims = np.random.standard_t(df=25., size=(1000, X.shape[1])) #* sigmas[i,:]
    
    l = scipy.linalg.cholesky(np.diag(sigmas[i,:]))
    
    covmat = comp.components_.T @ l @ comp.components_
    sims = sims @ covmat
    
    # sims = comp.inverse_transform(sims)
    sims = sims * stds + means # bring back to rets
    VaRs[i, :] = np.quantile(sims, q, axis=0) 
    
portVaRs = np.sum(VaRs * weights, axis=1)
portRets = np.sum(X * weights, axis=1)
violations = np.array(torch.Tensor(portVaRs > portRets).long())

print(f'ratio = {sum(violations) / len(violations)}')

#%% 
import numpy as np

violations = np.random.binomial(1, 0.10, size = (100,1))

def christoffersens_independence_test(violations):
    a00s = 0
    a01s = 0
    a10s = 0
    a11s = 0
    for i in range(violations.shape[0]):
        # independence
            if violations[i-1] == 0:
                if violations[i] == 0:
                    a00s += 1
                else:
                    a01s += 1
            else:
                if violations[i] == 0:
                    a10s += 1
                else:
                    a11s += 1
                    
    if a11s > 0 and a00s > 0:            
        qstar0 = a00s / (a00s + a01s)
        qstar1 = a10s / (a10s + a11s)
        qstar = (a00s + a10s) / (a00s+a01s+a10s+a11s)
        Lambda = (qstar/qstar0)**(a00s) * ((1-qstar)/(1-qstar0))**(a01s) * (qstar/qstar1)**(a10s) * ((1-qstar)/(1-qstar1))**(a11s)
        
        print(-2*np.log(Lambda))
        
        pval_chris = stats.chi2.ppf(-2*np.log(Lambda), df=1)
        #print(f'pvalue christoffersens test = {pval_chris}')
    else:
        pval_chris = 0
        #print('There are no consecutive exceedences, so we can accept independence')
    
    plt.plot(violations)
    return pval_chris

print(christoffersens_independence_test(violations))

#%%
from sklearn.decomposition import PCA
from data.datafuncs import GetData
from models.GARCH import univariate_garch
import numpy as np
import matplotlib.pyplot as plt

dist = 'normal'
dim_Z = 5
q = 0.05
X, _ = GetData('returns')
means = X.mean(axis=0)
stds  = X.std(axis=0)
X_standard = (X - means)/stds

weights = np.full((X.shape[0], X.shape[1]), 1.0/X.shape[1])

comp = PCA(n_components = dim_Z)
comp = comp.fit(X_standard)
transform = comp.transform(X_standard)
sigmas = univariate_garch(transform, dist).calibrate()

VaRs = np.zeros((len(sigmas), X.shape[1]))
for i in range(len(sigmas)):
    
    sims = np.random.normal(loc=0, scale=1, size=(1000, dim_Z)) * np.sqrt(sigmas[i,:])
    sims = comp.inverse_transform(sims)
    sims = sims * stds + means
    VaRs[i, :] = np.quantile(sims, q, axis=0) 
    
portVaRs = np.sum(VaRs * weights, axis=1)
portRets = np.sum(X * weights, axis=1)

plt.plot(portRets)
plt.plot(portVaRs)
plt.show()

violations = (portVaRs > portRets).astype(int)


print(f'ratio = {np.sum(violations) / len(violations)}')

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


