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
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats

# GenerateAllDataSets(delete_existing=True)
layerz = 1
dim_Z  = 3
q      = 0.10
epochs = 2000
# clean and write
X, weights = GetData('returns')

model = VAE(X, dim_Z, layers=layerz, standardize = True, batch_wise=True, done=False, plot=False, dist='t')
model.fit(epochs)

for i in range(4):
    plt.plot(model.MMs[:,i])
    plt.show()


#%%
z = model.encoder(model.X).detach().numpy()

for i in range(dim_Z):
    plt.hist(z[:,i])
    plt.show()

print(np.cov(z, rowvar=False))

#%%
X_standard = (X - X.mean(axis=0))/X.std(axis=0)

x_prime = model.decoder(model.encoder(model.X)).detach().numpy()

for i in range(X_standard.shape[1]):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
    ax1.plot(X_standard[:,i])
    #ax1.set_ylim([-20,20])
    ax2.plot(x_prime[:,i])
    #ax2.set_ylim([-20,20])
    
#%%
test = np.random.standard_t(6.0, (1000,4))

test2 = np.cov(test, rowvar=False)



#%%
model.fit_garch_latent()

portVaRs = model.latent_GARCH_HS().mean(axis=1)
portRets = X.mean(axis=1)
violations = np.array(torch.Tensor(portVaRs > portRets).long())

plt.plot(portRets)
plt.plot(portVaRs)
plt.tight_layout()
plt.show()

print('')
print(f'model REs = {model.REs.mean()}')
print(f'ratio     = {np.sum(violations)/len(violations)}')


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


