# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
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

# GenerateAllDataSets(delete_existing=True)
layerz = 6
dim_Z = 15
q = 0.05
epochs = 5000
# clean and write
X, weights = GetData('returns', correlated_dims=2, rho=0.75)


RE = 0
# model = GaussVAE(X, dim_Z)
for i in range(10):
    print(f'i is {i}')
    model = GaussVAE(X, dim_Z, layers=layerz, batch_wise=True, done=False)
    model.fit(epochs=epochs)
    RE += model.REs.mean().detach().numpy()
print('\n')
print(f'dim_z  = {dim_Z}')
print(f'epochs = {epochs}')
print(f'layers = {layerz}')
print(f'RE     = {RE/10}')
z = model.encoder(model.X).detach().numpy()

#%% 
from sklearn.decomposition import PCA
from models.Gauss_VAE import GaussVAE
from data.datafuncs import GetData, GenerateAllDataSets
import numpy as np
import seaborn as sb

layerz = 7
dim_Z = 15
q = 0.05
epochs = 5000

X, weights = GetData('returns')

# define PCs
decomp = PCA(n_components=X.shape[1])
decomp.fit(X)
data_comp = decomp.transform(X)
data_comp = (data_comp - np.mean(data_comp, axis=0)) / np.std(data_comp, axis=0)

# get z
model = GaussVAE(X, dim_Z, layers=layerz, batch_wise=True, done=False)
model.fit(epochs=epochs)
z = model.encoder(model.X).detach().numpy()

# plot/corr
PCs_Z = np.append(z, data_comp[:, :z.shape[1]], axis=1)
sb.heatmap(np.corrcoef(PCs_Z, rowvar = False))
X_Z = np.append(z, X, axis=1)
sb.heatmap(np.corrcoef(X_Z, rowvar = False))

plt.figure(figsize = (15,3))
plt.plot(range(z.shape[0]), z[:,1], alpha=0.3)
plt.plot(range(data_comp.shape[0]), data_comp[:,0], alpha=0.3)
plt.legend(['z', 'pc'])
plt.show()



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
