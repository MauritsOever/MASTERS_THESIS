# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
import os
# os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')

from models.Gauss_VAE import GaussVAE
#from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from models.MGARCH import DCC_garch, robust_garch_torch
from data.datafuncs import GetData, GenerateAllDataSets
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# GenerateAllDataSets(delete_existing=True)

dim_Z = 3
# clean and write
# X = pd.read_csv(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets\real_sets\MO_THESIS.03.csv').drop(0, axis=0)
# X = X.ffill()
# X = X.backfill()
# X = np.array(X.iloc[:,1:])
# X = X.astype(float)
# X = np.log(X[1:,:]) - np.log(X[:-1,:])
X = GetData('normal', 4, 0.75) # normal, t, mix

# model = GaussVAE(X, dim_Z)
model = GaussVAE(X, dim_Z, layers=3, batch_wise=True, done=True)

model.fit(epochs=10000)

model.fit_garch_latent(epochs=100)

VaRs = model.latent_GARCH_HS()
VaRsNP = VaRs.detach().numpy()

for col in range(VaRs.shape[1]):
    plt.plot(VaRs[:,col].detach().numpy(), alpha=0.3)
plt.show()

# z = model.encoder(model.X).detach().numpy()
# print(f'means are {z.mean(axis=0)}')
# print(f'stds are {z.std(axis=0)}')
# print(f'skews are {stats.skew(z)}')
# print(f'kurts are {stats.kurtosis(z)}')
# print('')


#%%
model.fit_garch_latent()
#%% 
for mat in model.garch.sigmas:
    torch.linalg.cholesky(mat)
#%%
z = model.encoder(model.X).detach().numpy()

print(f'means are {z.mean(axis=0)}')
print(f'stds are {z.std(axis=0)}')
print(f'skews are {stats.skew(z)}')
print(f'kurts are {stats.kurtosis(z)}')
print('')

_ = plt.hist(z)

# print(f'jb test of col 1 {stats.jarque_bera(z[:,0])}')
# print(f'jb test of col 2 {stats.jarque_bera(z[:,1])}')
# print(f'jb test of col 3 {stats.jarque_bera(z[:,2])}')
# print(f'jb test of col 4 {stats.jarque_bera(z[:,3])}')

standard_t = np.random.standard_t(5, size=(10000, 4))

fig = sm.qqplot(z[:,0], stats.t, fit=True, line="45")

#%% okay now check if Z_extreme == X_extreme
z_low  = z[z[:,0] < np.quantile(z[:,0], 0.10), :]
z_high = z[z[:,0] > np.quantile(z[:,0], 0.90), :]
z_mid  = np.empty((0,4))
z_extreme = np.empty((0,4))

selector_mid = np.full(z.shape[0], False)
selector_extreme = np.full(z.shape[0], False)
for row in range(z.shape[0]):
    selector_mid[row] = all([z[row,0] > np.quantile(z[:,0], 0.45), z[row,0] < np.quantile(z[:,0], 0.55)])
    selector_extreme[row] = any([z[row,0] < np.quantile(z[:,0], 0.05), z[row,0] > np.quantile(z[:,0], 0.95)])

z_mid = z[selector_mid, :]
z_extreme = z[selector_extreme, :]

# okay now do the check
X_low = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_low))).detach().numpy()
X_mid = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_mid))).detach().numpy()
X_high = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_high))).detach().numpy()
X_extreme = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_extreme))).detach().numpy()
X_full = model.unstandardize_Xprime(model.decoder(torch.Tensor(z))).detach().numpy()

_ = plt.hist(X_low)
_ = plt.hist(X_mid)
_ = plt.hist(X_high)
_ = plt.hist(X_extreme)
_ = plt.hist(X_full)