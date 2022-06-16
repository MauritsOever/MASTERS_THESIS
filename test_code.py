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

dim_Z = 3
q = 0.05
# clean and write
X = GetData('returns', correlated_dims=2, rho=0.75)

# model = GaussVAE(X, dim_Z)
model = GaussVAE(X, dim_Z, layers=3, batch_wise=True, done=True)

model.fit(epochs=2500)
z = model.encoder(model.X).detach().numpy()

model.fit_garch_latent(epochs=50)
VaRs, _ = model.latent_GARCH_HS()
VaRs = VaRs.detach().numpy()

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

from sklearn.decomposition import PCA

dim_Z = 3
q = 0.05


# data, PCA fit and compress, then store loadings
X = GetData('returns')
decomp = PCA(n_components=dim_Z)
decomp.fit(X)
data_comp = decomp.transform(X)
components = decomp.components_
loading_matrix = decomp.components_.T

# fit garch, and store its sigmas
garch = robust_garch_torch(torch.Tensor(data_comp), 'normal')
garch.fit(50)
garch.store_sigmas()
sigmas = []
for sigma in garch.sigmas:
    sigmas += [sigma.detach().numpy()] # 
    # sigmas += [loading_matrix @ sigma.detach().numpy() @ loading_matrix.T] # project into original space


VaRs = np.zeros((len(sigmas), X.shape[1]))
for i in range(len(sigmas)):
# for i in range(1):
    l = np.linalg.cholesky(sigmas[i])
    sims = np.random.normal(loc=0, scale=1, size=(1000, 3)) @ l
    sims = decomp.inverse_transform(sims)
    
    VaRs[i, :] = np.quantile(sims, 0.05, axis=0)
    



#%%

# ESsNP = ESs.detach().numpy()

violations_bool = VaRs > X
violations = np.zeros((violations_bool.shape[0], violations_bool.shape[1]))
for j in range(violations.shape[1]):
    for i in range(violations.shape[0]):
        violations[i,j] = 1 if violations_bool[i,j]==True else 0
# do statistical testing
# coverage
for i in range(violations.shape[1]):
    print(f'Ratio of violations for col {i} = {sum(violations[:,i]) / len(violations[:,i])}')
    #print(f'P-value binom test = {stats.binomtest(int(sum(violations[:,i])), len(violations[:,i]), p=q).pvalue}')
    # independence
    a00s = 0
    a01s = 0
    a10s = 0
    a11s = 0
    for j in range(1, violations.shape[0]):
        if violations[j-1, i] == 0:
            if violations[j,i] == 0:
                a00s += 1
            else:
                a01s += 1
        else:
            if violations[j,i] == 0:
                a10s += 1
            else:
                a11s += 1
    try:            
        qstar0 = a00s / (a00s + a01s)
        qstar1 = a10s / (a10s + a11s)
        qstar = (a00s + a10s) / (a00s+a01s+a10s+a11s)
        Lambda = (qstar/qstar0)**(a00s) * ((1-qstar)/(1-qstar0))**(a01s) * (qstar/qstar1)**(a10s) * ((1-qstar)/(1-qstar1))**(a11s)
        
        #print(f'pvalue christoffersens test = {stats.chi2.ppf(-2*np.log(Lambda), df=1)}')
    except ZeroDivisionError:
        print('There are no consecutive exceedences, so we can accept independence')


#%%

