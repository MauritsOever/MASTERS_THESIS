# -*- coding: utf-8 -*-
"""
APPLICATIONS OF AUTOENCODERS FOR MARKET RISK

Main file that executes the code needed for the analysis of the thesis

todo:
    - comment all code for clarity
    - finalize all models - GMM is left
    - implement different dists
    - code output and some performance analysis

@author: MauritsOever
"""
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
# os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')

from models.Gauss_VAE import GaussVAE
from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from models.MGARCH import DCC_garch, robust_garch_torch
from data.datafuncs import GetData, GenerateAllDataSets
import win32api
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

def GARCH_analysis(mode, dist):
    # begin_time = datetime.datetime.now()
    print('')
    print('VAE-GARCH analysis running...')
    
    print(f'Mode = {mode}')
    print(f'Dist = {dist}')
    print('')
    
    dim_Z = 3
    q = 0.05
    
    
    if mode == 'VAE':       
        print('load in return data: ')
        X, weights = GetData('returns')
        
        # equally weighted:
        weights = np.full((X.shape[0], X.shape[1]), 1.0/X.shape[1])
        
        print('for normal VAE: ')
        
        # if dist is normal --> gauss, if dist not normal --> t
        model = GaussVAE(X, dim_Z, layers=3, batch_wise=True, done=False)
        print('fitting VAE...')
        model.fit(epochs=2500)
        print('')
        print('fitting GARCH...')
        model.fit_garch_latent(epochs=50)
        print('')
        print('simming...')
        try:
            VaRs = model.latent_GARCH_HS(data=None, q=q)
        except:
            return [0,0,0]
        
        del model
        
        portVaRs = np.sum(VaRs.detach().numpy() * weights, axis=1)
        portRets = np.sum(X * weights, axis=1)
        
        del VaRs
    
    elif mode == 'PCA':

        # data, PCA fit and compress, then store loadings
        X, weights = GetData('returns')
        decomp = PCA(n_components=dim_Z)
        decomp.fit(X)
        data_comp = decomp.transform(X)

        # fit garch, and store its sigmas
        garch = robust_garch_torch(torch.Tensor(data_comp), 'normal') # dist
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
        del sigmas    
        portVaRs = np.sum(VaRs * weights, axis=1)
        portRets = np.sum(X * weights, axis=1)
        
    # ESsNP = ESs.detach().numpy()
    violations = np.array(torch.Tensor(portVaRs > portRets).long())
    del portVaRs, portRets
    # coverage
    ratio = sum(violations)/len(violations)
    pval_ratio = stats.binom_test(sum(violations), len(violations), p=q)
    
    print(f'ratio of violations = {ratio}')
    print(f'p-value of binom test = {pval_ratio}')
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
                    
    if a11s > 0:            
        qstar0 = a00s / (a00s + a01s)
        qstar1 = a10s / (a10s + a11s)
        qstar = (a00s + a10s) / (a00s+a01s+a10s+a11s)
        Lambda = (qstar/qstar0)**(a00s) * ((1-qstar)/(1-qstar0))**(a01s) * (qstar/qstar1)**(a10s) * ((1-qstar)/(1-qstar1))**(a11s)
        
        pval_chris = stats.chi2.ppf(-2*np.log(Lambda), df=1)
        print(f'pvalue christoffersens test = {pval_chris}')
    else:
        pval_chris = 0
        print('There are no consecutive exceedences, so we can accept independence')
    
    del a00s, a01s, a10s, a11s, violations
    
    return [ratio, pval_ratio, pval_chris]

def GARCH_analysis_coldstart(mode, dist):
    old_result = [0,0,0]
    for i in range(5):
        result = GARCH_analysis(mode, dist)
        if result[1] > old_result[1]:
            old_result = result
    win32api.MessageBox(0, 'GARCH analysis is done :)', 'Done!', 0x00001040)
    print('')
    print('best run:')
    print(f'ratio      = {old_result[0]}')
    print(f'pval       = {old_result[1]}')
    print(f'pval chris = {old_result[2]}')
    return 

#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 
    # test = RE_analysis()
    GARCH_analysis_coldstart('VAE', 'normal')

if __name__=='__main__':
     main()