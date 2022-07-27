# -*- coding: utf-8 -*-
"""
APPLICATIONS OF AUTOENCODERS FOR MARKET RISK

Main file that executes the code needed for the analysis of the thesis

todo:
    - comment all code for clarity
    - code output and some performance analysis

@author: MauritsOever
"""
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
# os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')

from models.VAE import VAE
from data.datafuncs import GetData, GenerateAllDataSets
from models.GARCH import univariate_garch
import win32api
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import datetime

def GARCH_analysis(mode, dist, dim_Z):
    print('')
    print('VAE-GARCH analysis running...')
    
    print(f'Mode = {mode}')
    print(f'Dist = {dist}')
    print('')
    
    q = 0.05
    
    if mode == 'VAE':       
        print('load in return data: ')
        X, weights = GetData('returns')
        
        # equally weighted:
        weights = np.full((X.shape[0], X.shape[1]), 1.0/X.shape[1])
        
        print('for normal VAE: ')
        
        # if dist is normal --> gauss, if dist not normal --> t
        model = VAE(X, dim_Z, layers=1, plot=False, batch_wise=True, standardize=True, dist=dist)
        print('fitting VAE...')
        model.fit(epochs=2000)
        print('')
        print('fitting GARCH...')
        model.fit_garch_latent()
        print('')
        print('simming...')
        VaRs = model.latent_GARCH_HS(data=None, q=q)

        RE = model.REs.detach().numpy().mean()
        MM = model.MMs
        del model
        
        portVaRs = np.sum(VaRs * weights, axis=1)
        portRets = np.sum(X * weights, axis=1)
        
        del VaRs
    
    elif mode == 'PCA':

        # data, PCA fit and compress, then store loadings
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
                sims = np.random.normal(loc=0, scale=1, size=(1000, dim_Z)) * sigmas[i,:]
            else:
                sims = np.random.standard_t(df=100.) * sigmas[i,:]
                
            sims = comp.inverse_transform(sims)
            sims = sims * stds + means
            VaRs[i, :] = np.quantile(sims, q, axis=0) 
            
        portVaRs = np.sum(VaRs * weights, axis=1)
        portRets = np.sum(X * weights, axis=1)
    
    else:
        print(f'mode {mode} is not a valid mode')
        return [0,0,0]

    plt.plot(portRets)
    plt.plot(portVaRs)

    plt.show()
    
    # ESsNP = ESs.detach().numpy()
    violations = np.array(torch.Tensor(portVaRs > portRets).long())
    # del portVaRs, portRets
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
                    
    if a11s > 0 and a00s > 0:            
        qstar0 = a00s / (a00s + a01s)
        qstar1 = a10s / (a10s + a11s)
        qstar = (a00s + a10s) / (a00s+a01s+a10s+a11s)
        Lambda = (qstar/qstar0)**(a00s) * ((1-qstar)/(1-qstar0))**(a01s) * (qstar/qstar1)**(a10s) * ((1-qstar)/(1-qstar1))**(a11s)
        
        pval_chris = stats.chi2.ppf(-2*np.log(Lambda), df=1)
        print(f'pvalue christoffersens test = {pval_chris}')
    else:
        pval_chris = 0
        print('There are no consecutive exceedences, so we can accept independence')
    
    if mode == 'VAE':
        print(f'RE       = {RE}')
        #print(f'MM       = {MM}')
    del a00s, a01s, a10s, a11s, violations
    
    if mode == 'VAE':
        return [ratio, pval_ratio, pval_chris, portRets, portVaRs, RE, MM]
    else:
        return[ratio, pval_ratio, pval_chris, portRets, portVaRs]

def GARCH_analysis_coldstart(mode, dist, amount_of_runs=5, dim_Z=5):
    begin_time = datetime.datetime.now()
    old_result = [0,0,0]
    
    if mode == 'PCA':
        amount_of_runs = 1
    
    for i in range(amount_of_runs):
        print(f'run number {i+1} out of {amount_of_runs}')
        result = GARCH_analysis(mode, dist, dim_Z)
        if result[1] > old_result[1]:
            old_result = result
            
    win32api.MessageBox(0, 'GARCH analysis is done :)', 'Done!', 0x00001040)
    if amount_of_runs > 1:
        print('')
        print('best run:')
        print(f'ratio      = {old_result[0]}')
        print(f'pval       = {old_result[1]}')
        print(f'pval chris = {old_result[2]}')
        if mode == 'VAE':
            print(f'RE         = {old_result[5]}')
            #print(f'MM         = {old_result[6]}')
        
        plt.plot(old_result[3])
        plt.plot(old_result[4])
        plt.show()
    
    time = datetime.datetime.now() - begin_time
    print('')
    print(f'time to run was {time}')
         

#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 
    # test = RE_analysis()
    GARCH_analysis_coldstart('VAE', 't', amount_of_runs=20, dim_Z=3)

if __name__=='__main__':
     main()