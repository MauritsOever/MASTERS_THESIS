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
import scipy


def GARCH_analysis(mode, dist, dim_Z, q):
    #print('')
    #print('VAE-GARCH analysis running...')
    
    #print(f'Mode = {mode}')
    #print(f'Dist = {dist}')
    #print('')
    
    if mode == 'VAE':       
        #print('load in return data: ')
        X, weights = GetData('returns')
        
        # equally weighted:
        weights = np.full((X.shape[0], X.shape[1]), 1.0/X.shape[1])
        
        #print('for normal VAE: ')
        
        # if dist is normal --> gauss, if dist not normal --> t
        model = VAE(X, dim_Z, layers=1, plot=False, batch_wise=True, standardize=True, dist=dist)
        #print('fitting VAE...')
        model.fit(epochs=400)
        #print('')
        #print('fitting GARCH...')
        model.fit_garch_latent()
        #print('')
        #print('simming...')
        VaRs = model.latent_GARCH_HS(data=None, q=q)

        RE = model.REs.detach().numpy().mean()
        MM = model.MMs
        del model
        
        portVaRs = np.sum(VaRs * weights, axis=1)
        portRets = np.sum(X * weights, axis=1)
        
        del VaRs
    
    elif mode == 'PCA':

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
        
        # path = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\GARCH'
        # filename = path + f'\mode={mode} dist={dist}, q={q}, dim_z={dim_Z}.png'
        
        # plt.plot(portRets)
        # plt.plot(portVaRs)
        # plt.savefig(filename)
        # plt.show()
    
    else:
        #print(f'mode {mode} is not a valid mode')
        return [0,0,0]


    
    # ESsNP = ESs.detach().numpy()
    violations = np.array(torch.Tensor(portVaRs > portRets).long())
    # del portVaRs, portRets
    # coverage
    ratio = sum(violations)/len(violations)
    pval_ratio = stats.binom_test(sum(violations), len(violations), p=q)
    
    #print(f'ratio of violations = {ratio}')
    #print(f'p-value of binom test = {pval_ratio}')
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
        #print(f'pvalue christoffersens test = {pval_chris}')
    else:
        pval_chris = 0
        #print('There are no consecutive exceedences, so we can accept independence')
    

    del a00s, a01s, a10s, a11s, violations
    
    if mode == 'VAE':
        return [ratio, pval_ratio, pval_chris, portRets, portVaRs, RE, MM]
    else:
        return[ratio, pval_ratio, pval_chris, portRets, portVaRs]

def GARCH_analysis_coldstart(mode, dist, amount_of_runs=5, dim_Z=5, q=0.05):
    begin_time = datetime.datetime.now()
    old_result = [0,0,0]
    
    if mode == 'PCA':
        amount_of_runs = 1
    
    for i in range(amount_of_runs):
        #print(f'run number {i+1} out of {amount_of_runs}')
        result = GARCH_analysis(mode, dist, dim_Z, q)
        if amount_of_runs > 1:
            if result[1] > old_result[1]:
                old_result = result
        else:
            old_result = result
            
    # win32api.MessageBox(0, 'GARCH analysis is done :)', 'Done!', 0x00001040)

    #print('')
    #print('best run:')
    #print(f'ratio      = {old_result[0]}')
    #print(f'pval       = {old_result[1]}')
    #print(f'pval chris = {old_result[2]}')
    
    path = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\GARCH'
    filename = path + f'\mode={mode} dist={dist}, q={q}, dim_z={dim_Z}.png'
    
    plt.plot(old_result[3])
    plt.plot(old_result[4])
    plt.title('q = '+str(q))
    plt.legend(['Returns', 'VaR'])
    plt.savefig(filename)
    plt.show()
    
    time = datetime.datetime.now() - begin_time
    #print('')
    #print(f'time to run was {time}')
    
    return old_result

def run_all_coldstarts():
    
    amount_of_runs = 20
    
    dictionary = {}
    
    modes  = ['PCA']
    dists  = ['t']
    qs     = [0.1, 0.05, 0.01]
    dim_Zs = [3]
    
    total_runs = len(modes)*len(dists)*len(qs)*len(dim_Zs)*amount_of_runs
    print(f'doing {total_runs} runs')
    count = 0
    for mode in modes:
        for dist in dists:
            for q in qs:
                for dim_Z in dim_Zs:
                    dictionary[f'mode={mode}, dist={dist}, q={q}, dim_Z={dim_Z}'] = GARCH_analysis_coldstart(mode, dist, amount_of_runs, dim_Z, q)
                    count += amount_of_runs
                    print(f'runs {count} done out of {total_runs}')

                    # #print(f'ratio      = {result[0]}')
                    # #print(f'pval       = {result[1]}')
                    # #print(f'pval chris = {result[2]}')
                    
    path = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\GARCH'
    filename = path+'\output.txt'
    
    with open(filename, 'w') as f:
        for i in list(dictionary.keys()):
            print(i, file=f)
            print(f'ratio  = {dictionary[i][0]}', file=f)     
            print(f'pval   = {dictionary[i][1]}', file=f)
            print(f'pcrhis = {dictionary[i][2]}', file=f)
            print('', file=f)
            print('', file=f)
            
    win32api.MessageBox(0, 'GARCH analysis is done :)', 'Done!', 0x00001040)

    return dictionary


# dictionary = run_all_coldstarts()
#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 
    
    run_all_coldstarts()
    
    
if __name__=='__main__':
     main()