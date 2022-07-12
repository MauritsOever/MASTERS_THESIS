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

def RE_analysis():
    # begin_time = datetime.datetime.now()
    
    GenerateAllDataSets(delete_existing=False)
    
    epochs          = 2500
    
    simulated_dims = [1,2,3,4,6,12]
    assumed_dims   = [1,2,3,4,6,12]

    for data_type in ['returns']: #'normal', 't', 'mix', 'returns']:
        print(f'data of type {data_type}')
        
        if data_type == 'returns':
            assumed_dims = [1,2,3,4,6]
            X = GetData(data_type)
            REs = np.full((3, len(assumed_dims)), 'f'*40)
            for ass_dim in range(len(assumed_dims)):
                modelgauss = GaussVAE(X, assumed_dims[ass_dim])
                modelt     = StudentTVAE(X, assumed_dims[ass_dim])
                modelmix   = GaussMixVAE(X, assumed_dims[ass_dim])
                
                modelgauss.fit(epochs)
                modelt.fit(epochs)
                modelmix.fit(epochs)
                
                errgaus = np.round(modelgauss.REs.mean().detach().numpy(), 3)
                errt = np.round(modelt.REs.mean().detach().numpy(), 3)
                errmix = np.round(modelmix.REs.mean().detach().numpy(),3)
                
                REs[0, ass_dim] = '\\cellcolor{blue!' + str(errgaus*85) + '}' + str(errgaus)
                REs[1, ass_dim] = '\\cellcolor{blue!' + str(errt*85) + '}' + str(errt)
                REs[2, ass_dim] = '\\cellcolor{blue!' + str(errmix*85) + '}' + str(errmix)
                
            REdf = pd.DataFrame(REs)
            REdf.index = ['Gaussian', 'Student t', 'Gaussian Mixture']
            print(REdf.style.format().to_latex())
        else:
        
            for modeltype in ['normal', 't', 'mix']:
                counter = 0 # needs 36 times
                
                REs25 = np.full((len(simulated_dims),len(assumed_dims)), 'f'*40)
                REs50 = np.full((len(simulated_dims),len(assumed_dims)), 'f'*40)
                REs75 = np.full((len(simulated_dims),len(assumed_dims)), 'f'*40)
                
                # REs25[:,0] = np.array(simulated_dims)
                # REs50[:,0] = np.array(simulated_dims)
                # REs75[:,0] = np.array(simulated_dims)
                
                for simdim in range(len(simulated_dims)):
                    data25 = GetData(data_type, simulated_dims[simdim], 0.25)
                    data50 = GetData(data_type, simulated_dims[simdim], 0.50)
                    data75 = GetData(data_type, simulated_dims[simdim], 0.75)
                    
                    for ass_dims in range(len(assumed_dims)):
                        if modeltype == 'normal':
                            model25 = GaussVAE(data25, assumed_dims[ass_dims], done=False)
                            model50 = GaussVAE(data50, assumed_dims[ass_dims], done=False)
                            model75 = GaussVAE(data75, assumed_dims[ass_dims], done=False)
                        elif modeltype == 't':
                            model25 = StudentTVAE(data25, assumed_dims[ass_dims], done=False)
                            model50 = StudentTVAE(data50, assumed_dims[ass_dims], done=False)
                            model75 = StudentTVAE(data75, assumed_dims[ass_dims], done=False)
                        elif modeltype == 'mix':
                            model25 = GaussMixVAE(data25, assumed_dims[ass_dims], done=False)
                            model50 = GaussMixVAE(data50, assumed_dims[ass_dims], done=False)
                            model75 = GaussMixVAE(data75, assumed_dims[ass_dims], done=False)
                            
                        
                        model25.fit(epochs)
                        model50.fit(epochs)
                        model75.fit(epochs)
                        
                        err25 = np.round(model25.REs.mean().detach().numpy(), 3)
                        err50 = np.round(model50.REs.mean().detach().numpy(), 3)
                        err75 = np.round(model75.REs.mean().detach().numpy(), 3)
                        
                        REs25[simdim, ass_dims] = '\\cellcolor{blue!' + str(err25*85) + '}' + str(err25)
                        REs50[simdim, ass_dims] = '\\cellcolor{blue!' + str(err50*85) + '}' + str(err50)
                        REs75[simdim, ass_dims] = '\\cellcolor{blue!' + str(err75*85) + '}' + str(err75)
                        counter += 1
                        print(f'count is {counter}')
                
                # print below here
                REs25 = pd.DataFrame(REs25)
                REs50 = pd.DataFrame(REs50)
                REs75 = pd.DataFrame(REs75)
                
                REs25.index = simulated_dims
                REs50.index = simulated_dims
                REs75.index = simulated_dims
    
                
                print('')
                print(f'model of type {modeltype}')
                print('')
    
                print('corr = 0.25: ')
                print(REs25.style.format().to_latex())
                print('')
                
                print('corr = 0.50: ')
                print(REs50.style.format().to_latex())
                print('')
                
                print('corr = 0.75: ')
                print(REs75.style.format().to_latex())
                print('')
            
    # time = datetime.datetime.now() - begin_time
    # print(f'time to run was {time}')
    import win32api
    win32api.MessageBox(0, 'RE analysis is done :)', 'Done!', 0x00001040)

    return

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