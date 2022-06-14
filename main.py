# -*- coding: utf-8 -*-
"""
THE EFFECT OF DISTRIBUTIONAL ASSUMPTIONS ON AUTOENCODER PERFORMANCE

Main file that executes the code needed for the analysis of the thesis


todo:
    - comment all code for clarity
    - finalize all models - GMM is left
    - code output and some performance analysis
    - sim, garch, decode, VaR/ES -- sim is done
    - backtest VaR/ES

@author: MauritsOever
"""
import os
from models.Gauss_VAE import GaussVAE
from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from data.datafuncs import GetData, GenerateAllDataSets
import numpy as np
import pandas as pd
# import datetime
from scipy import stats
import torch

def RE_analysis():
    # begin_time = datetime.datetime.now()
    
    os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')
    # os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')

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

def GARCH_analysis():
    # begin_time = datetime.datetime.now()
    
    os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')
    # os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
    
    print('load in return data: ')
    X = GetData('returns')
    
    print('for normal VAE: ')
    dim_Z = 3
    q = 0.05
    
    model = GaussVAE(X, dim_Z, layers=3, batch_wise=True, done=True)
    print('fitting VAE:')
    model.fit(epochs=2500)
    print('')
    model.fit_garch_latent(epochs=50)

    VaRs, ESs = model.latent_GARCH_HS(data=None, q=q)
    
    violations = (VaRs > torch.Tensor(X)).long()
    # do statistical testing
    # coverage
    for i in range(violations.shape[1]):
        print(f'col {i}: ratio, p-value = {sum(violations[:,i])/len(violations[:,i]), stats.binomtest(int(sum(violations[:,i])), len(violations[:,i]), p=q).pvalue}')
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
            
            print(f'pvalue christoffersens test = {stats.chi2.ppf(-2*np.log(Lambda), df=1)}')
        except ZeroDivisionError:
            print('There are no consecutive exceedences, so we can accept independence')
        print('')
        # print output
    # time = datetime.datetime.now() - begin_time
    # print(f'time to run was {time}')
    import win32api
    win32api.MessageBox(0, 'GARCH analysis is done :)', 'Done!', 0x00001040)

#%%

def main():
    # test = RE_analysis()
    GARCH_analysis()
    
            
if __name__=='__main__':
    main()