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
#from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from data.datafuncs import GetData, GenerateAllDataSets
import numpy as np
import pandas as pd
import datetime


def RE_analysis():
    begin_time = datetime.datetime.now()
    
    os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
    delete_existing = False
    GenerateAllDataSets(delete_existing=False)
    
    datatype        = 'normal' # normal, t, mix, returns
    epochs          = 10
    
    simulated_dims = [1,2,3,4,6,12]
    assumed_dims = [1,2,3,4,6,12]

    for data_type in ['normal']: #, 't', 'mix']:
        print(f'data of type {datatype}')
        
        for modeltype in ['normal', 't']:
            counter = 0 # needs 36 times
            
            REs25 = np.zeros((len(simulated_dims),len(assumed_dims)+1))
            REs50 = np.zeros((len(simulated_dims),len(assumed_dims)+1))
            REs75 = np.zeros((len(simulated_dims),len(assumed_dims)+1))
            
            REs25[:,0] = np.array(simulated_dims)
            REs50[:,0] = np.array(simulated_dims)
            REs75[:,0] = np.array(simulated_dims)
            
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
                    
                    model25.fit(epochs)
                    model50.fit(epochs)
                    model75.fit(epochs)
                    
                    REs25[simdim, ass_dims+1] = model25.REs.mean().detach().numpy()
                    REs50[simdim, ass_dims+1] = model50.REs.mean().detach().numpy()
                    REs75[simdim, ass_dims+1] = model75.REs.mean().detach().numpy()
                    counter += 1
                    print(f'count is {counter}')
            
            # print below here
            
            print('')
            print(f'model of type {modeltype}')
            print('')

            print('corr = 0.25: ')
            print(pd.DataFrame(REs25).round(decimals=3).to_latex(index=False))
            print('')
            
            print('corr = 0.50: ')
            print(pd.DataFrame(REs50).round(decimals=3).to_latex(index=False))
            print('')
            
            print('corr = 0.75: ')
            print(pd.DataFrame(REs75).round(decimals=3).to_latex(index=False))
            print('')
            
    time = datetime.datetime.now() - begin_time
    print(f'time to run was {time}')
    import win32api
    win32api.MessageBox(0, 'RE analysis is done :)', 'Done!', 0x00001040)


def GARCH_analysis():
    raise NotImplementedError()


def main():
    RE_analysis()
            
            
if __name__=='__main__':
    main()