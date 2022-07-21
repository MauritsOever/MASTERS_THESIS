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

from models.VAE import VAE
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
import subprocess

def RE_analysis():
    # begin_time = datetime.datetime.now()
    
    GenerateAllDataSets(delete_existing=False)
    
    epochs          = 2000
    
    simulated_dims = [1,2,3,4,6,12]
    assumed_dims   = [1,2,3,4,6,12]

    for data_type in ['returns']: #'normal', 't', 'mix', 'returns']:
        print(f'data of type {data_type}')
        
        if data_type == 'returns':
            assumed_dims = [5,10,15,20,25]
            X, _ = GetData(data_type)
            REs = np.full((2, len(assumed_dims)), 'f'*40)
            for ass_dim in range(len(assumed_dims)):
                errgaus = 0
                errt    = 0
                subruns = 1
                print(f'assumed dim {ass_dim+1} out of {len(assumed_dims)}')
                for i in range(subruns):
                    print(f'subrun {i+1} out of {subruns}')
                    
                    modelgauss = VAE(X, assumed_dims[ass_dim], dist='normal')
                    modelt     = VAE(X, assumed_dims[ass_dim], dist='t')
                
                    modelgauss.fit(epochs)
                    modelt.fit(epochs)
                
                    errgaus += np.round(modelgauss.REs.mean().detach().numpy(), 3)
                    errt += np.round(modelt.REs.mean().detach().numpy(), 3)
                
                errgaus = errgaus / subruns
                errt    = errt / subruns
                
                REs[0, ass_dim] = '\\cellcolor{blue!' + str(errgaus*85) + '}' + str(errgaus)
                REs[1, ass_dim] = '\\cellcolor{blue!' + str(errt*85) + '}' + str(errt)
                
            REdf = pd.DataFrame(REs)
            REdf.index = ['Gaussian', 'Student t']
            
            write_dir = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\RE tables'
            files      = write_dir + '\RE_returns.txt'
            print(files)
            with open(files, 'w') as f:
                    print(REdf.style.format().to_latex(), file=f)
                    
        else:
            for modeltype in ['t']: #, 't']:
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
                        subrun_count = 1
                                            
                        err25 = 0
                        err50 = 0
                        err75 = 0
                        
                        for i in range(subrun_count):
                            print(f'subrun {i+1} out of {subrun_count}')
                            model25 = VAE(data25, assumed_dims[ass_dims], done=False, dist=modeltype)
                            model50 = VAE(data50, assumed_dims[ass_dims], done=False, dist=modeltype)
                            model75 = VAE(data75, assumed_dims[ass_dims], done=False, dist=modeltype)
                            
                            model25.fit(epochs)
                            model50.fit(epochs)
                            model75.fit(epochs)
                            
                            err25 += model25.REs.mean().detach().numpy()
                            err50 += model50.REs.mean().detach().numpy()
                            err75 += model75.REs.mean().detach().numpy()
                            
                        err25 = np.round(err25/subrun_count, 3)
                        err50 = np.round(err50/subrun_count, 3)
                        err75 = np.round(err75/subrun_count, 3)
                        
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
                
                write_dir = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\RE tables'
                files      = write_dir + '\RE_' + data_type + '_' + modeltype + '.txt'
                print(files)
                with open(files, 'w') as f:
                    print('', file=f)
                    print(f'model of type {modeltype}', file=f)
                    print('', file=f)
        
                    print('corr = 0.25: ', file=f)
                    print(REs25.style.format().to_latex(), file=f)
                    print('', file=f)
                    
                    print('corr = 0.50: ', file=f)
                    print(REs50.style.format().to_latex(), file=f)
                    print('', file=f)
                    
                    print('corr = 0.75: ', file=f)
                    print(REs75.style.format().to_latex(), file=f)
                    print('', file=f)
            
    # time = datetime.datetime.now() - begin_time
    # print(f'time to run was {time}')
    import win32api
    win32api.MessageBox(0, 'RE analysis is done :)', 'Done!', 0x00001040)
    subprocess.Popen('explorer '+write_dir)            


    return


#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 
    RE_analysis()

if __name__=='__main__':
    main()