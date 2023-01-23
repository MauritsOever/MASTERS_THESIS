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
import datetime

def RE_analysis(data_type, in_sample=True):
    
    begin_time = datetime.datetime.now()
        
    GenerateAllDataSets(delete_existing=False)
    
    epochs          = 2500
    
    simulated_dims = [1,2,3,4,6,12]
    assumed_dims   = [1,2,3,4,6,12]


    print(f'data of type {data_type}')
    
    if all([data_type == 'returns', in_sample==True]):
        epochs = 10000
        assumed_dims = [1,2,3,4,5]
        X, _ = GetData(data_type)
        REs = np.full((3, len(assumed_dims)), 'f'*40)
        for ass_dim in range(len(assumed_dims)):
            errgaus = 0
            errt    = 0
            subruns = 10
            print(f'assumed dim {ass_dim+1} out of {len(assumed_dims)}')
            for i in range(subruns):
                print(f'subrun {i+1} out of {subruns}')
                
                modelgauss = VAE(X, assumed_dims[ass_dim], layers=2, dist='normal')
                modelt     = VAE(X, assumed_dims[ass_dim], layers=2, dist='t')
            
                modelgauss.fit(epochs)
                modelt.fit(epochs)
                
                # if np.round(modelgauss.REs.mean().detach().numpy(), 3) < errgaus:
                errgaus += np.round(modelgauss.REs.mean().detach().numpy(), 3)
                
                # if np.round(modelt.REs.mean().detach().numpy(), 3) < errt:
                errt += np.round(modelt.REs.mean().detach().numpy(), 3)
                
            errgaus = errgaus / subruns
            errt    = errt / subruns
            
            X_standard = (X - X.mean(axis=0))/X.std(axis=0)
            
            decomp = PCA(n_components=assumed_dims[ass_dim])
            decomp = decomp.fit(X_standard)
            transform = decomp.transform(X_standard)
            X_prime   = decomp.inverse_transform(transform)
            errpca    = ((X_prime - X_standard)**2).mean()
            del decomp
            
            REs[0, ass_dim] = '\\cellcolor{blue!' + str(errpca*85) + '}' + str(errpca)
            REs[1, ass_dim] = '\\cellcolor{blue!' + str(errgaus*85) + '}' + str(errgaus)
            REs[2, ass_dim] = '\\cellcolor{blue!' + str(errt*85) + '}' + str(errt)
            
        REdf = pd.DataFrame(REs)
        REdf.index = ['PCA', 'Gaussian', 'Student t']
        
        write_dir = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\RE tables'
        files      = write_dir + '\RE_returns.txt'
        print(files)
        with open(files, 'w') as f:
                print(REdf.style.format().to_latex(), file=f)
    
    else:
        if in_sample:
            for modeltype in ['normal', 't']: #, 't']:
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
                        subrun_count = 10
                                            
                        err25 = 0
                        err50 = 0
                        err75 = 0
                        
                        for i in range(subrun_count):
                            print(f'subrun {i+1} out of {subrun_count}')
                            model25 = VAE(data25, assumed_dims[ass_dims], layers=2, done=False, dist=modeltype)
                            model50 = VAE(data50, assumed_dims[ass_dims], layers=2, done=False, dist=modeltype)
                            model75 = VAE(data75, assumed_dims[ass_dims], layers=2, done=False, dist=modeltype)
                            
                            model25.fit(epochs)
                            model50.fit(epochs)
                            model75.fit(epochs)
                            
                            #if model25.REs.mean().detach().numpy() < err25:
                            err25 += model25.REs.mean().detach().numpy()
                                
                            #if model50.REs.mean().detach().numpy() < err50:
                            err50 += model50.REs.mean().detach().numpy()
                            
                            #if model75.REs.mean().detach().numpy() < err75:
                            err75 += model75.REs.mean().detach().numpy()

                            
                        err25 = np.round(err25, 3) / subrun_count
                        err50 = np.round(err50, 3) / subrun_count
                        err75 = np.round(err75, 3) / subrun_count
                        
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
                    
        elif not in_sample:
            counter = 0
            print('we doing out of sample')
            
            simulated_dims = [4]
            
            REs_PCA  = np.full((len(simulated_dims),len(assumed_dims)), 'f'*40)
            REs_gaus = np.full((len(simulated_dims),len(assumed_dims)), 'f'*40)
            REs_t = np.full((len(simulated_dims),len(assumed_dims)), 'f'*40)
            

            
            for sim_dim in range(len(simulated_dims)):
                if data_type == 'returns':
                    data, _ = GetData(data_type, simulated_dims[sim_dim], 0.5)
                else:
                    data = GetData(data_type, simulated_dims[sim_dim], 0.5)
                # print(f'simdim = {simulated_dims[sim_dim]}')
                
                for ass_dim in range(len(assumed_dims)):
                    print(sim_dim, ass_dim)
                    err_pca = 0
                    err_gaus = 0
                    err_t    = 0
                    
                    for i in range(10):
                        # define model w 10th missing, get error on remaining tenth
                        counter += 1
                        print(f'count {counter} out of {len(assumed_dims)*len(simulated_dims)*10}')
                        
                        length = int(len(data) / 10)
                        index  = np.linspace(length*i, length*(i+1)-1, length).astype(int)
                        xtrain = np.delete(data, index, axis=0)
                        xtest  = data[index]
                        
                        xtrain_standard =(xtrain - xtrain.mean(axis=0))/xtrain.std(axis=0) 
                        xtest_standard = (xtest - xtest.mean(axis=0))/xtest.std(axis=0)

                        decomp = PCA(n_components=assumed_dims[ass_dim])
                        decomp = decomp.fit(xtrain_standard)
                        
                        transform = decomp.transform(xtest_standard)
                        X_prime   = decomp.inverse_transform(transform)
                        err_pca    = ((X_prime - xtest_standard)**2).mean()
                        
                        if ((X_prime - xtest_standard)**2).mean() > err_pca:
                            err_pca = ((X_prime - xtest_standard)**2).mean()
                                                                           
                        modelgaus = VAE(xtrain, ass_dim, layers=3,  dist='normal')
                        modelt    = VAE(xtrain, ass_dim, layers=3,  dist='t')
                        
                        modelgaus.fit(epochs=2500)
                        modelt.fit(epochs=2500)
                        
                        gaus_output = modelgaus.decoder(modelgaus.encoder(torch.Tensor(xtest_standard))).detach().numpy()
                        t_output    = modelt.decoder(modelt.encoder(torch.Tensor(xtest_standard))).detach().numpy()
                        
                        # if ((gaus_output - xtest_standard)**2).mean() < err_gaus:
                        #     err_gaus = ((gaus_output - xtest_standard)**2).mean()
                        
                        # if ((t_output - xtest_standard)**2).mean() < err_t:
                        #     err_t = ((t_output - xtest_standard)**2).mean()
                        
                        err_gaus += ((gaus_output - xtest_standard)**2).mean()
                        err_t += ((t_output - xtest_standard)**2).mean()
                        
                    err_gaus /= 10
                    err_t    /= 10
                    
                    REs_PCA[sim_dim, ass_dim]  = '\\cellcolor{blue!' + str(err_pca*85) + '}' + str(err_pca)
                    REs_gaus[sim_dim, ass_dim] = '\\cellcolor{blue!' + str(err_gaus*85) + '}' + str(err_gaus)
                    REs_t[sim_dim, ass_dim]    = '\\cellcolor{blue!' + str(err_t*85) + '}' + str(err_t)
                    
            REs_pca  = pd.DataFrame(REs_PCA)
            REs_gaus = pd.DataFrame(REs_gaus)
            REs_t    = pd.DataFrame(REs_t)
            REs_gaus.index = simulated_dims
            REs_t.index = simulated_dims
            
            write_dir = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\RE tables'
            files     = write_dir + '\RE_outofsample.txt'
            
            with open(files, 'w') as f:
                print('mixed data out of sample errors', file=f)
                print('', file=f)
                
                print('PCA', file=f)
                print(REs_pca.style.format().to_latex(), file=f)
                print('', file=f)
                
                print('Gaussian VAE', file=f)
                print(REs_gaus.style.format().to_latex(), file=f)
                print('', file=f)
                
                print('Student t VAE', file=f)
                print(REs_t.style.format().to_latex(), file=f)
                print('', file=f)
                
                
    time = datetime.datetime.now() - begin_time
    print('')
    print(f'time to run was {time}')
    subprocess.Popen('explorer '+write_dir) 
    import win32api
    win32api.MessageBox(0, 'RE analysis is done :)', 'Done!', 0x00001040)
               
    return


#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 
    RE_analysis('returns', in_sample=True)

if __name__=='__main__':
    main()