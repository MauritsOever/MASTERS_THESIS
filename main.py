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



def getdata_fitmodel_and_output(modeltype, datatype, correlated_dims, dim_Z, rho, epochs):
    print(f'Getting data, datatype = {datatype}')
    print('')
    X = GetData(datatype, correlated_dims, rho)
    print(f'Data loaded, initializing model of modeltype {modeltype}:')
    
    if modeltype == 'normal':
        model = GaussVAE(X, dim_Z)
    elif modeltype == 't':
        model = StudentTVAE(X, dim_Z)
    elif modeltype == 'GausMix':
        #model = StudentTVAE(X, dim_Z)
        pass
    
    print('Fitting model, and plotting REs and LLs')
    model.fit(epochs)
    
    # write something here that analyzes final residuals model.REs
    
    # write something here that generates data according to Z, then decode, 
    # then unstandardize then compare distribution wise to X  

def main():
    # handle imports 
    os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
    delete_existing = False
    GenerateAllDataSets(delete_existing=True)
    
    modeltype       = 'normal' # normal, t, gausmix
    datatype        = 't' # normal, t, mix, returns, (interest rates)
    correlated_dims = 3
    dim_Z           = 3
    rho             = 0.75
    epochs          = 1000

    
    getdata_fitmodel_and_output(modeltype, datatype, correlated_dims, dim_Z, rho, epochs)
    # then repeat this for 

if __name__=='__main__':
    main()