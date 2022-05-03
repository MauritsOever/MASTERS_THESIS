# -*- coding: utf-8 -*-
"""
THE EFFECT OF DISTRIBUTIONAL ASSUMPTIONS ON AUTOENCODER PERFORMANCE

Main file that executes the code needed for the analysis of the thesis


todo:
    - finalize all models
    - code output and some performance analysis
    

@author: MauritsOever
"""

def getdata_fitmodel_and_output(modeltype, datatype, correlated_dims, dim_Z, rho, epochs):
    from models.Gauss_VAE import GaussVAE
    #from models.GaussMix_VAE import GaussMixVAE
    from models.StudentT_VAE import StudentTVAE
    from data.datafuncs import GetData, GenerateAllDataSets

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
    from data.datafuncs import GenerateAllDataSets
    import os
    os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
    delete_existing = False
    GenerateAllDataSets(delete_existing)
    
    modeltype       = 'normal' # normal, t, gausmix
    datatype        = 't' # normal, t, mix, returns, (interest rates)
    correlated_dims = 3
    dim_Z           = 3
    rho             = 0.8
    epochs          = 2

    
    getdata_fitmodel_and_output(modeltype, datatype, correlated_dims, dim_Z, rho, epochs)
    # then repeat this for 

if __name__=='__main__':
    main()