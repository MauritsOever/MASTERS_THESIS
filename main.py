# -*- coding: utf-8 -*-
"""
todo:
    - code data functions in separate file and import here
    - make it such that u can just import model here as well
    - then code output and maybe some mf metrics or something...
    

@author: MauritsOever
"""

def main():
    # handle imports 
    import os
    os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')
    os.getcwd()
    from models.Gauss_VAE import GaussVAE
    #from models.GaussMix_VAE import GaussMixVAE
    #from models.StudentT_VAE import StudentTVAE
    from data.datafuncs import GetData
    
    
    dim_Z = 3
    
    print('Getting data')
    X = GetData('t')
    
    model_gauss = GaussVAE(X, dim_Z)
    model_gauss.fit()
    
    # get insample metrics when obtaining xprime
    # do some out of sample analysis too

    #model_gaussmix = GaussMixVAE(X, dim_Z)
    #model_gaussmix.fit()
    
    #model_studentt = StudentTVAE(X, dim_Z)
    #model_studentt.fit()

if __name__=='__main__':
    main()