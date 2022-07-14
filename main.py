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


def _find_IV_model(dist):
    X = GetData('IV')
    modeldict = {}
    for i in range(30):
        model = GaussVAE(X, dim_Z=3, layers=2, plot=False, batch_wise=True, standardize=True)
        model.fit(epochs=2000)
        modeldict[str(i)] = model
        del model
    
    best_model = modeldict['0']
    for key in list(modeldict.keys())[1:]:
        if modeldict[key].weighted_RE < best_model.weighted_RE:
            best_model = modeldict[key]
            
    print('')
    print(f'best model has a weighted RE of {best_model.weighted_RE}')
    return best_model      
    
    
def Implied_volatility_analysis(dist):
    model = _find_IV_model(dist)
    
    columns = ['date', 'atmVola', 'adjusted close', 'numOptions', 'futTTM', 'opTTM']
    data = pd.read_csv(os.getcwd()+'\\data\\datasets\\real_sets\\FuturesAndatmVola.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['date', 'opTTM'])[columns]
    
    dates = ['2008-08-29', '2008-12-16', '2008-12-17', '2009-11-03', '2009-11-18', '2011-08-12', '2011-09-07', '2011-10-04', '2012-01-06', '2012-01-17', 
             '2012-02-14', '2017-05-24', '2017-06-05', '2018-11-28']
    
    counter = 1
    for date in dates:
        old_curve = data[data['date'] == date][columns[1:]]
        
        curve_to_correct = old_curve.copy()
        curve_to_correct['atmVola'] = curve_to_correct['atmVola'].interpolate(method='linear')
        
        new_curve = model.forward(np.array(curve_to_correct))
        
        fig, axs = plt.subplots(1,2, figsize=(15, 5))
        axs[0].plot(np.array(old_curve['atmVola']))
        axs[0].set_title(date + ' - original curve')
        axs[0].set_ylim(bottom=0, top=1)
        axs[1].plot(new_curve[:,0])
        axs[1].set_title(date + ' - corrected curve')
        axs[1].set_ylim(bottom=0, top=1)
        plt.tight_layout()
        plt.savefig(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\\Documenten\\THESIS\\present\corrected curve ' + str(counter) +'.png')
        counter += 1
        plt.show()

#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 

    Implied_volatility_analysis('normal')
    
if __name__=='__main__':
    main()