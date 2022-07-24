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


def _find_IV_model(amount_of_runs, dist):
    X = GetData('IV')
    modeldict = {}
    print(f'finding best IV model for {amount_of_runs} runs')
    for i in range(amount_of_runs):
        print(f'subrun {i+1} out of {amount_of_runs}')
        model = VAE(X, dim_Z=3, layers=2, plot=False, batch_wise=True, standardize=True, dist=dist )
        model.fit(epochs=2500)
        modeldict[str(i)] = model
        del model
    
    best_model = modeldict['0']
    for key in list(modeldict.keys())[1:]:
        if modeldict[key].weighted_RE < best_model.weighted_RE:
            best_model = modeldict[key]
            
    print('')
    print(f'best model has a weighted RE of {best_model.weighted_RE}')
    
    RE_path = r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\curves\model RE '
    files   = RE_path + dist + '.txt'
    
    with open(files, 'w') as f:
        print(f'best model has a weighted RE of {best_model.weighted_RE}', file=f)

    
    return best_model      
    
    
def Implied_volatility_analysis(amount_of_runs, dist):
    model = _find_IV_model(amount_of_runs, dist)
    
    columns = ['date', 'atmVola', 'adjusted close', 'numOptions', 'futTTM', 'opTTM']
    data = pd.read_csv(os.getcwd()+'\\data\\datasets\\real_sets\\FuturesAndatmVola.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    data = data.sort_values(['date', 'opTTM'])[columns]
    
    all_dates    = np.array(data['date'].unique()).astype(str)
    
    dates = ['2008-08-29', '2008-12-16', '2008-12-17', '2009-11-03', '2009-11-18', '2011-08-12', '2011-09-07', '2011-10-04', '2012-01-06', '2012-01-17', 
             '2012-02-14', '2017-05-24', '2017-06-05', '2018-11-28']
    
    for date in range(len(all_dates)):
        all_dates[date] = all_dates[date][:10]
    
    counter = 1
    for date in dates:
        datelocation = np.where(all_dates == date)[0][0]
        
        date_before = str(all_dates[datelocation-1])
        date_after  = str(all_dates[datelocation+1])
                        
        curve1 = data[data['date'] == date_before].iloc[:,1:]
        curve2 = data[data['date'] == date].iloc[:,1:] 
        curve3 = data[data['date'] == date_after].iloc[:,1:]
        
        curve_to_correct = curve2.copy()
        curve_to_correct['atmVola'] = curve_to_correct['atmVola'].interpolate(method='linear')
        
        new_curve = model.forward(np.array(curve_to_correct))[:,0]
        
        fig, axs = plt.subplots(1,2, figsize=(15, 5))
        axs[0].plot(np.array(curve1['opTTM']), np.array(curve1['atmVola']), color='orange', alpha=0.3)        
        axs[0].plot(np.array(curve2['opTTM']), np.array(curve2['atmVola']), color='cornflowerblue')
        axs[0].plot(np.array(curve3['opTTM']), np.array(curve3['atmVola']), color='green', alpha=0.3)        
        axs[0].set_title(date + ' - original curve')
        axs[0].set_ylim(bottom=0, top=1)
        axs[0].legend([f'curve on {date_before}', f'curve on {date}', f'curve on {date_after}'])
        axs[0].set_xlabel('Option time to maturity')
        axs[0].set_ylabel('Implied volatility')
        
        axs[1].plot(np.array(curve2['opTTM']), new_curve, color='cornflowerblue')
        axs[1].set_title(date + ' - corrected curve')
        axs[1].set_ylim(bottom=0, top=1)
        axs[1].set_xlabel('Option time to maturity')

        plt.tight_layout()
        plt.savefig(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\present\curves\modeldist '+ dist + ' corrected curve ' + str(counter) +'.png')
        counter += 1
        plt.show()
        

#%% 
def main():
    import warnings
    warnings.filterwarnings("ignore") 

    Implied_volatility_analysis(30, 'normal')
    
    import win32api
    win32api.MessageBox(0, 'IV analysis is done :)', 'Done!', 0x00001040)

    
if __name__=='__main__':
    main()