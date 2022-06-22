# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')
# os.chdir(r'C:\Users\gebruiker\Documents\GitHub\MASTERS_THESIS')

from models.Gauss_VAE import GaussVAE
from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from models.MGARCH import DCC_garch, robust_garch_torch
from data.datafuncs import GetData, GenerateAllDataSets
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats

# GenerateAllDataSets(delete_existing=True)
dim_Z = 3
q = 0.05
# clean and write
X, weights = GetData('returns', correlated_dims=2, rho=0.75)

# model = GaussVAE(X, dim_Z)
model = GaussVAE(X, dim_Z, layers=3, batch_wise=True, done=True)

model.fit(epochs=2500)
z = model.encoder(model.X).detach().numpy()

#%% 
# get all things in indiv series:
    # merge on date, and return big one
X = pd.read_csv(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\THESIS\data\indiv series\UIWM.csv').iloc[1:,:]
X['Name'] = pd.to_datetime(X['Name'])
