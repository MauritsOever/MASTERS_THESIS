# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')


from models.Gauss_VAE import GaussVAE
#from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from data.datafuncs import GetData
import torch
import numpy as np
import mpmath

correlated_dims = 3
dim_Z           = 4
rho             = 0.8
epochs          = 100


X = GetData('returns', correlated_dims, rho) # normal, t, returns, interestrates

model = GaussVAE(X, dim_Z)

model.fit(epochs)

test = model.forward(X[1:2500, :])
