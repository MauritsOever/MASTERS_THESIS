# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
# import os
# os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')

from models.Gauss_VAE import GaussVAE
#from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from data.datafuncs import GetData, GenerateAllDataSets
import torch
import numpy as np
import mpmath
import matplotlib.pyplot as plt

rho = 0.5
correlated_dims = 4
datatype = 't'


GenerateAllDataSets()
X = GetData(datatype, correlated_dims, rho)




#np.loadtxt(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets\')
