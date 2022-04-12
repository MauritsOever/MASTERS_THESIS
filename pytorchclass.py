# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:12:19 2022

@author: MauritsOever

TODO:
    - start with normal VAE for now
    - explore other distances
    - explore more punishing tail errors in distribution
    - get it working :)
"""

# imports
import numpy as np
import math
import time

import torch
from torch import nn

#from doqu import document_base
#from doqu.utils import from_numpy, to_numpy, make_data_loader
#from doqu import gaussian_nll, standard_gaussian_nll, gaussian_kl_divergence, reparameterize

# probably have to use pythorch's NLL's and KL's

#%% generate a dataset(normal) here
def GenerateNormalData(list_of_tuples, n):
    """
    Parameters
    ----------
    list_of_tuples : the tuples contain the mean and var of the variables, the length
                     the list determines the amount of variables
    n              : int, amount of observations

    Returns
    -------
    an m by n array of uncorrelated normal data (diagonal covar matrix)

    """
    import numpy as np
    array = np.empty((n,len(list_of_tuples)))
    
    for variable in range(len(list_of_tuples)):
        array[:,variable] = np.random.normal(list_of_tuples[variable][0], list_of_tuples[variable][1], n)
        
    return array

n = 100000
list_of_tuples = [(0,1),(1,0.01),(40,12),(0,1)]
X = GenerateNormalData(list_of_tuples, n)
X = torch.tensor(X)

#%%
# create class that creates VAE based on hyperparameters
# mainly taking over github code and trying to understand it

class GaussianNetwork:
    def __init__(self, dim_X, dim_Z):
        """
        Initialises the class by creating Gaussian network that the other classes use

        Parameters
        ----------
        dim_X : dimensions of the original data, but can be coded to be found
        dim_Z : dimensions of the latent space
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(GaussianNetwork, self).__init__()
        
        self.dim_X = dim_X
        self.dim_Z = dim_Z
        n_h        = int((dim_X - dim_Z) / 2) 
        
        
        # encoder
        self.le1 = nn.Sequential(
            nn.Linear(dim_X, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(), 
        ) # makes the actual architecture, dims stay constant after a short bottleneck,
          # includes both linear and tanh transformations, as nn.Tanh() is not able to reduce dimensions
          # itself
        
        self.le2_mu = nn.Linear(n_h, dim_Z) # last step of 
        self.le2_ln_var = nn.Linear(n_h, dim_Z)
        
        
        # Decoder
        self.ld1 = nn.Sequential(
            nn.Linear(dim_Z, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        ) # same as the encoder, but reversed
        
        self.ld2_mu = nn.Linear(n_h, dim_X) # same as 
        self.ld2_ln_var = nn.Linear(n_h, dim_X)
        
        
        def encode(self, x):
            # 
            h = self.le1(x)
            return self.le2_mu(h), self.le2_ln_var(h)

        def decode(self, z):
            h = self.ld1(z)
            return self.ld2_mu(h), self.ld2_ln_var(h)
        
        
test = GaussianNetwork(4, 1)

#%%
# 

class GaussianVAE:
    
    def __init__(self, n_in, n_latent, n_out):
        """
        Inherits from GaussianNetwork class, and also contains the fitting, 
        generation and prediction methods

        Parameters
        ----------
        dim_X : dimensions of the original data, but can be coded to be found
        dim_Z : dimensions of the latent space

        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = GaussianNetwork(n_in, n_latent, n_h).to(self.device)

# .fit(data) method, which also reconstructs

# .predict(out_of_sample) method

# .generate() method, which generates data according to the latent distribution

# 