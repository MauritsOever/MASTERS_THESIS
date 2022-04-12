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
import math
import time

import torch
from torch import nn

#from doqu import document_base
#from doqu.utils import from_numpy, to_numpy, make_data_loader
#from doqu import gaussian_nll, standard_gaussian_nll, gaussian_kl_divergence, reparameterize

# probably have to use pythorch's NLL's and KL's

# generate a dataset(normal first) here
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




# create class that creates VAE based on hyperparameters
class Gaussian_VAE:
    
    def __init__(self, n_in, n_latent, n_out):
        """
        Initialises the class by creating Gaussian network that the other methods use

        Parameters
        ----------
        dim_X : dimensions of the original data, but can be coded to be found
        dim_Z : dimensions of the latent space

        """


# .fit(data) method, which also reconstructs

# .predict(out_of_sample) method

# .generate() method, which generates data according to the latent distribution

# 