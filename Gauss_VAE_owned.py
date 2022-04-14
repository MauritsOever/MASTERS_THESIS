# -*- coding: utf-8 -*-
"""
Own implementations of GAUSS VAE

Created on Thu Apr 14 11:29:10 2022

@author: gebruiker
"""
# imports
import numpy as np
import torch
from torch import nn



#%% block of code that can generate normal uncorrelated data
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
list_of_tuples = [(0,1), (0,0.01), (0,12), (0,10), (0,6), (0,85)]


#%% block of code that will bet return data
def Yahoo(list_of_ticks, startdate, enddate, retsorclose = 'rets'):
    '''
    Parameters
    ----------
    list_of_ticks : list of strings, tickers
    startdate     : string, format is yyyy-mm-dd
    enddate       : string, format is yyyy-mm-dd
    retsorclose   : string, 'rets' for returns and 'close' for adjusted closing prices
    
    
    Returns
    -------
    dataframe of stock returns or prices based on tickers and date

    '''
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    dfclose = pd.DataFrame(yf.download(list_of_ticks, start=startdate, end=enddate))['Adj Close']
    dfrets  = np.log(dfclose) - np.log(dfclose.shift(1))
    
    if retsorclose == 'rets':
        return dfrets
    else:
        return dfclose

list_of_ticks = ['AAPL', 'MSFT', ]


#%% create VAE module

# create class that inherits from nn.Module
# super
# define encoder, decoder
# define forward based on these two
# define loss
# define train function
# done...


class GaussVAE(nn.Module):
    """
    Inherits from nn.Module to construct Gaussian VAE based on given data and 
    desired dimensions. 
    
    INPUTS for instantiating:
    -------------------------
    X     : m by n tensor where m is the amount of observations and n is the amount of
            dimensions
    
    dim_Z : desired amount of dimensions in the latent space
    
    To do:
        - generalise encoder/decoder construction
        - generalise activation function
        - change KL loss for non-unit mult normal
        - KL feels janky, show to other people for confirmations
        
    """
    def __init__(self, X, dim_Z):
        super(GaussVAE, self).__init__()
        
        #self.float()
        if X.type() != 'torch.FloatTensor':
            self.X = torch.tensor(self.X).float()
            print('forcing X to float tensor...')
        else:
            self.X = X
        
        
        self.dim_X = X.shape[1]
        self.dim_Z = dim_Z
        self.dim_Y = int((self.dim_X + self.dim_Z) / 2)
        
        
        self.beta = 0 # setting beta to zero is equivalent to a normal autoencoder
            
        
        # sigmoid for now, but could also be ReLu, GeLu, tanh, etc
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_X, self.dim_Y), nn.Sigmoid(),
            nn.Linear(self.dim_Y, self.dim_Y), nn.Sigmoid(),
            nn.Linear(self.dim_Y, dim_Z))
        
        # same as encoder but in reversed order
        self.decoder = nn.Sequential(
            nn.Linear(dim_Z, self.dim_Y), nn.Sigmoid(),
            nn.Linear(self.dim_Y, self.dim_Y), nn.Sigmoid(),
            nn.Linear(self.dim_Y, self.dim_X))
    
    
    def objective_function(self, x):
        z       = self.encoder(x)
        x_prime = self.decoder(z)
        
        target = torch.randn(z.shape) # unit multivariate for now, might change for later
        
        KL_loss = nn.KLDivLoss(reduction = 'batchmean') # becomes a method
        
        KL = KL_loss(nn.functional.log_softmax(z, dim=1), target)
        RE = ((x - x_prime)**2).mean() # mean squared error of reconstruction
        
        return RE + self.beta * KL # function stolen from Bergeron et al. (2021) 
    
    def fit(self):
        """
        Function that fits the model based on previously passed data
        
        To do:
            - code it 
            - try different optimizers
            - tweak loss function if necessary
            - 
        """
        pass
    # okay now we have to code a function that fits i.e. optimizes this whole thing
        
#%% test code here
dim_Z = 3
X     = torch.tensor(GenerateNormalData(list_of_tuples, n)).float()

model = GaussVAE(X, dim_Z)
        
z = model.objective_function(X).detach().numpy() # detach is required as z required grad

