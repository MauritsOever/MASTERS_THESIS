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

import matplotlib.pyplot as plt



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

list_of_ticks = ['AAPL', 'MSFT', 'KO', 'PEP', 'MS', 'GS', 'WFC', 'TSM']
startdate     = '2010-01-01'
enddate       = '2020-12-31'


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
    
    To do:
        - generalise encoder/decoder construction
        - generalise activation function
        - change KL loss for non-unit mult normal
        - KL feels janky, show to other people for confirmations
        
    """
    def __init__(self, X, dim_Z):
        """
        Constructs attributes, such as the autoencoder structure itself

        Inputs for instantiating:
        -------------------------
        X     : float tensor of data
        
        dim_Z : desired amount of dimensions in the latent space 

        """
        super(GaussVAE, self).__init__()
        
        # force it to be a 
        
        self.X     = X
        self.dim_X = X.shape[1]
        self.dim_Z = dim_Z
        self.dim_Y = int((self.dim_X + self.dim_Z) / 2)
        
        
        self.beta = 1 # setting beta to zero is equivalent to a normal autoencoder
            
        
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
    
    
    def objective_function(self):
        """
        Function that calculates the loss of the autoencoder by adding the
        RE and the (weighted) KL. 

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        z       = self.encoder(self.X)
        
        x_prime = self.decoder(z)
        
        target = torch.randn(z.shape) # unit multivariate for now, might change for later
        
        KL_loss = nn.KLDivLoss(reduction = 'batchmean') # as a method, and call it later
        
        KL = KL_loss(nn.functional.log_softmax(z, dim=1), target)
        RE = ((self.X - x_prime)**2).mean() # mean squared error of reconstruction
        
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
        self.train()
        epochs  = 5 # amount of iterations        
        losses  = []
        
        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
        
        for epoch in range(epochs):
            loss = self.objective_function()
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses += [loss]
        
        plt.plot(range(epochs), losses)
        plt.show()
        print(losses)
        self.eval()
        
        
#%% test code here:
# get data to test on
#X     = GenerateNormalData(list_of_tuples, n)
X = Yahoo(list_of_ticks, startdate, enddate).iloc[1:, :]
X = torch.tensor(np.array(X)).float()

#%% actually run here
dim_Z = 3
model = GaussVAE(X, dim_Z)   
# z = model.objective_function().detach().numpy() # detach is required as z required grad

model.fit()


# standardize data 
# 