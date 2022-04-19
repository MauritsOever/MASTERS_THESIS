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
        - code a random/grid search (may already exist)
        
    """
    def __init__(self, X, dim_Z):
        """
        Constructs attributes, such as the autoencoder structure itself

        Inputs for instantiating:
        -------------------------
        X     : multidimensional np array or pd dataframe
        
        dim_Z : desired amount of dimensions in the latent space 

        """
        super(GaussVAE, self).__init__()
        
        # make X a tensor and standardize it
        self.X     = self.standardize_X(self.force_tensor(X)) # first force X to be a float tensor, and then standardize it

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
    
    def standardize_X(self, X):
        # write code that standardizes X
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def force_tensor(self, X):
        # write code that forces X to be a tensor
        if type(X) != torch.Tensor:
            return torch.Tensor(X).float()
        else:
            return X.float() # force it to float anyway
    
    def RE_KL_metric(self):
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
        
        return (RE, KL) # function stolen from Bergeron et al. (2021) 

    
    def loss_function(self, RE_KL):
        return RE_KL[0] + self.beta * RE_KL[1]
    
    def fit(self):
        """
        Function that fits the model based on previously passed data
        
        To do:
            - code it (yea kind of ready)
            - try different optimizers
            - tweak loss function if necessary bc it feels janky
        """
        self.train() # turn into training mode
        epochs  = 1000 # amount of iterations        
        REs  = []
        KLs  = []
        
        optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8) # specify some hyperparams for the optimizer
        
        for epoch in range(epochs):
            RE_KL = self.RE_KL_metric() # store RE and KL in tuple
            loss = self.loss_function(RE_KL) # calculate loss function based on tuple
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            REs += [RE_KL[0]]
            KLs += [RE_KL[1]] # RE and KLs are stored so we can look at them
        
        fig, axs = plt.subplots(2)
        axs[0]   = plt.plot(range(epochs), REs)
        axs[1]   = plt.plot(range(epochs), KLs)
        axs[0].set_title('Reconstruction errors')
        axs[1].set_title('KL distances')
        plt.tight_layout()
        plt.show()
        self.eval() # turn back into performance mode
        
        
#%% test code here:
# get data to test on
X     = GenerateNormalData(list_of_tuples, n)
#X = np.array(Yahoo(list_of_ticks, startdate, enddate).iloc[1:, :])

#%% actually run here
dim_Z = 3
model = GaussVAE(X, dim_Z)   
# z = model.objective_function().detach().numpy() # detach is required as z required grad

# model.fit()


# standardize data 
# 