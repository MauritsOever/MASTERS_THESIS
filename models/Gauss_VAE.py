# -*- coding: utf-8 -*-
"""
Own implementations of GAUSS VAE

Created on Thu Apr 14 11:29:10 2022

@author: gebruiker
"""
# imports
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class GaussVAE(nn.Module):
    """
    Inherits from nn.Module to construct Gaussian VAE based on given data and 
    desired dimensions. 
    
    To do:
        - generalise activation function (maybe)
        
        - change KL loss for different distributions
        - KL feels janky, show to other people for confirmations
        
        - code a random/grid search (may already exist, probs best to do it myself)
        
        - standardisation causes issues for Xprime analysis (discuss w Rens)
        - write code that can unstandardize X_prime (maybe solution for point above)
    """
    
    def __init__(self, X, dim_Z, layers=3, standardize = True):
        """
        Constructs attributes, such as the autoencoder structure itself

        Inputs for instantiating:
        -------------------------
        X           : multidimensional np array or pd dataframe
        
        dim_Z       : desired amount of dimensions in the latent space 
        
        layers      : int, amount of layers for the encoder and decoder, default = 3, must be >= 2
        
        standardize : bool, if true than X gets mean var standardized

        """
        # imports
        super(GaussVAE, self).__init__()
        from collections import OrderedDict
        import numpy as np
        import torch
        from torch import nn
        import matplotlib.pyplot as plt
        
        # make X a tensor, and standardize based on standardize
        if standardize:
            self.X     = self.standardize_X(self.force_tensor(X)) # first force X to be a float tensor, and then standardize it
        else:
            self.X     = self.force_tensor(X)


        self.dim_X = X.shape[1]
        self.dim_Z = dim_Z
        self.dim_Y = int((self.dim_X + self.dim_Z) / 2)
        
        
        self.beta = 1 # setting beta to zero is equivalent to a normal autoencoder
            
        
        # sigmoid for now, but could also be ReLu, GeLu, tanh, etc
        self.encoder = self.construct_encoder(layers)
        self.decoder = self.construct_decoder(layers)
        
        
    def construct_encoder(self, layers):
        network = OrderedDict()
        network['0'] = nn.Linear(self.dim_X, self.dim_Y)
        network['1'] = nn.Sigmoid()
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.Sigmoid()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_Z)
        
        return nn.Sequential(network)
    
        
    def construct_decoder(self, layers):
        network = OrderedDict()
        network['0'] = nn.Linear(self.dim_Z, self.dim_Y)
        network['1'] = nn.Sigmoid()
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.Sigmoid()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_X)
        
        return nn.Sequential(network)
        
    
    def standardize_X(self, X):
        # write code that stores mean and var, so u can unstandardize X_prime
        
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def unstandardize_Xprime(self, X_prime):
        pass
    
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
        self.REs = (self.X - x_prime)**2
        RE = self.REs.mean() # mean squared error of reconstruction
        
        return (RE, KL) # function stolen from Bergeron et al. (2021) 

    
    def loss_function(self, RE_KL):
        return RE_KL[0] + self.beta * RE_KL[1]
    
    def fit(self, epochs):
        """
        Function that fits the model based on previously passed data
        
        To do:
            - code it (yea kind of ready)
            - try different optimizers
            - tweak loss function if necessary bc it feels janky
        """
        from tqdm import tqdm
        
        self.train() # turn into training mode
        REs  = []
        KLs  = []
        
        optimizer = torch.optim.Adam(self.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8) # specify some hyperparams for the optimizer
        
        for epoch in tqdm(range(epochs)):
            RE_KL = self.RE_KL_metric() # store RE and KL in tuple
            loss = self.loss_function(RE_KL) # calculate loss function based on tuple
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            REs += [RE_KL[0]]
            KLs += [RE_KL[1]] # RE and KLs are stored for analysis
            
        plt.plot(range(epochs), REs)
        plt.title('Reconstruction errors')
        plt.show()
        plt.plot(range(epochs), KLs)
        plt.title('KL distances')
        plt.show()
        self.eval() # turn back into performance mode
        
        return
