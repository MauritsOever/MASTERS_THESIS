# -*- coding: utf-8 -*-
"""
Own implementation of Student-t VAE

Created on Thu Apr 14 11:29:10 2022

@author: gebruiker
"""
# imports
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import mpmath


class StudentTVAE(nn.Module):
    """
    Inherits from nn.Module to construct Student t VAE based on given data and 
    desired dimensions. 
    
    To do:
        - optimize hyperparams
    """
    
    def __init__(self, X, dim_Z, layers=3, standardize = True, batch_wise=True):
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
        super(StudentTVAE, self).__init__()
        from collections import OrderedDict
        import numpy as np
        import torch
        from torch import nn
        import matplotlib.pyplot as plt
        import mpmath
        
        # make X a tensor, and standardize based on standardize
        if standardize:
            self.X     = self.standardize_X(self.force_tensor(X)) # first force X to be a float tensor, and then standardize it
        else:
            self.X     = self.force_tensor(X)


        self.dim_X = X.shape[1]
        self.dim_Z = dim_Z
        self.n     = X.shape[0]
        self.dim_Y = int((self.dim_X + self.dim_Z) / 2)
        
        
        self.beta = 1 # setting beta to zero is equivalent to a normal autoencoder
        self.nu   = 4
        self.batch_wise = batch_wise
            
        
        # Tanh for now, but could also be ReLu, GeLu, tanh, etc
        self.encoder = self.construct_encoder(layers)
        self.decoder = self.construct_decoder(layers)
        
        
    def construct_encoder(self, layers):
        """
        Generates the encoder neural net dynamically based on layers parameter

        Parameters
        ----------
        layers : int, amount of layers, same as the decoder

        Returns
        -------
        instantiation of the nn.Sequential class, with the appropriate amount
        of layers

        """
        network = OrderedDict()
        network['0'] = nn.Linear(self.dim_X, self.dim_Y)
        network['1'] = nn.Tanh()
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.Tanh()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_Z)
        
        return nn.Sequential(network)
    
        
    def construct_decoder(self, layers):
        """
        Generates the decoder neural net dynamically based on layers parameter

        Parameters
        ----------
        layers : int, amount of layers, same as the enoder

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        network = OrderedDict()
        network['0'] = nn.Linear(self.dim_Z, self.dim_Y)
        network['1'] = nn.Tanh()
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.Tanh()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_X)
        
        return nn.Sequential(network)
    
    def standardize_X(self, X):
        """
        Class method that stores the mean and variances of the given data for 
        later unstandardisation, and standardizes the data

        Parameters
        ----------
        X : multidimensional float tensor

        Returns
        -------
        Standardized version of multidimensional float tensor

        """
        # write code that stores mean and var, so u can unstandardize X_prime
        self.means_vars_X = (X.mean(axis=0), X.std(axis=0))
        
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def unstandardize_Xprime(self, X_prime):
        """
        Using previously stores means and variances, unstandardize the predicted
        data

        Parameters
        ----------
        X_prime : multidimensial float tensor

        Returns
        -------
        Rescaled multidimensional float tensor

        """
        return (X_prime * self.means_vars_X[1] + self.means_vars_X[0])
    
    def force_tensor(self, X):
        """
        forces the given object into a float tensor

        Parameters
        ----------
        X : np.array or pd.DataFrame of data

        Returns
        -------
        float tensor of given data

        """
        # write code that forces X to be a tensor
        if type(X) != torch.Tensor:
            return torch.Tensor(X).float()
        else:
            return X.float() # force it to float anyway
    
    def forward(self, data):
        """
        Function that standardizes the given data, and feeds it through the 
        architecture

        Parameters
        ----------
        data : Multidimensional array of data, has to match the model 
        instantiation in terms of feature count

        Returns
        -------
        Data that has been fed through the model

        """
        if self.X.shape[1] != data.shape[1]:
            print('data does not match instantiation data in feature count')
            return None
        
        data = self.standardize_X(self.force_tensor(data))
        
        return self.unstandardize_Xprime(self.decoder(self.encoder(data))).detach().numpy()
        
    def ANLL(self, z):
        """
        Function that calculates log likelihood based on latent space distribution
        and reduced data z

        Parameters
        ----------
        z : reduced data after encoding data

        Returns
        -------
        Average negative log likelihood

        """
        
        LL = 0
        for row in range(self.n):
            fx = (1 + 1/self.n*(z[row,:]@torch.inverse(self.covar)@z[row,:]))**((-self.n+self.dim_Z)/2)
            LL += torch.log(self.c*fx)
        
        return -1*LL/self.n

    
    def RE_LL_metric(self, epoch):
        """
        Function that calculates the loss of the autoencoder by
        RE and LL. 

        Returns
        -------
        tuple of RE and LL

        """
        # batch-wise optimisation
        batch = int(self.X.shape[0]/100)
        epoch_scale_threshold = 0.8
        
        if self.X.shape[0] < 1000:
            self.batch_wise = False
        
        if epoch > self.epochs * epoch_scale_threshold:
            batch += int((self.X.shape[0]-batch) / 
                         (self.epochs - self.epochs*epoch_scale_threshold) * 
                         (epoch-self.epochs*epoch_scale_threshold))
        
        if self.batch_wise == True:
            X = self.X[torch.randperm(self.X.shape[0])[0:batch],:]
            self.n = X.shape[0]
        else:
            X = self.X

        
        z       = self.encoder(self.X)
        
        x_prime = self.decoder(z)
        
        # get negative average log-likelihood here
        LL = self.ANLL(z)
        
        self.REs = (self.X - x_prime)**2
        RE = self.REs.mean() # mean squared error of reconstruction
        
        return (RE, LL) # function stolen from Bergeron et al. (2021) 

    
    def loss_function(self, RE_LL):
        """
        function that reconciles RE and LL in loss equation

        Parameters
        ----------
        RE_LL : tuple of RE and LL

        Returns
        -------
        calculated loss as a product of RE and LL

        """
        # return RE_LL[0] + self.beta * RE_LL[1]
        return RE_LL[0]/ 2 * RE_LL[0]**2 + RE_LL[1]
    
    def fit(self, epochs):
        """
        Function that fits the model based on previously passed data
        """
        from tqdm import tqdm
        
        self.train() # turn into training mode
        REs  = []
        LLs  = []
        
        optimizer = torch.optim.AdamW(self.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8) # specify some hyperparams for the optimizer
        
        self.covar = torch.Tensor(np.eye(self.dim_Z)) * (self.nu/(self.nu-2))
        
        self.c  = (((self.nu*np.pi)**(-self.dim_Z/2)) * float(mpmath.gamma(self.nu/2 + self.dim_Z/2)/mpmath.gamma(self.nu/2)) * 
              torch.det(self.covar)**-0.5)
        
        self.epochs = epochs
        
        for epoch in tqdm(range(self.epochs)):
            RE_LL = self.RE_LL_metric(epoch) # store RE and KL in tuple
            loss = self.loss_function(RE_LL) # calculate loss function based on tuple
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            REs += [RE_LL[0].detach().numpy()]
            LLs += [RE_LL[1].detach().numpy()] # RE and KLs are stored for analysis
        
        plt.plot(range(epochs), REs)
        plt.title('Reconstruction errors')
        plt.show()
        plt.plot(range(epochs), LLs)
        plt.title('neg avg LLs')
        plt.show()
        self.eval() # turn back into performance mode
        self.done()
        
        return
    
    def done(self):
        import win32api
        win32api.MessageBox(0, 'The model is done calibrating :)', 'Done!', 0x00001040) 
        return