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
        - optimize hyperparams
    """
    
    def __init__(self, X, dim_Z, layers=3, standardize = True, batch_wise=True, done=False, plot=False):
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

        self.standardize = standardize
        self.dim_X = X.shape[1]
        self.dim_Z = dim_Z
        self.dim_Y = int((self.dim_X + self.dim_Z) / 2)
        self.n     = X.shape[0]
        self.done_bool = done
        self.plot = plot
        
        self.beta = 1 # setting beta to zero is equivalent to a normal autoencoder
        self.batch_wise = batch_wise
            
        # LeakyReLU for now
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
        network['1'] = nn.LeakyReLU() 
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.LeakyReLU()
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
        network['1'] = nn.LeakyReLU()
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.LeakyReLU()
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
        try:
            return X / self.maxs
        except:
            self.maxs = torch.max(X, dim=0)[0]
            return X / self.maxs
        
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
        return X_prime * self.maxs
    
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
        
        if self.standardize:
            data = self.standardize_X(self.force_tensor(data))
            return self.unstandardize_Xprime(self.decoder(self.encoder(data))).detach().numpy()
        else:
            data = self.force_tensor(data)
            return self.decoder(self.encoder(data)).detach().numpy()
        
    def MM(self, z):
        # UNIVARIATE SEPARATE 
        
        means = z.mean(dim=0)
        diffs = z - means
        std = z.std(dim=0)
        zscores = diffs / std
        skews = (torch.pow(zscores, 3.0)).mean(dim=0)
        kurts = torch.pow(zscores, 4.0).mean(dim=0) - 3
        
        mean_score = (means**2).mean()
        std_score = ((std - torch.Tensor([1]*self.dim_Z))**2).mean()
        skew_score = (skews**2).mean()
        kurt_score = (kurts**2).mean()
    
        # MULTIVARIATE
        # cov_z = torch.cov(z.T)
        
        # # first moment, expected value of all variables
        # mean_score = torch.linalg.norm(z.mean(dim=0), ord=2)
        
        # # second moment
        # std_score = torch.linalg.norm(cov_z - torch.eye(z.shape[1]), ord=2)
        
        # # third and fourth moment
        # Y = torch.t(torch.linalg.inv(torch.linalg.cholesky(cov_z))@torch.t(z - z.mean(dim=0)))
        
        # kron3 = torch.empty((Y.shape[0], Y.shape[1]**3))
        # vec   = torch.empty(Y.shape[0])
        
        # for row in range(Y.shape[0]):
        #     kron3[row,:] = torch.kron(Y[row,:], torch.kron(Y[row,], Y[row,:]))
        #     vec[row]     = Y[row,:]@torch.t(Y[row,:])
        
        # skew_score = torch.linalg.norm(kron3.mean(dim=0), ord=2) # works but subject to sample var
        # kurt_score = torch.mean(vec - 3)
        
        return mean_score + std_score + skew_score + kurt_score
    
    
    def RE_MM_metric(self, epoch):
        """
        Function that calculates the loss of the autoencoder by
        RE and MM. 

        Returns
        -------
        tuple of RE and MM

        """
        # batch-wise optimisation
        # batch = int(self.X.shape[0]/100)
        batch = 500
        
        epoch_scale_threshold = 0.95
        
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
        
        z       = self.encoder(X)
        
        x_prime = self.decoder(z)
        
        # get negative average log-likelihood here
        MM = self.MM(z)

        weights = (X[:,2] * self.maxs[2])**4 
        
        self.REs = (X - x_prime)**2
        RE = (self.REs[:,0] * weights).mean() # mean squared error of reconstruction, 
                                                # but only of the curves, and weighted with numOptions
                                                
        self.weighted_RE = RE
        # RE = self.REs[:,0].mean()
        return (RE, MM)

    
    def loss_function(self, RE_MM):
        """
        function that reconciles RE and MM in loss equation

        Parameters
        ----------
        RE_MM : tuple of RE and MM

        Returns
        -------
        calculated loss as a product of RE and MM

        """
        return RE_MM[0] + self.beta * RE_MM[1]
        # return RE_MM[0]/ 2 * RE_MM[0]**2 + RE_MM[1]

    
    def fit(self, epochs):
        """
        Function that fits the model based on instantiated data
        """
        from tqdm import tqdm
        
        self.train() # turn into training mode
        REs  = []
        MMs  = []
        
        optimizer = torch.optim.AdamW(self.parameters(),
                             lr = 0.001,
                             weight_decay = 1e-2) # specify some hyperparams for the optimizer
        
        
        self.epochs = epochs
        
        REs = np.zeros(epochs)
        MMs = np.zeros(epochs)
        
        for epoch in tqdm(range(epochs)):
        # for epoch in range(epochs):
            RE_MM = self.RE_MM_metric(epoch) # store RE and KL in tuple
            loss = self.loss_function(RE_MM) # calculate loss function based on tuple
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            REs[epoch] = RE_MM[0].detach().numpy()
            MMs[epoch] = RE_MM[1].detach().numpy() # RE and KLs are stored for analysis
        if self.plot:
            plt.plot(range(epochs), REs)
            plt.title('Reconstruction errors')
            plt.show()
            plt.plot(range(epochs), MMs)
            plt.title('avg MMs')
            plt.show()
        self.eval() # turn back into performance mode
        if self.done_bool:
            self.done()
        
        return 
    
    def done(self):
        import win32api
        win32api.MessageBox(0, 'The model is done calibrating :)', 'Done!', 0x00001040)
        return
    
    def fit_garch_latent(self, epochs=None):
        from models.MGARCH import robust_garch_torch

        data = self.encoder(self.X) # latent data from fitted autoencoder

        garch = robust_garch_torch(data, dist='norm')
        if epochs == None:
            epochs = 100
        
        garch.fit(epochs)
        garch.store_sigmas()
        self.garch = garch
        
        # from models.MGARCH import DCC_garch
        # data = self.encoder(self.X).detach().numpy()
        
        # garch = DCC_garch(dist='norm')
        # garch.fit(data)
        # garch.predict()
        # garch.sigmas = []
        # garch.sigmas = torch.Tensor(garch.H_t)
        
        # garch.sigmas[0] = garch.sigmas[1]
        
        # self.garch = garch
        return
    
    def latent_GARCH_HS(self, data = None, q=0.05):
        """
        Simulate data and take VaR and ES for all days in the data
        
        Parameters
        ----------
        data : float tensor, np array or pd dataframe of data. if data is passed then analysis is out-of-sample

        Returns
        -------
        None.

        """
        from tqdm import tqdm
        n = 1000
        
        try:
            if data == None:
                sigmas = self.garch.sigmas
            else:
                X = self.standardize_X(self.force_tensor(data))
                z = self.encoder(X)
                sigmas = self.garch.estimate_sigmas(z)
        except:
            print("Error: garch is not yet fitted")
            return
        
        VaRs = torch.empty((len(sigmas), self.dim_X))
        # ESs  = torch.empty((len(sigmas), self.dim_X))
        for i in tqdm(range(len(sigmas))):
        # for i in range(len(sigmas)):
            l = torch.linalg.cholesky(sigmas[i])
            sims = torch.randn((n, sigmas[0].shape[0]))
            for row in range(n):
                sims[row] = l@sims[row]
            
            # put through decoder    
            Xsims = self.unstandardize_Xprime(self.decoder(sims))
            # take quantile
            VaRs[i,:] = torch.quantile(Xsims, q, dim=0)
            # return time series of quantiles
            # for col in range(Xsims.shape[1]):
            #     ESs[i, col] = torch.mean(Xsims[Xsims[:,col]<VaRs[i,col],col])
            del sims
                
        return VaRs #, ESs


            