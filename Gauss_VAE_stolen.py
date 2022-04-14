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

from torch.utils.data import TensorDataset, DataLoader

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
    array = np.empty((n,len(list_of_tuples)))
    
    for variable in range(len(list_of_tuples)):
        array[:,variable] = np.random.normal(list_of_tuples[variable][0], list_of_tuples[variable][1], n)
        
    return array

n = 100000
list_of_tuples = [(0,1), (0,0.01), (0,12), (0,10), (0,6), (0,85)]

#%%
# create class that creates VAE based on hyperparameters
# mainly taking over github code and trying to understand it

class GaussianNetwork(nn.Module):
    def __init__(self, dim_X, dim_Z):
        """
        Initialises the class by creating Gaussian network that the other classes use

        Parameters
        ----------
        dim_X : dimensions of the original data, but can be coded to be found
        dim_Z : dimensions of the latent space
        """
        # see if my system supports GPU usage, otherwise just use CPU
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
    
    def reparameterize(self, mu, ln_var):
        # function required for self.forward()
        std = torch.exp(0.5 * ln_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def gaussian_nll(self, x, mu, ln_var, dim=1):
        # function required for self.forward()
        prec = torch.exp(-1 * ln_var)
        x_diff = x - mu
        x_power = (x_diff * x_diff) * prec * -0.5
        return torch.sum((ln_var + math.log(2 * math.pi)) * 0.5 - x_power, dim=dim)
    
    def standard_gaussian_nll(x, dim=1):
        return torch.sum(0.5 * math.log(2 * math.pi) + 0.5 * x * x, dim=dim)
    
    def gaussian_kl_divergence(mu, ln_var, dim=1):
        # function required for self.forward()
        return torch.sum(-0.5 * (1 + ln_var - mu.pow(2) - ln_var.exp()), dim=dim)
    
    def forward(self, x, k=1):
        # computes RE and KL, so we can tweak this to do what we want
        mu_enc, ln_var_enc = self.encode(x)

        RE = 0
        for i in range(k):
            z = self.reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            mu_dec, ln_var_dec = self.decode(z)
            RE += self.gaussian_nll(x, mu=mu_dec, ln_var=ln_var_dec) / k

        KL = self.gaussian_kl_divergence(mu=mu_enc, ln_var=ln_var_enc)
        return RE, KL
    
    def evidence_lower_bound(self, x, k=1):
        RE, KL = self.forward(x, k=k)
        return -1 * (RE + KL)
    
    def importance_sampling(self, x, k=1):
        # used much later to obtain test_score
        mu_enc, ln_var_enc = self.encode(x)
        lls = []
        for i in range(k):
            z = self.reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            mu_dec, ln_var_dec = self.decode(z)
            ll = -1 * self.gaussian_nll(x, mu=mu_dec, ln_var=ln_var_dec, dim=1)
            ll -= self.standard_gaussian_nll(z, dim=1)
            ll += self.gaussian_nll(z, mu=mu_enc, ln_var=ln_var_enc, dim=1)
            lls.append(ll[:, None])

        return torch.cat(lls, dim=1).logsumexp(dim=1) - math.log(k)
        
        
class GaussianVAE:    
    def __init__(self, dim_X, dim_Z):
        """
        Inherits from GaussianNetwork class, and also contains the fitting, 
        generation and prediction methods

        Parameters
        ----------
        dim_X : dimensions of the original data, but can be coded to be found
        dim_Z : dimensions of the latent space

        """
        
        # n_h          = int((dim_X - dim_Z) / 2)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = GaussianNetwork(dim_X, dim_Z)#.to(self.device)
        
        self.train_losses = []
        self.train_times  = []
        self.reconstruction_errors = []
        self.kl_divergences = []
        self.valid_losses = []
        self.min_valid_loss = float("inf")
    
    def _loss_function(self, x, k=1, beta=1):
        RE, KL = self.network(x, k=k)
        RE_sum = RE.sum()
        KL_sum = KL.sum()
        loss = RE_sum + beta * KL_sum
        return loss, RE_sum, KL_sum
    
    def _from_numpy(self, array, device, dtype=np.float32):
        array = torch.from_numpy(array.astype(dtype)).to(device) 
        return array
    
    def _to_numpy(self, tensor, device):
        if device.type == 'cuda': # not really necessary here
            tensor = tensor.cpu()
            
        return tensor.data.numpy()
    
    def _make_data_loader(self, array, device, batch_size):
        return DataLoader(
            TensorDataset(self._from_numpy(array, device)),
            batch_size=batch_size, shuffle=True)
    
    def _reparameterize(self, mu, ln_var):
        std = torch.exp(0.5 * ln_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    
    def fit(self, X, k=1, batch_size=100, learning_rate=0.001, n_epoch=500,
        warm_up=False, warm_up_epoch=100,
        is_stoppable=False, X_valid=None, path=None):

        self.network.train() # 'switch' to put network in training mode
        N = X.shape[0]
        data_loader = self._make_data_loader(X, device=self.device, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        if is_stoppable:
            X_valid = self._from_numpy(X_valid, self.device)

        for epoch in range(n_epoch):
            start = time.time()

            # warm-up
            beta = 1 * epoch / warm_up_epoch if warm_up and epoch <= warm_up_epoch else 1

            mean_loss = 0
            mean_RE = 0
            mean_KL = 0
            for _, batch in enumerate(data_loader):
                optimizer.zero_grad()
                loss, RE, KL = self._loss_function(batch[0], k=k, beta=beta)
                loss.backward()
                mean_loss += loss.item() / N
                mean_RE += RE.item() / N
                mean_KL += KL.item() / N
                optimizer.step()

            end = time.time()
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.reconstruction_errors.append(mean_RE)
            self.kl_divergences.append(mean_KL)

            print(f"epoch: {epoch} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f}", end='')

            if warm_up and epoch < warm_up_epoch:
                print(" / Warm-up", end='')
            elif is_stoppable:
                valid_loss, _, _ = self._loss_function(X_valid, k=k, beta=1)
                valid_loss = valid_loss.item() / X_valid.shape[0]
                print(f" / Valid: {valid_loss:0.3f}", end='')
                self.valid_losses.append(valid_loss)
                self._early_stopping(valid_loss, path)

            print('')

        if is_stoppable:
            self.network.load_state_dict(torch.load(path))

        self.network.eval() # 'switch' that puts network out of training mode
        
    def _early_stopping(self, valid_loss, path):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            torch.save(self.network.state_dict(), path)
            print(" / Save", end='')
    
    def encode(self, X):
        mu, ln_var = self.network.encode(self._from_numpy(X, self.device))
        return self._to_numpy(mu, self.device), self._to_numpy(ln_var, self.device)
    
    def decode(self, Z):
        mu, ln_var = self.network.decode(self._from_numpy(Z, self.device))
        return self._to_numpy(mu, self.device), self._to_numpy(ln_var, self.device)
    
    def reconstruct(self, X):
        mu_enc, ln_var_enc = self.network.encode(self._from_numpy(X, self.device))
        z = self._reparameterize(mu=mu_enc, ln_var=ln_var_enc)
        mu_dec, ln_var_dec = self.network.decode(z)
        return self._to_numpy(mu_dec, self.device), self._to_numpy(ln_var_dec, self.device)
        
    def evidence_lower_bound(self, X, k=1):
        return self._to_numpy(self.network.evidence_lower_bound(self._from_numpy(X, self.device), k=k), self.device)
    
    def importance_sampling(self, X, k=1):
        return self._to_numpy(self.network.importance_sampling(self._from_numpy(X, self.device), k=k), self.device)
    
    
        

#%% run the VAE here

X = GenerateNormalData(list_of_tuples, n)
dim_X = X.shape[1]

model = GaussianVAE(dim_X, dim_Z = 2)

learning_rate = 0.001

model.fit(X, k=1, batch_size=100,
              learning_rate=learning_rate, n_epoch=500,
              warm_up=False, is_stoppable=False,
              X_valid=X)



#%%

# .fit(data) method, which also reconstructs

# .predict(out_of_sample) method

# .generate() method, which generates data according to the latent distribution
