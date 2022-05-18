# -*- coding: utf-8 -*-
"""
Code solely to test other files

Created on Mon Apr 25 17:14:17 2022

@author: MauritsvandenOeverPr
"""
import os
os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')

from models.Gauss_VAE import GaussVAE
#from models.GaussMix_VAE import GaussMixVAE
from models.StudentT_VAE import StudentTVAE
from data.datafuncs import GetData, GenerateAllDataSets
import torch
import numpy as np
import pandas as pd
import mpmath
import matplotlib.pyplot as plt
from scipy import stats

# rho = 0.75
# correlated_dims = 4
dim_Z = 4
# datatype = 'normal'

# clean and write
X = pd.read_csv(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS\data\datasets\real_sets\MO_THESIS.03.csv').drop(0, axis=0)
X = X.ffill()
X = X.backfill()
X = np.array(X.iloc[:,1:])
X = X.astype(float)
X = np.log(X[1:,:]) - np.log(X[:-1,:])
# X = GetData('t', 4, 0.75)

# model = GaussVAE(X, dim_Z)
model = GaussVAE(X, dim_Z, layers=4, batch_wise=False)

model.fit(epochs=10000)

z = model.encoder(model.X).detach().numpy()

print(f'means are {z.mean(axis=0)}')
print(f'stds are {z.std(axis=0)}')
print(f'skews are {stats.skew(z)}')
print(f'kurts are {stats.kurtosis(z)}')
print('')

plt.hist(z)

print(f'jb test of col 1 {stats.jarque_bera(z[:,0])}')
print(f'jb test of col 2 {stats.jarque_bera(z[:,1])}')
print(f'jb test of col 3 {stats.jarque_bera(z[:,2])}')
print(f'jb test of col 4 {stats.jarque_bera(z[:,3])}')

#%% okay now check if Z_extreme == X_extreme
z_low  = z[z[:,0] < np.quantile(z[:,0], 0.10), :]
z_high = z[z[:,0] > np.quantile(z[:,0], 0.90), :]
z_mid  = np.empty((0,4))
z_extreme = np.empty((0,4))

selector_mid = np.full(z.shape[0], False)
selector_extreme = np.full(z.shape[0], False)
for row in range(z.shape[0]):
    selector_mid[row] = all([z[row,0] > np.quantile(z[:,0], 0.45), z[row,0] < np.quantile(z[:,0], 0.55)])
    selector_extreme[row] = any([z[row,0] < np.quantile(z[:,0], 0.05), z[row,0] > np.quantile(z[:,0], 0.95)])

z_mid = z[selector_mid, :]
z_extreme = z[selector_extreme, :]

# okay now do the check
X_low = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_low))).detach().numpy()
X_mid = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_mid))).detach().numpy()
X_high = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_high))).detach().numpy()
X_extreme = model.unstandardize_Xprime(model.decoder(torch.Tensor(z_extreme))).detach().numpy()
X_full = model.unstandardize_Xprime(model.decoder(torch.Tensor(z))).detach().numpy()

plt.hist(X_low)
plt.hist(X_mid)
plt.hist(X_high)
plt.hist(X_extreme)
plt.hist(X_full)
#%% 
# Y = torch.Tensor([[1,2,3],
#                   [4,5,6],
#                   [7,8,9]])

# Y = torch.Tensor([[1,2,3,4],
#                   [5,6,7,8],
#                   [9,10,11,12],
#                   [13,14,15,16]])

# import torch
# import numpy as np

# dim = 3
# Y = torch.randn((5000, dim))

# kron3 = torch.empty((Y.shape[0], Y.shape[1]**3))

# for row in range(Y.shape[0]):
#     kron3[row,:] = torch.kron(Y[row,:], torch.kron(Y[row,], Y[row,:]))

# kron3 = kron3.detach().numpy()


#kron_reg = torch.kron(torch.kron(Y, Y), Y).detach().numpy()
#kron_try = kron_reg[np.linspace(0, (Y.shape[0]**3)-1, num= Y.shape[0]).astype(int), :]

# list_of_bools = []
# for column in range(kron3.shape[1]):
#     list_of_bools += [all(kron3.round()[:,column] == kron_try.round()[:,column])]

# print(all(list_of_bools))

#%% 
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

class NeuralNet(nn.Module):

    def __init__(self, X):
        # imports
        super(NeuralNet, self).__init__()
        from collections import OrderedDict
        import numpy as np
        import torch
        from torch import nn
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        
        self.dim_Z = 2
        self.dim_X = X.shape[1]
        self.dim_Y = int((self.dim_Z + self.dim_X)/2)
        
        self.n = X.shape[0]
        
        self.encoder = self.construct_encoder()
        self.X = torch.Tensor(X).float()

    def construct_encoder(self):
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
        for i in range(3-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.LeakyReLU()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_Z)
        
        return nn.Sequential(network)
    
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
        covar = torch.Tensor(np.eye(self.dim_Z))
        
        LL = 0
        for row in range(self.n):
            LL += -0.5 * z[row,:]@torch.inverse(covar)@z[row,:]
            
        LL = LL + self.LL_constant * self.n
        
        return -1*LL/self.n
    
    def loss(self, z):
        # UNIVARIATE SEPARATE
        means = z.mean(dim=0)
        diffs = z - means
        var = torch.pow(diffs, 2.0)
        std = torch.pow(var, 0.5).mean(dim=0)
        zscores = diffs / std
        skews = (torch.pow(zscores, 3.0)).mean(dim=0)
        kurts = torch.pow(zscores, 4.0).mean(dim=0) - 3
        
        mean_score = (means**2).mean()
        std_score = (std**2).mean()
        skew_score = (skews**2).mean()
        kurt_score = (kurts**2).mean()
        
        # MULTIVARIATE
        # cov_z = torch.cov(torch.t(z))
        
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
    
    def done(self):
        import win32api
        win32api.MessageBox(0, 'The model is done calibrating :)', 'Done!', 0x00001040)
        return
        
    def fit(self):
        self.train() # turn into training mode
        epochs = 10000
        
        optimizer = torch.optim.AdamW(self.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8) # specify some hyperparams for the optimizer
        
        self.LL_constant = - self.n*self.dim_Z/2*torch.log(torch.tensor(2*np.pi)) - self.n/2*torch.log(torch.det(torch.Tensor(np.eye(self.dim_Z))))

        for epoch in tqdm(range(epochs)):
            # loss = self.ANLL(self.encoder(self.X)) # calculate loss function based on tuple
            loss = self.loss(self.encoder(self.X))
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        self.done()
            
X = torch.Tensor(np.random.standard_t(4, (1000, 4)))
model = NeuralNet(X)
model.fit()
z = model.encoder(model.X).detach().numpy()
X = X.detach().numpy()

print('')
print(f'means are {np.mean(z, axis=0)}, while X has {np.mean(X, axis=0)}')
print(f'stds are {np.std(z, axis=0)}, while X has {np.std(X, axis=0)}')
print(f'skews are {stats.skew(z)}, while X has {stats.skew(X)}')
print(f'kurts are {stats.kurtosis(z)}, while X has {stats.kurtosis(X)}')
