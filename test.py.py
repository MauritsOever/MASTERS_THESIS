# -*- coding: utf-8 -*-
"""
Trying to implement some VAEs over here...
https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html#3.-Autoencoder-with-Scikit-Learn

Created on Fri Mar  4 11:16:29 2022

@author: MauritsOever
"""

from sknn import ae
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np

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
    
    
# data definition
data  = Yahoo(['AAPL', 'ZION', 'NVDA', 'EBAY', 'MSFT', 'IPG'], startdate = '2000-01-01', enddate='2022-01-01').iloc[1:, :]
data = data.ffill()

# Shape of input and latent variable

n_input = 28*28

# Encoder structure
n_encoder1 = 6
n_encoder2 = 4

n_latent = 2

# Decoder structure
n_decoder2 = 4
n_decoder1 = 6


reg = MLPRegressor(hidden_layer_sizes = (n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1), 
                   activation = 'tanh', 
                   solver = 'adam', 
                   learning_rate_init = 0.0001, 
                   max_iter = 10000000, 
                   tol = 0.0000001, 
                   verbose = True)

reg.fit(data, data)


# getting some info on the latent data
def encoder(data):
    data = np.asmatrix(data)
    
    encoder1 = data*reg.coefs_[0] + reg.intercepts_[0]
    encoder1 = (np.exp(encoder1) - np.exp(-encoder1))/(np.exp(encoder1) + np.exp(-encoder1))
    
    encoder2 = encoder1*reg.coefs_[1] + reg.intercepts_[1]
    encoder2 = (np.exp(encoder2) - np.exp(-encoder2))/(np.exp(encoder2) + np.exp(-encoder2))
    
    latent = encoder2*reg.coefs_[2] + reg.intercepts_[2]
    latent = (np.exp(latent) - np.exp(-latent))/(np.exp(latent) + np.exp(-latent))
    
    return np.asarray(latent)

test_latent = encoder(data)

plt.figure(figsize = (10,10))
#plt.scatter(test_latent[np.argmax(data, axis = 1) == 1,0], test_latent[np.argmax(data, axis = 1) == 1,1], label = '1')

plt.scatter(test_latent[:,0], test_latent[:,1])
plt.title('Latent Space', fontsize=15)
plt.xlabel('Z1', fontsize=15)
plt.ylabel('Z2', fontsize=15)
plt.legend(fontsize = 15)
plt.axis('equal')
plt.show()






# data generation

