# -*- coding: utf-8 -*-
"""
Trying to implement some VAEs over here...
Useless:
https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html#3.-Autoencoder-with-Scikit-Learn


Useful (maybe):
https://keras.io/examples/generative/vae/ regular VAE
https://github.com/AntixK/PyTorch-VAE
https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
https://colab.research.google.com/github/tvhahn/Manufacturing-Data-Science-with-Python/blob/master/Metal%20Machining/1.B_building-vae.ipynb
https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
https://www.youtube.com/watch?v=A6mdOEPGM1E regular VAE
https://github.com/fpcasale/GPPVAE gaussian
https://jmetzen.github.io/2015-11-27/vae.html not entirely gaussian
https://jaketae.github.io/study/vae/ gaussian diagonal
https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/ but generalise to complicated likelihoods 
https://towardsdatascience.com/on-distribution-of-zs-in-vae-bb9a05b530c6 on the KL divergence
https://stats.stackexchange.com/questions/306640/learning-normal-distribution-with-vae probs useless
https://libs.garden/python/psanch21/VAE-GMVAE what is gaussian mixture??
https://pyro.ai/examples/vae.html examples
https://github.com/takahashihiroshi/t_vae main hopes on this one lmao

MAIN IDEA:
    - make a class that contains all the methods of fitting and generating a VAE both for normal and student t

Created on Fri Mar  4 11:16:29 2022

@author: MauritsOever
"""



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
list_of_tuples = [(0,1),(1,0.01),(40,12),(0,1)]
test = GenerateNormalData(list_of_tuples, n)
    
    
    
    
    