# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:31:58 2022

@author: gebruiker
"""
def GenerateNormalData(list_of_tuples, n, correlated_dims, rho):
    """
    Parameters
    ----------
    list_of_tuples : the tuples contain the mean and var of the variables, the length
                     the list determines the amount of variables
    n              : int, amount of observations

    Returns
    -------
    an m by n array of correlated normal data (diagonal covar matrix)

    """
    import numpy as np
  
    array = np.empty((n,len(list_of_tuples)))
    
    for variable in range(len(list_of_tuples)):
        array[:,variable] = np.random.normal(0, 1, n)
        
    amount_of_cols_per_dim = int(len(list_of_tuples) / correlated_dims)
    
    counter = 0
    for i in range(0, correlated_dims):
        for col in range(1, amount_of_cols_per_dim):
            array[:,counter+col] = rho*array[:,counter] + np.sqrt(1-rho**2)* array[:,counter+col]
        counter += amount_of_cols_per_dim
    
    for col in range(len(list_of_tuples)):
        array[:,col] = array[:,col] * list_of_tuples[col][1] + list_of_tuples[col][0]
    
    return array

def GenerateStudentTData(list_of_tuples, n, correlated_dims, rho):
    """
    Parameters
    ----------
    list_of_tuples : the tuples contain the mean, var and degrees of freedom of 
                     the variables, the length of the list determines the 
                     amount of variables
                     
    n              : int, amount of observations

    Returns
    -------
    an m by n array of correlated student t data (diagonal covar matrix)
    
    t.ppf(x, df, loc, scale)

    """
    import numpy as np
    from scipy.stats import t
    array = np.random.uniform(size = (n, len(list_of_tuples)))
    
    for column in range(len(list_of_tuples)):
        df    = list_of_tuples[column][2]
        array[:, column] = t.ppf(array[:, column], df = df, loc = 0, scale = 1)
        
    amount_of_cols_per_dim = int(len(list_of_tuples) / correlated_dims)
    
    counter = 0
    for i in range(0, correlated_dims):
        for col in range(1, amount_of_cols_per_dim):
            array[:,counter+col] = rho*array[:,counter] + np.sqrt(1-rho**2)* array[:,counter+col]
        counter += amount_of_cols_per_dim
    
    return array

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

    
def GetData(datatype, correlated_dims, rho):
    """
    Generates a data array based on input

    Parameters
    ----------
    datatype : string, choose between 'normal', 't', 'returns'

    Returns
    -------
    array of generated or downloaded data

    """
    import numpy as np
    n = 1000
    
    if datatype == 'normal':
        list_of_tuples = [(0,1), (-0.5,0.01), (6,12), (80,10), (-10,6), (100,85),
                          (25, 5), (36, 6), (2, 1), (73, 30), (-10,2.5), (-20, 4)]
        return GenerateNormalData(list_of_tuples, n, correlated_dims, rho)
    
    elif datatype == 't':
        list_of_tuples = [(0,1,100), (-0.5,0.01,4), (6,12,50), (80,10,3), (-10,6,75), (100,85,25),
                          (25, 5,5), (36, 6, 6), (2, 1, 8), (73, 30, 10), (-10,2.5,15), (-20, 4, 20)]
        return GenerateStudentTData(list_of_tuples, n, correlated_dims, rho)
    
    elif datatype == 'returns':
        list_of_ticks = ['AAPL', 'MSFT', 'KO', 'PEP', 'MS', 'GS', 'WFC', 'TSM']
        startdate     = '2010-01-01'
        enddate       = '2020-12-31'
        return np.array(Yahoo(list_of_ticks, startdate, enddate).iloc[1:, :])
    elif datatype == 'interestrates':
        print('This is gonna be a feature, but its not done yet!')
    else:
        print('datatype not recognized, please consult docstring for information on valid data types')
