# -*- coding: utf-8 -*-
"""
Functions that simulates and loads in data for the thesis

To do:
    - finalize simulated simple data, normal is done, only t
    - make a dataset that has non-linear dependencies, maybe categorical data etc
    - 

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
    
    # for col in range(len(list_of_tuples)):
    #     array[:,col] = array[:,col] * list_of_tuples[col][1] + list_of_tuples[col][0]
    
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
  
    array = np.empty((n,len(list_of_tuples)))
    
    for variable in range(len(list_of_tuples)):
        array[:,variable] = np.random.standard_t(list_of_tuples[variable][2], n)
        # array[:,variable] = np.random.normal(0, 1, n)

        
    amount_of_cols_per_dim = int(len(list_of_tuples) / correlated_dims)
    
    counter = 0
    for i in range(0, correlated_dims):
        chi_square = np.random.chisquare(list_of_tuples[i][2])
        for col in range(1, amount_of_cols_per_dim):
            array[:,counter+col] = rho*array[:,counter] + np.sqrt(1-rho**2)* array[:,counter+col]
            # array[:,counter+col] = array[:,counter+col] / np.sqrt(chi_square/list_of_tuples[i][2])
        counter += amount_of_cols_per_dim
    
    for col in range(len(list_of_tuples)):
        array[:,col] = array[:,col] * list_of_tuples[col][1] + list_of_tuples[col][0] # scale
    
    return array

def GenerateMixOfData(n, rho):
    """
    

    Returns
    -------
    None.

    """
    import numpy as np
    from scipy.stats import bernoulli
    array = np.zeros((n,12))
    
    
    # normal correlated
    list_of_tuples = [(0,1), (-0.5,0.01), (6,12)]
    array[:,0:3] = GenerateNormalData(list_of_tuples, n, 1, rho)
    
    # student t correlated
    list_of_tuples = [(0,1,8), (-0.5,0.01,4), (6,12,5)]
    array[:,3:6] = GenerateStudentTData(list_of_tuples, n, 1, rho)
    
    # bernoulli correlated
    array3 = bernoulli.rvs(0.5, size=(n,3))
    for row in range(5,n):
        corrs = np.corrcoef(array3[0:row,:], rowvar=False)
        if corrs[0,1] < rho:
            array3[row,1] = array3[row,0]
        if corrs[0,2] < rho:
            array3[row,2] = array3[row,0]
        if corrs[1,2] < rho:
            array3[row,2] = array3[row,1]
    
    array[:,6:9] = array3
    
    # other non-linear copula
    

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
    dfclose = dfclose.ffill()
    dfclose = dfclose.backfill()
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
    n = 10000
    
    if datatype == 'normal':
        list_of_tuples = [(0,1), (-0.5,0.01), (6,12), (80,10), (-10,6), (100,85),
                          (25, 5), (36, 6), (2, 1), (73, 30), (-10,2.5), (-20, 4)]
        return GenerateNormalData(list_of_tuples, n, correlated_dims, rho)
    
    elif datatype == 't':
        list_of_tuples = [(0,1,4), (-0.5,0.01,4), (6,12,5), (80,10,3), (-10,6,6), (100,85,4.5),
                          (25, 5,5), (36, 6, 6), (2, 1, 8), (73, 30, 5), (-10,2.5,10), (-20, 4, 4.44)]
        return GenerateStudentTData(list_of_tuples, n, correlated_dims, rho)
    
    elif datatype == 'returns':
        list_of_ticks = ['NSRGY', 'VWS.CO', 'BCS', 'ING', 'STM', 'DB', 'VWAGY', 
                         'GMAB.CO', 'BP', 'HM-B.ST', 'SAN', 'MPCK.HA', 'POL.OL', 'TIS.MI']
        startdate     = '2001-01-01'
        enddate       = '2021-12-31'
        return np.array(Yahoo(list_of_ticks, startdate, enddate).iloc[1:, :])
    elif datatype == 'mix':
        return GenerateMixOfData(n,rho)
    elif datatype == 'interestrates':
        print('This is gonna be a feature, but its not done yet!')
    else:
        print('datatype not recognized, please consult docstring for information on valid data types')
