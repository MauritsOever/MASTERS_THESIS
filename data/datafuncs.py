# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:31:58 2022

@author: gebruiker
"""
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

def GenerateStudentTData(list_of_tuples, n):
    """
    Parameters
    ----------
    list_of_tuples : the tuples contain the mean, var and degrees of freedom of 
                     the variables, the length of the list determines the 
                     amount of variables
                     
    n              : int, amount of observations

    Returns
    -------
    an m by n array of uncorrelated student t data (diagonal covar matrix)
    
    t.pdf(x, df, loc, scale)

    """
    import numpy as np
    from scipy.stats import t
    array = np.random.uniform(size = (n, len(list_of_tuples)))
    
    for column in range(len(list_of_tuples)):
        loc   = list_of_tuples[column][0]
        scale = list_of_tuples[column][1]
        df    = list_of_tuples[column][2]
        
        array[:, column] = t.ppf(array[:, column], df = df, loc = loc, scale = scale)
    
    return array
    
def GetData(datatype):
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
    
    if datatype == 'normal':
        n = 100000
        list_of_tuples = [(0,1), (-0.5,0.01), (6,12), (80,10), (-10,6), (100,85)]
        return GenerateNormalData(list_of_tuples, n)
    
    elif datatype == 't':
        n = 100000
        list_of_tuples = [(0,1,100), (-0.5,0.01,4), (6,12,50), (80,10,3), (-10,6,75), (100,85,25)]
        return GenerateStudentTData(list_of_tuples, n)
    
    elif datatype == 'returns':
        list_of_ticks = ['AAPL', 'MSFT', 'KO', 'PEP', 'MS', 'GS', 'WFC', 'TSM']
        startdate     = '2010-01-01'
        enddate       = '2020-12-31'
        return np.array(Yahoo(list_of_ticks, startdate, enddate).iloc[1:, :])
    
    else:
        print('datatype not recognized, please consult docstring for information on valid data types')
