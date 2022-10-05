# Imports 
from time import time
startTime = time()

import numpy as np
import sys
import os
import pandas as pd

from copy import deepcopy

from sklearn.utils import check_random_state



# Deterministic randomised data
ourSeed = 7132
np.random.seed(ourSeed)
randomState = check_random_state(ourSeed)

# 3D extrapolation of Vandermonde matrix
maxPolydeg = 20
matrixColoumns = []

for i_ in range(1, maxPolydeg+1):
    for k_ in range(i_+1):
        matrixColoumns.append(f'x^{(i_-k_):1.0f} y^{k_:1.0f}')
matrixColoumns.append(' ')

'''
Useful functions
'''

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def polydeg2features(polydeg):
    """
    Function for finding the length of β for a given polynomial degree in 2 dimensions.

    Parameters
    ----------
    polydeg : int
        order of 2D polynomial 

    Returns
    -------
    int
        number of features in model
    """
    n = polydeg
    p = int((n+1)*(n+2)/2) - 1
    return p


def features2polydeg(features):
    """
    Function for finding the order of the 2D polynomial for a given length of β.

    Parameters
    ----------
    features : int
        number of features in model

    Returns
    -------
    int
        order of 2D polynomial
    """    
    p = features + 1
    
    coeff = [1, 3, 2-2*p]
    ns = np.roots(coeff)
    n = int(np.max(ns))
    if p == 153 or p ==152:
        n = 16 # ...
    return n


def featureMatrix_2Dpolynomial(x, y, max_polydeg=maxPolydeg):
    # plan to add option intercept coloumn
    xx = x.ravel(); yy = y.ravel()
    n = len(xx)
    cols = []
    idx = [f'({xi:6.3f}, {yi:6.3f})' for xi, yi in zip(xx, yy)]
    max_features = polydeg2features(max_polydeg)
    X = np.ones((len(xx), max_features))

    j = 0
    for i in range(1, max_polydeg+1):
        for k in range(i+1):
            X[:,j] = (xx**(i-k))*(yy**k)
            cols.append(f'x^{(i-k):1.0f} y^{k:1.0f}')
            j+=1

    Xpd = pd.DataFrame(data=X, columns=cols, index=idx)

    return Xpd
