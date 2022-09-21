import numpy as np
import sys
import os
import pandas as pd

from sklearn.utils import check_random_state

ourSeed = 7132
np.random.seed(ourSeed)
randomState = check_random_state(ourSeed)





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
    p = int((n+1)*(n+2)/2)
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
    p = features
    coeff = [1, 3, 2-2*p]
    ns = np.roots(coeff)
    n = int(np.max(ns))
    return n
