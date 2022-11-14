# Imports 
from time import time
startTime = time()


import numpy as np
import sys
import os
from IPython import embed
import pandas as pd

from copy import deepcopy
from collections.abc import Callable
import numpy.typing as npt
from tqdm import trange

# from sklearn.utils import check_random_state



# Deterministic randomised data
ourSeed = 1697
np.random.seed(ourSeed)
# randomState = check_random_state(ourSeed)

ndarray = npt.NDArray[np.float64]


def Z_score_normalise(X:np.ndarray, y:np.ndarray, keepdims:bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalised the input and output data.

    Args:
        X (ndarray): Input data.
        y (ndarray): Output data.
        keepdims (bool, optional): If true, dimensions are kept constant. Defaults to True.

    Returns:
        tuple[ndarray, ndarray]: Scaled X, y
    """
    X_mu = np.mean(X, axis=0, keepdims=keepdims)
    X_sigma = np.std(X, axis=0, keepdims=keepdims)

    y_mu = np.mean(y)
    y_sigma = np.std(y)

    X -= X_mu; X /= X_sigma
    y -= y_mu; y /= y_sigma

    return X, y


def train_test_split(X:np.ndarray, y:np.ndarray, train_size:float=0.80, seed:int=200) -> tuple[np.ndarray]:
    """Splits the input and corresponding output data into a training set and a testing set. 

    Args:
        X (ndarray): Input data.
        y (ndarray): Output data.
        train_size (float, optional): Fraction of data to be used for training. Defaults to 0.80.
        seed (int, optional): Random seed. Defaults to 200.

    Returns:
        tuple[np.ndarray]: X_train, y_train, X_test, y_test
    """
    np.random.seed(seed)
    n_obs = np.shape(X)[0]
    indices = np.arange(n_obs)
    np.random.shuffle(indices)
    cut = int(n_obs*train_size)
    train_indices = indices[:cut]
    test_indices = indices[cut:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test



def Z_score_normalise_split(X:np.ndarray, y:np.ndarray, train_size:float=0.80, seed:int=200) -> tuple[np.ndarray]:
    """Splits input and output data into training and test data, while also performing Z-score normalisation.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Output data.
        train_size (float, optional): Fraction of data to be used for training. Defaults to 0.80.
        seed (int, optional): Random seed. Defaults to 200.

    Returns:
        tuple[np.ndarray]: X, y, X_train, y_train, X_test, y_test
    """
    np.random.seed(seed)
    n_obs = np.shape(X)[0]
    indices = np.arange(n_obs)
    np.random.shuffle(indices)
    cut = int(n_obs*train_size)
    train_indices = indices[:cut]
    test_indices = indices[cut:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    # sort!
    
    X_mu = np.mean(X_train, axis=0, keepdims=True)
    X_sigma = np.std(X_train, axis=0, keepdims=True)
    y_mu = np.mean(y_train)
    y_sigma = np.std(y_train)

    X_train -= X_mu; X_train /= X_sigma
    y_train -= y_mu; y_train /= y_sigma
    X_test -= X_mu; X_test /= X_sigma
    y_test -= y_mu; y_test /= y_sigma
    X -= X_mu; X /= X_sigma
    y -= y_mu; y /= y_sigma


    return X, y, X_train, y_train, X_test, y_test

def feature_scale_split(X:np.ndarray, y:np.ndarray, train_size:float=0.80) -> tuple[np.ndarray]:
    """Scales only the features of the input data, while splitting everything into train and test.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Output data.
        train_size (float, optional): Fraction of data to be used for training. Defaults to 0.80.

    Returns:
        tuple[np.ndarray]: X, y, X_train, y_train, X_test, y_test
    """
    n_obs = np.shape(X)[0]
    indices = np.arange(n_obs)
    np.random.shuffle(indices)
    cut = int(n_obs*train_size)
    train_indices = indices[:cut]
    test_indices = indices[cut:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    X_mu = np.mean(X_train, axis=0, keepdims=True)
    X_sigma = np.std(X_train, axis=0, keepdims=True)

    X_train -= X_mu; X_train /= X_sigma
    X_test -= X_mu; X_test /= X_sigma
    X -= X_mu; X /= X_sigma

    return X, y, X_train, y_train, X_test, y_test