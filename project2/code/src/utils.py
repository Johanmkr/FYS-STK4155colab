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


def Z_score_normalise(X:ndarray, y:ndarray, keepdims:bool=True) -> tuple[ndarray, ndarray]:
    X_mu = np.mean(X, axis=0, keepdims=keepdims)
    X_sigma = np.std(X, axis=0, keepdims=keepdims)

    y_mu = np.mean(y)
    y_sigma = np.std(y)

    X -= X_mu; X /= X_sigma
    y -= y_mu; y /= y_sigma

    return X, y


def Z_score_normalise_split(X:ndarray, y:ndarray, train_size=0.80) -> tuple[ndarray, ndarray]:
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