import numpy as np
import sys
import os
import pandas as pd


from plot import *

from src.designMatrix import DesignMatrix
from src.Regression import LeastSquares
from src.Resampling import Bootstrap
from src.betaMatrix import betaParameter, betaCollection



testSize = 1/5

Nx, Ny = 60, 60
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


noise = lambda eta: eta*np.random.randn(Ny, Nx)

