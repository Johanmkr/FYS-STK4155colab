import numpy as np
import sys
import os
import pandas as pd


from plot import *

from src.designMatrix import DesignMatrix
from src.Regression import LeastSquares
from src.Resampling import Bootstrap, CrossValidation
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

z = FrankeFunction(x, y) + noise(0)
polydegs = range(1,20+1)

mse_train = []
mse_cv = []
r2_train = []
r2_cv = []
var_cv = []
bias_cv = []



for n in polydegs:
    dM = DesignMatrix(n)
    dM.createX(x, y) 
    reg = LeastSquares(x, dM)
    CV = CrossValidation(reg)
    list_w_stuff = CV(k_folds=10)
    for i, nrlist in enumerate([mse_train, mse_cv, r2_train, r2_cv, var_cv, bias_cv]):
        nrlist.append(list_w_stuff[i])

plt.figure()
plt.plot(polydegs, mse_train, ls="-", label='train MSE')
plt.plot(polydegs, mse_cv, ls="--", label='test MSE')
plt.legend()

plt.figure()
plt.plot(polydegs, r2_train,  ls="-", label='train $R^2$')
plt.plot(polydegs, r2_cv,  ls="--", label='test $R^2$')
plt.legend()

plt.figure()
plt.plot(polydegs, mse_cv, label="mse")
plt.plot(polydegs, var_cv, label="var")
plt.plot(polydegs, bias_cv, label="bias")
plt.legend()

plt.show()
