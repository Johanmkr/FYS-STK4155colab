import numpy as np
import sys
import os
import pandas as pd


import plot as PLOT
PLOT.init('on')


from src.designMatrix import DesignMatrix
from src.Regression import LinearRegression
from src.Resampling import Bootstrap
from src.betaMatrix import betaParameter, betaCollection

seed = 7132
np.random.seed(seed)
# testSize = 1/5

Nx, Ny = 40, 40
# x = np.linspace(0, 1, Nx)
# y = np.linspace(0, 1, Ny)
x = np.random.uniform(0,1, Nx)
y = np.random.uniform(0,1, Ny)
x, y = np.meshgrid(x, y)


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


noise = lambda eta: eta*np.random.randn(Ny, Nx)



z = FrankeFunction(x, y) + noise(.5)
polydegs = range(1,20+1)

Trainings = []
Predictions = [] 

for n in polydegs:
    dM = DesignMatrix(n)
    dM.createX(x, y)
    reg = LinearRegression(z, dM)

    trainer, predictor = reg.split()
    #trainer.scale()
    #predictor.scale()
    beta = trainer.train() 
    trainer.computeModel()
    trainer.computeExpectationValues()

    predictor.setOptimalbeta(beta)
    predictor.computeModel()
    predictor.computeExpectationValues()

    Trainings.append(trainer)
    Predictions.append(predictor)


# PLOT.ptC_Hastie(Trainings, Predictions, pdf_name='Hastie', show=True)


Bootstrappings = []
for train, test in zip(Trainings, Predictions):
    BS = Bootstrap(train, test)
    BS(no_bootstraps=50)
    BS.bias_varianceDecomposition()
    Bootstrappings.append(BS)



PLOT.ptC_tradeoff(Bootstrappings, pdf_name='tradeoff', show=True)

# PLOT.ptC_bootstrap_hist(Bootstrappings, show=True)
