import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

from src.designMatrix import DesignMatrix
from src.Regression import LeastSquares
from src.betaMatrix import betaMatrix
from src.Resampling import Bootstrap
from plot import *

testSize = 1/5
Nx, Ny = 200, 200
maxPolydeg = 5
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)


dM = DesignMatrix(maxPolydeg)
dM.create_X(x, y)

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


noise = lambda eta: eta*np.random.randn(Ny, Nx)
z = FrankeFunction(x, y) + noise(.1)
# z = FrankeFunction(x, y)

# plot_franke(x, y, z)

LS = LeastSquares(z, dM)
LS.split()
LS("OLS")

Beta = betaMatrix(z, dM, LS.beta)
BS = Bootstrap(LS, dM, Beta)
BS.perform(z, 100) #should be z_train
print(BS.Beta)








# beta_test = betaMatrix(z, dM, LS.beta)
# # print(beta_test)
# beta_test.compVariance()
# print(beta_test)
# print(np.var(beta_test.beta))
# print(beta_test.variance)


# from IPython import embed; embed()
# polynomial_degrees = np.arange(1,2,1)
# models = {}

# for n in polynomial_degrees:
#     X = DesignMatrix(n)
#     X.create_X(x, y)
#     LS = LeastSquares(z, X)
#     LS.split()
#     LS._sklOLS()
#     LS.evaluate_fit()
#     models[f"{n}"] = LS


# train_MSE = []
# train_R2 = []
# test_MSE = []
# test_R2 = []
# error = []
# bias = []
# variance = []

# for n in polynomial_degrees:
#     train_MSE.append(models[str(n)].train_MSE)
#     test_MSE.append(models[str(n)].test_MSE)
#     train_R2.append(models[str(n)].train_R2)
#     test_R2.append(models[str(n)].test_R2)
#     error.append(models[str(n)].error)
#     bias.append(models[str(n)].bias)
#     variance.append(models[str(n)].variance)

# plt.plot(polynomial_degrees, train_MSE, label="train mse")
# plt.plot(polynomial_degrees, test_MSE, label="test mse")
# plt.legend()

# plt.show()

# plt.plot(polynomial_degrees, train_R2, label="train r2")
# plt.plot(polynomial_degrees, test_R2, label="test r2")
# plt.legend()

# plt.show()


# plt.plot(polynomial_degrees, test_MSE, label="error")
# plt.plot(polynomial_degrees, bias, label="bias")
# plt.plot(polynomial_degrees, variance, label="variance")
# # plt.ylim(0,1)

# plt.legend()

# plt.show()


