import numpy as np
import sys
import os
import pandas as pd


print('\nImporting...\n')
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.designMatrix import DesignMatrix
from src.Regression import LeastSquares
from src.Resampling import Bootstrap

from src.betaMatrix import betaMatrix

testSize = 1/5

Nx, Ny = 60, 60
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)


print('\ndesign matrix\n')
dM = DesignMatrix(5)

print('Creating X\n')
dM.createX(x, y)


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


noise = lambda eta: eta*np.random.randn(Ny, Nx)
z = FrankeFunction(x, y) + noise(0.1)

print('\nregression...\n')
reg = LeastSquares(z, dM, mode='skl')

reg.scale()

trainer, predictor = reg.split()
trainer.train()



BS = Bootstrap(trainer, predictor)
BS(10)
#print(BS.betas)
#print(BS.betas2)

BS.bias_varianceDecomposition()




sys.exit()


# DIVERSE:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



from sklearn.model_selection import train_test_split


X_train, X_test, z_train, z_test = train_test_split(dM.X, z.ravel(), test_size=testSize)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

dM_train = dM.newObject(X_train)
dM_test = dM.newObject(X_test)

def OrdinaryLeastSquares(designMatrix, z_data):
    X = designMatrix.X
    Hinv = designMatrix.Hinv
    beta = Hinv @ X.T @ z_data.ravel()
    return beta

beta = OrdinaryLeastSquares(dM_train, z_train)
betapd = pd.DataFrame(data=beta, index=[f'β_{j}' for j in range(len(beta))])




def mean_squared_error(data, model):
    MSE = np.sum((data.ravel()-model.ravel())**2)/np.size(data.ravel())
    return MSE

def bias_squared(data, model):
    #data = data.ravel()
    #model = model.ravel()
    #bias2 = np.dot((data - np.mean(model)), (data - np.mean(model)))
    bias2 = mean_squared_error(data, np.mean(model))
    return bias2

def variance(model):
    var = np.var(model)
    return var


import matplotlib.pyplot as plt
fig, ax = plt.subplots()



polydegs = [1,2,3,4,5,6]
beta_list = []
MSE_list = []
bias2_list = []
var_list = []

for n in polydegs:
    dM = DesignMatrix(n)
    dM.create_X(x, y)
    X_train, X_test, z_train, z_test = train_test_split(dM.X, z.ravel(), test_size=testSize)

    dM_train = dM.newObject(X_train)
    dM_test = dM.newObject(X_test)
    beta = OrdinaryLeastSquares(dM_train, z_train)

    ztilde = dM_train.X @ beta
    zpredict = dM_test.X @ beta

    MSE = mean_squared_error(zpredict, z_test)
    bias2 = bias_squared(zpredict, z_test)
    var = variance(zpredict)

    beta_list.append(beta)
    MSE_list.append(MSE)
    bias2_list.append(bias2)
    var_list.append(var)
    print(f'\nn = {n}')
    print('     MSE: ', MSE)
    print('  bias^2: ', bias2)
    print('variance: ', var)



