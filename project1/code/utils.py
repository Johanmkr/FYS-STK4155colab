from operator import index
import numpy as np
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def polydeg2features(polydeg):
        # polynomial degree to number of features
        n = polydeg
        p = int((n+1)*(n+2)/2)
        return p

def feature2polydeg(features):
        # number of features to polynomial degree
        p = features
        coeff = [1, 3, 2-2*p]
        ns = np.roots(coeff)
        n = int(np.max(ns))
        return n



class Data2D:

    def __init__(self, x, y, z):
        if len(np.shape(x)) == 1:
            self.x, self.y = np.meshgrid(x, y)
        else:
            self.x, self.y = x, y
        
        self.z = z

    def scale(self, scaler=StandardScaler()):
        self.z = scaler.fit_transform(self.z)

    

    def create_designMatrix(self, max_polydeg=20):
        self.max_polydeg = max_polydeg
        # create design matrix
        xx = self.x.ravel(); yy = self.y.ravel()
        self.X = np.ones((len(xx), polydeg2features(self.max_polydeg)))


        j = 1
        cols = ['x^0 y^0']
        for i in range(1, self.max_polydeg+1):
            for k in range(i+1):
                self.X[:,j] = (xx**(i-k))*(yy**k)
                cols.append(f'x^{(i-k):1.0f} y^{k:1.0f}')
                j+=1

        self.Xpd = pd.DataFrame(self.X, columns=cols)

    def prepare_model(self, polydeg, test_size=0.2):
        n = polydeg2features(polydeg)
        modeldata = PolyModel2D(self.X[:,:n], self.z)
        modeldata.split(test_size)
        return modeldata

class PolyModel2D:

    def __init__(self, X, z):
        self.X = X
        self.z = z
        self.data = z

        self.features = np.shape(self.X)[1]
        self.polydeg = feature2polydeg(self.features)

    def __str__(self):
        pass

    def split(self, test_size=0.2):
        self.Ny, self.Nx = np.shape(self.z)
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z.ravel(), test_size=test_size)
        self.train = Train(X_train, z_train)
        self.test = Test(X_test, z_test)

    def optimize(self, method='OLS'):
        self.method = method
        if self.method == 'OLS':     
            beta = OrdinaryLeastSquares(self.train.X, self.train.z)
        
        self.train.set_optimal_beta(beta)
        self.test.set_optimal_beta(beta)
        self.set_optimal_beta(beta)

    def compute(self, reshape=True):
        self.model = self.X @ self.beta
        if reshape:
            self.model = np.reshape(self.model, (self.Ny, self.Nx))


    def set_optimal_beta(self, beta):
        self.beta = beta
        self.betapd = pd.DataFrame(data=self.beta, index=[f'β_{j}' for j in range(len(beta))])

    def __call__(self):
        self.compute()
        
        self.train.compute(False)
        self.train.mean_squared_error()
        self.train.R2_score()

        self.test.compute(False)
        self.test.mean_squared_error()
        self.test.R2_score()
        self.test.Bias_squared()
        self.test.Variance()
        return self.model

    def mean_squared_error(self):
        #n = np.size(self.data.ravel())
        #self.MSE = np.sum((self.data.ravel()-self.model.ravel())**2)/n
        data = self.data.ravel()
        model = self.model.ravel()
        self.MSE = np.mean((data-model)**2)
        #self.MSE = np.sum((data-model)**2)/np.size(data)

    def R2_score(self):
        self.R2 = 1 - np.sum((self.data.ravel()-self.model.ravel())**2) / np.sum((self.data.ravel()-np.mean(self.data.ravel()))**2)

    def Bias_squared(self):
        data = self.data.ravel()
        model = self.model.ravel()
        #self.bias2 = np.mean((data - np.mean(model, axis=1, keepdims=True))**2)
        self.bias2 = np.mean((data - np.mean(model))**2)

    def Variance(self):
        data = self.data.ravel()
        model = self.model.ravel()
        self.var = np.var(model)


class Parts(PolyModel2D):
    def __init__(self, X, z):
        super().__init__(X, z)

    def split(self, test_size=0.2):
        pass

class Train(Parts):
    def __init__(self, X, z):
        super().__init__(X, z)
        self.tag = 'train'

    def __str__(self):
        s = 'Train data '
        return s

class Test(Parts):
    def __init__(self, X, z):
        super().__init__(X, z)
        self.tag = 'test'

    def __str__(self):
        s = 'Test data'
        return s



def OrdinaryLeastSquares(X, z):
    H = X.T @ X
    beta = np.linalg.pinv(H) @ X.T @ z.ravel()
    return beta


def RidgeRegression(X, z, lmbda):
    H = X.T @ X
    p = np.shape(X)[1]
    I = np.eye(p, p)
    beta = np.linalg.inv(X.T @ X+lmbda*I) @ X.T @ z.ravel()
    return beta



class RegressionMethod:

    def __init__(self, train):
        self.X = self.train.X
        self.z = self.train.data
    
    def beta(self):
        return self.beta


class _OrdinaryLeastSquares(RegressionMethod):

    def __init__(self, train):
        super().__init__(train)

    def beta(self):
        H = self.X.T @ self.X
        self.beta = np.linalg.pinv(H) @ self.X.T @ self.z.ravel()
    

class _RidgeRegression(RegressionMethod):

    def __init__(self, train):
        super().__init__(train)

    def beta(self, lmbda):
        H = self.X.T @ self.X
        p = self.train.features
        I = np.eye(p, p)
        self.beta = np.linalg.inv(H+lmbda*I) @ self.X.T @ self.z.ravel()



