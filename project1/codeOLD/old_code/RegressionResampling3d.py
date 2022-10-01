import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


"""
1st draft for pipeline

Bare rør, her trengs det nye ideer
"""



class DataModel:

    def __init__(self, z, X, tag='data'):
        self.z = z
        self.X = X # design matrix
        self.tag = tag

        self.H = self.X.T @ self.X # Hessian

    def split(self, test_size=0.2):
        # Split into train and test
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z.ravel(), test_size=test_size)
        self.train = DataModel(z_train, X_train, 'train')
        self.test = DataModel(z_test, X_test, 'test')

        return self.train, self.test

    def __call__(self, beta):
        self.beta = beta
        self.ztilde = self.X @ self.beta
        self.MSE = self.mean_square_error()
        self.R2 = self.R2_score()

    def mean_square_error(self):
        n = np.size(self.z)
        return np.sum((self.z-self.ztilde)**2)/n

    def R2_score(self):
        return 1 - np.sum((self.z - self.ztilde) ** 2) / np.sum((self.z - np.mean(self.z)) ** 2)


class Regression(DataModel):

    def __init__(self, x, y, z, polynomial_degree):
        self.x, self.y = np.meshgrid(x,y)

        n = polynomial_degree
        xx = np.ravel(self.x); yy = np.ravel(self.y)

        N = len(xx)
        l = int((n+1)*(n+2)/2)
        X = np.ones((N,l))
        print(np.shape(X))

        for i in range(1, n+1):
            q = int((i)+(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (xx**(i-k))*(yy**k)


        super().__init__(z, X)


    def perform_regression(self, method='OLS'):
        super().split()

        if method == 'OLS':
            self.beta = np.linalg.pinv(self.train.H) @ self.train.X.T @ self.train.z.ravel()

        self.train.__call__(self.beta)
        self.test.__call__(self.beta)
        self.__call__(self.beta)










class Model:
    def __init__(self, method):
        pass

class System: #diverse...

    def __init__(self, x, y):
        self.x, self.y = np.meshgrid(x,y)

    def design_designMatrix(self, polynomial_degree):
        n = polynomial_degree
        xx = np.ravel(self.x); yy = np.ravel(self.y)

        N = len(xx)
        l = int((n+1)*(n+2)/2)
        X = np.ones((N,l))
        print(np.shape(X))

        for i in range(1, n+1):
            q = int((i)+(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (xx**(i-k))*(yy**k)

        return X



class Regression:

    def __init__(self, data, method='OLS'):
        self.org_data = data
        self.train_data, self.test_data = self.org_data.split()


        if method == 'OLS':
            self.beta = np.linalg.pinv(self.train_data.H) @ self.train_data.X.T @ self.train_data.z.ravel()


    def __call__(self):
        self.ztilde = self.train_data.model(self.beta)
        self.zpredict = self.test_data.model(self.beta)

    def test_MSE(self):
        return self.test_data.mean_square_error(self.zpredict)

    def train_MSE(self):
        return self.train_data.mean_square_error(self.ztilde)






"""
Prøver igjen

"""



class Regression:

    def __init__(self):
        pass



class idk:
    def __init__(self, data, model):
        self.data, self.model = data, model
        self.MSE = self.mean_square_error()
        self.R2 = self.R2_score()

    def mean_square_error(self):
        n = np.size(self.data)
        return np.sum((self.data-self.model)**2)/n

    def R2_score(self):
        return 1 - np.sum((self.data - self.model) ** 2) / np.sum((self.data - np.mean(self.data)) ** 2)



class Train(idk):
    def __init__(self, z, X):
        self.data = z
        self.model = lambda beta: self.X @ beta
        super().__init__(self.data, self.model)


class Test(idk):
    def __init__(self):
        pass

















'''

'''
