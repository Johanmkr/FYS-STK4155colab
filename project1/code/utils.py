from ast import Pass
import numpy as np
import sys
import pandas as pd

from sklearn.model_selection import train_test_split



class SomeClass:

    def __init__(self, x, y, z):
        if len(np.shape(x)) == 1:
            self.x, self.y = np.meshgrid(x, y)
        else:
            self.x, self.y = x, y
        
        self.z = self.z

    def scale(self):
        # scale or center data
        pass

    def polydeg2features(self, polydeg):
        # polynomial degree to number of features
        n = polydeg
        p = int((n+1)*(n+2)/2)
        return p

    def create_designMatrix(self, max_polydeg=20, goto_polydeg=5):
        # create design matrix
        xx = self.x.ravel(); yy = self.y.ravel()
        self.X = np.ones((len(xx), self.polydeg2features(max_polydeg)))


        j = 1
        cols = [r'$x^0 y^0$']
        for i in range(1, max_polydeg+1):
            for k in range(i+1):
                self.X[:,j] = (xx**(i-k))*(yy**k)
                cols.append(r'$x^{%i} y^{%i}$'%((i-k), k))
                j+=1

        p = self.polydeg2features(goto_polydeg)
        '''Xstar = self.X[:, :p]
        cols = cols[:p]
        Xpd = pd.DataFrame(X, columns=cols)'''

    def change_gotoPolydeg(self, goto_polydeg):
        # change go-to /default polynomial degree

        pass

    def split(self, test_size=0.2):
        # maybe different (sub)class
        X_train, X_test, z_train, z_test = train_test_split(self.X, self.z.ravel(), test_size=test_size)
        self.train = Train(X_train, z_train)
        self.test = Test(X_test, z_test)
        pass

    def set_optimal_beta(self, beta):
        self.beta = beta
    

    def __call__(self):
        self.ztilde = Train()
        self.zpredict = Test()



class Splitting:
    def __init__(self, X, z):
        self.X = X
        self.data = z
    
    def __call__(self, beta):
        self.model = self.X @ beta
        self.mean_square_error()
        self.R2_score()
        return self.model

    def mean_square_error(self):
        n = np.size(self.data)
        self.MSE = np.sum((self.data-self.model)**2)/n

    def R2_score(self):
        self.R2 = 1 - np.sum((self.data-self.model)**2) / np.sum((self.data-np.mean(self.data))**2)


        
class Train(Splitting):
    def __init__(self, X, z):
        super().__init__(X, z)
        pass

  

class Test(Splitting):
    def __init__(self, X, z):
        super().__init__(X, z)
        pass


class PolynomialRegression:

    def __init__(self, something, polynomial_degree):
        pass

    def __call__(self):
        self.advance()
        self.something.set_optimal_beta(self.beta)


    def advance(self):
        pass
        # NotImplementedError?



class OrdinaryLeastSquares(PolynomialRegression):
    def __init__(self, data):
        super().__init__()
        pass

    def advance(self):
        H = self.train.X.T @ self.train.X
        self.beta = np.linalg.pinv(H) @ X_train.T @ z_train.ravel()
        pass


class Ridge(PolynomialRegression):
    def __init__(self):
        super().__init__()
        pass

    def advance(self, lmbda):
        pass