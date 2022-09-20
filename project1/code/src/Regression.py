from hashlib import new
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, metrics
from src.betaMatrix import betaMatrix

from src.betaMatrix import betaParameter



manualModes = ['manual', 'MANUAL', 'Manual', 'own'] # own scripts
autoModes = ['auto', 'skl', 'SKL', 'sklearn', 'scikit'] # sklearn methods

olsMethods = ['ols', 'OLS', 'OrdinaryLeastSquares'] # OLS method
ridgeMethods = ['ridge', 'Ridge', 'RidgeRegression'] # Ridge method
lassoMethods = ['lasso', 'Lasso', 'LassoRegression'] # Lasso method

class LeastSquares:

    def __init__(self, data, designMatrix, method='ols', mode='manual'):
        self._setMode(mode)
        self._setMethod(method)
            
        self.dM = designMatrix
        self.data = data

        self.notTrainer = not isinstance(self, Training)
        self.notPredictor = not isinstance(self, Prediction)
        #self.notSubclass = not issubclass(self, LeastSquares)

    def _setMode(self, mode):
        mode = mode.strip().lower()
        if mode in manualModes:
            self.mode = 'manual'
        elif mode in autoModes:
            self.mode = 'auto'
        else:
            raise ValueError('Not a valid mode.')

    def _setMethod(self, method):
        method = method.strip().lower()
        if method in olsMethods:
            self.method = 'ols'
        elif method in ridgeMethods:
            self.method = 'ridge'
        elif method in lassoMethods:
            self.method = 'lasso'
        else:
            raise ValueError('Not a valid method.')
        

    def changeMode(self, new_mode='other'):
        assert self.notPredictor
        old_mode = self.mode
        if new_mode == 'other':
            if old_mode in manualModes:
                self.mode = 'auto'
            elif old_mode in autoModes:
                self.mode = 'manual'
        else:
            self._setMode(new_mode)

    def changeMethod(self, new_method):
        assert self.notPredictor and self.notTrainer
        self._setMethod(new_method)


    def __call__(self, lmbda=0):
        assert self.notPredictor

        if self.mode == 'auto':
            pre = '_skl'
        elif self.mode == 'manual':
            pre = '_man'
        
        if self.method in olsMethods:
            post = f'OLS'
        elif self.method in ridgeMethods:
            post = f'Ridge'
        elif self.method in lassoMethods:
            post = f'Lasso'
            pre = '_skl' # OBS! Have no manual setting for this
        
        self.lmbda = lmbda

        eval(f'self.{pre}{post}()')

        return self.beta_current

    def setOptimalbeta(self, beta):
        self.beta = beta

    def split(self, test_size=0.2):
        assert self.notPredictor
        if self.notTrainer and test_size!=0:
            X_train, X_test, z_train, z_test = train_test_split(self.dM.X, self.data.ravel(), test_size=test_size)

            dM_train = self.dM.newObject(X_train)
            dM_test = self.dM.newObject(X_test)

            self.TRAINER = Training(self, z_train, dM_train)
            self.PREDICTOR = Prediction(self, z_test, dM_test)

        elif self.notTrainer and test_size == 0:
            self.TRAINER = Training(self, self.data, self.dM)
            #self.PREDICTOR = Prediction(self, z_test, dM_test)

        else: 
            print('Not yet implemented for splitting training data')
            sys.exit()
        return self.TRAINER, self.PREDICTOR

    def scale(self, scaler=StandardScaler()):
        self.dM.scale(scaler=scaler)


    def _sklOLS(self):
        reg = linear_model.LinearRegression(fit_intercept=not self.dM.scaled)
        X = self.dM.X
        z = self.data.ravel()
        reg.fit(X, z)

        beta = reg.coef_
        beta[0] = reg.intercept_
        self.beta_current = betaParameter(beta) 


    def _manOLS(self):
        Hinv = self.dM.Hinv
        X = self.dM.X
        z = self.data.ravel()
        beta = Hinv @ X.T @ z

        self.beta_current = betaParameter(beta) 


    def _sklRidge(self):
        pass

    def _manRidge(self):
        pass

    def _sklLasso(self):
        pass


    def fit(self):
        self.model = self.dM.X @ self.beta
        return self.model.ravel()

    def computeExpectationValues(self):
        z = self.data.ravel()
        X = self.dM.X
        ztilde = self.model.ravel()

        self.MSE = np.sum((z - ztilde)**2)/len(z)
        self.R2 = 1 - np.sum((z - ztilde)**2) / np.sum((z - np.mean(z))**2)


        # Only for predicted model??
        self.bias2 = np.mean((z - ztilde))
        self.var = np.var(ztilde)

        # ---




''' def evaluate_fit(self):
        z_train = self.train['data'].ravel()
        X_train = self.train['design matrix'].X
        self.train_MSE = np.sum((z_train - X_train@self.beta)**2)/len(z_train)
        self.train_R2 = 1 - np.sum((z_train - X_train@self.beta)**2) / np.sum((z_train - np.mean(z_train))**2)

        z_test = self.test['data'].ravel()
        X_test = self.test['design matrix'].X
        self.test_MSE = np.sum((z_test - X_test@self.beta)**2)/len(z_test)
        self.test_R2 = 1 - np.sum((z_test - X_test@self.beta)**2) / np.sum((z_test - np.mean(z_test))**2)

        z_pred = X_test @ self.beta 
        # z_pred = z_pred.ravel()
        # z_test = z_test.ravel()
        self.error = np.mean((z_test - z_pred)**2)
        self.bias = np.mean((z_test - z_pred))
        self.variance = np.var(z_pred)
        # from IPython import embed; embed()'''


class Prediction(LeastSquares):
    def __init__(self, regressor, test_data, test_design_matrix):
        self.reg = regressor
        super().__init__(test_data, test_design_matrix, regressor.method, regressor.mode)

    def bias_variance_tradeoff(self):
        pass



class Training(LeastSquares):

    def __init__(self, regressor, training_data, training_design_matrix):
        self.reg = regressor
        super().__init__(training_data, training_design_matrix, regressor.method, regressor.mode)


    def train(self, lmbda=0):
        super().__call__(lmbda)
        return self.beta_current

    def randomShuffle(self):
        z = self.data.ravel()
        X = self.dM.X
        idx = np.random.randint(0, len(z), len(z))
        zstar = z[idx]
        Xstar = X[idx]
        dMstar = self.dM.newObject(Xstar)

        newtrainer = Training(self, zstar, dMstar)
        return newtrainer
