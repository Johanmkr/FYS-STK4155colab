import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, metrics


manualModes = ['manual', 'MANUAL', 'Manual'] # own scripts
autoModes = ['auto', 'skl', 'SKL', 'sklearn', 'scikit'] # sklearn methods

olsMethods = ['ols', 'OLS', 'OrdinaryLeastSquares'] # OLS method
ridgeMethods = ['ridge', 'Ridge', 'RidgeRegression'] # Ridge method
lassoMethods = ['lasso', 'Lasso', 'LassoRegression'] # Lasso method

class LeastSquares:

    def __init__(self, mode='manual'):
        mode = mode.strip().lower()
        if mode in manualModes:
            self.mode = 'manual'
        elif mode in autoModes:
            self.mode = 'auto'

    def changeMode(self, new_mode='other'):
        old_mode = self.mode
        if new_mode == 'other':
            if old_mode in manualModes:
                self.mode = 'auto'
            elif old_mode in autoModes:
                self.mode = 'manual'
        elif new_mode in manualModes:
            self.mode = 'manual'
        elif new_mode in autoModes:
            self.mode = 'auto'

    def __call__(self, method, lmbda=0):
        method = method.strip().lower()
        if self.mode == 'auto':
            pre = '_skl'
        elif self.mode == 'manual':
            pre = '_man'
        
        if method in olsMethods:
            post = f'OLS'
        elif method in ridgeMethods:
            post = f'Ridge'
        elif method in lassoMethods:
            post = f'Lasso'
            pre = '_skl' # OBS! Have no manual setting for this
        
        self.lmbda = lmbda
        eval(f'self.{pre}{post}()')


    def split(self, design_matrix, data, test_size=0.2, scaler=StandardScaler()):

        X_train, X_test, z_train, z_test = train_test_split(design_matrix.X, data.ravel(), test_size=test_size)

        self.scaled = False
        if scaler != 'none':
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            self.scaled = True

        dM_train = design_matrix.newObject(X_train)
        dM_test = design_matrix.newObject(X_test)
        self.train = {'design matrix': dM_train, 'data': z_train}
        self.test = {'design matrix': dM_test, 'data': z_test}


    def _sklOLS(self):
        reg = linear_model.LinearRegression(fit_intercept=not self.scaled)
        X = self.train['design matrix'].X
        z = self.train['data'].ravel()
        reg.fit(X, z)

        self.beta = reg.coef_
        self.beta[0] = reg.intercept_

    def _manOLS(self):
        Hinv = self.train['design matrix'].Hinv
        X = self.train['design matrix'].X
        z = self.train['data'].ravel()
        betaOLS = Hinv @ X.T @ z

        self.beta = betaOLS


    def _sklRidge(self):
        pass

    def _manRidge(self):
        pass

    def _sklLasso(self):
        pass
    






class OrdinaryLeastSquares:

    def __init__(self) -> None:
        pass