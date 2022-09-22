from src.utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


from src.parameterVector import ParameterVector




### Define some global varibles
## Valid 'codes' for settings:

manualModes = ['manual', 'MANUAL', 'Manual', 'own'] # own scripts
autoModes = ['auto', 'skl', 'SKL', 'sklearn', 'scikit'] # sklearn methods

olsMethods = ['ols', 'OLS', 'OrdinaryLinearRegression'] # OLS method
ridgeMethods = ['ridge', 'Ridge', 'RidgeRegression'] # Ridge method
lassoMethods = ['lasso', 'Lasso', 'LassoRegression'] # Lasso method


class LinearRegression:

    """
    Class for performing linear regression fitting.
    """    

    def __init__(self, data, design_matrix, method='ols', mode='manual'):
        """
        Constructor that holds information necessary to the regression (or something of the sort...). 

        Parameters
        ----------
        data : ndarray
            the (input) data points 
        design_matrix : designMatrix
            the design/feature matrix
        method : str, optional
            the method (in (global) olsMethods, ridgeMethods or lassoMethods) with which to perform the least squares fitting, by default 'ols'
        mode : str, optional
            the setting (in (global) manualModes or autoModes) for performing least squares fitting, by default 'manual'
        """        
        self._setMode(mode)
        self._setMethod(method)
            
        self.dM = design_matrix
        self.data = data

        self.polydeg = self.dM.n
        self.features = self.dM.p

        self.notTrainer = not isinstance(self, Training)
        self.notPredictor = not isinstance(self, Prediction)

        self.fit_intercept = not self.dM.scaled

    def _setMode(self, mode):
        """
        (intended local) Setting 'manual' (own code) or 'auto' (codes from scikit learn).

        Parameters
        ----------
        mode : str
            the setting (in (global) manualModes for own code or autoModes for skl)

        Raises
        ------
        ValueError
            the string "mode" is not in any of the globally defined lists - maybe append?
        """        
        mode = mode.strip().lower()
        if mode in manualModes:
            self.mode = 'manual'
        elif mode in autoModes:
            self.mode = 'auto'
        else:
            raise ValueError('Not a valid mode.')

    def _setMethod(self, method):
        """
        (intended local) Select method for regression.

        Parameters
        ----------
        method : str
            the method (in (global) olsMethods for Ordinary Least Squares, ridgeMethods for Ridge Regression or lassoMethods Lasso Regression)

        Raises
        ------
        ValueError
            the string "method" is not in any of the globally defined lists - maybe append?
        """        
        method = method.strip().lower()
        if method in olsMethods:
            self.method = 'ols'
        elif method in ridgeMethods:
            self.method = 'ridge'
        elif method in lassoMethods:
            self.method = 'lasso'
            self._setMode('auto')
        else:
            raise ValueError('Not a valid method.')      

    def changeMode(self, new_mode='other'):
        """
        Change setting in self-object.

        Parameters
        ----------
        new_mode : str, optional
            the (new) setting (in *Modes or 'other' for the opposite of what it currently is), by default 'other' 
        """        
        assert self.notPredictor
        old_mode = self.mode
        if new_mode == 'other':
            if old_mode in manualModes:
                new_mode = 'auto'
            elif old_mode in autoModes:
                new_mode = 'manual'
        self._setMode(new_mode)

    def changeMethod(self, new_method):
        """
        Change the method of the self-object.

        Parameters
        ----------
        new_method : str
            the (new) method to use (in *Methods)
        """        
        assert self.notPredictor and self.notTrainer
        self._setMethod(new_method)

    def __call__(self, lmbda=0):
        """
        Minimize cost function to find β-parameter.

        Parameters
        ----------
        lmbda : int, optional
            λ-parameter in Ridge/Lasso Regression, by default 0

        Returns
        -------
        ParameterVector
            the resulting beta from the least squares fitting
        """        
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
        
        self.lmbda = lmbda

        self.beta_current = ParameterVector(eval(f'self.{pre}{post}()'))

        return self.beta_current

    def setOptimalbeta(self, beta):
        """
        Set self-object's parameter β. OBS! Check the variance-thing.

        Parameters
        ----------
        beta : ParameterVector/ndarray/list
            the β-parameter
        """        
        self.beta = ParameterVector(beta)
        # self.beta.computeVariance(self) # Check this!

    def split(self, test_size=0.2):
        """
        Split the data and design matrix in training set and test set. OBS! Under construction.

        Parameters
        ----------
        test_size : float, optional
            the part of the data saved for the prediction model, by default 0.2

        Returns
        -------
        Training, Prediction
            new object for training, new object for testing
        """        

        assert self.notPredictor
        if self.notTrainer and test_size!=0:
            X_train, X_test, z_train, z_test = train_test_split(self.dM.X, self.data.ravel(), test_size=test_size, random_state=randomState)

            dM_train = self.dM.newObject(X_train)
            dM_test = self.dM.newObject(X_test)

            self.TRAINER = Training(self, z_train, dM_train)
            self.PREDICTOR = Prediction(self, z_test, dM_test)

        elif self.notTrainer and test_size == 0:
            self.TRAINER = Training(self, self.data, self.dM)
            #self.PREDICTOR = Prediction(self, z_test, dM_test)
            self.PREDICTOR = None

        else: 
            print('Not yet implemented for splitting training data')
            sys.exit()
        return self.TRAINER, self.PREDICTOR.


    def scale(self, type=StandardScaler()):
        """
        Scale the design matrix of test and train(???). CHECK!!!!
        
        Parameters
        ----------
        scaler : (not sure), optional
            scaler (from skl), by default StandardScaler()
        """ 
        if self.notPredictor and self.notTrainer:
            scaler = self.TRAINER.dM.buildScaler(type)
            self.TRAINER.dM.scale(scaler)
            self.PREDICTOR.dM.scale(scaler)
            # SCALE DATA! ?????
            self.fit_intercept = False
        else:
            print('Not ok yet')
            sys.exit()


    def _sklOLS(self):
        """
        (intended local) The Ordinary Least Squares algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed β parameter
        """        
        reg = linear_model.LinearRegression(fit_intercept=self.fit_intercept)
        X = self.dM.X
        z = self.data.ravel()
        reg.fit(X, z)
        beta = reg.coef_
        beta[0] = reg.intercept_
        return beta

    def _manOLS(self):
        """
        (intended local) The Ordinary Least Squares algorithm using own code.

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        HInv = self.dM.Hinv
        X = self.dM.X
        z = self.data.ravel()
        beta = HInv @ X.T @ z
        return beta
    
    def _sklRidge(self):
        """
        (intended local) The Ridge Regression algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        reg = linear_model.Ridge(fit_intercept=self.fit_intercept, alpha=self.lmbda)
        X = self.dM.X
        z = self.data.ravel()
        reg.fit(X, z)
        beta = reg.coef_
        beta[0] = reg.intercept_
        return beta

    def _manRidge(self):
        """
        (intended local) The Ridge Regression algorithm using own code

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        HInv = self.dM.Hinv
        X = self.dM.X
        z = self.data.ravel()
        I = np.eye(self.p)
        beta = (HInv  + 1/self.lmbda * I) @ X.T @ z
        return beta

    def _sklLasso(self):
        """
        (intended local) The Lasso Regression algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        reg = linear_model.Lasso(fit_intercept=self.fit_intercept, max_iter=int(1e6), alpha=self.lmbda)
        X = self.dM.X
        z = self.data.ravel()
        reg.fit(X, z)
        beta = reg.coef_
        beta[0] = reg.intercept_

    def computeModel(self):
        """
        Computes the model.

        Returns
        -------
        ndarray
            the model
        """

        self.model = self.dM * self.beta # see __mul__ and __rmul__ in respective classes
        return self.model.ravel()

    def computeExpectationValues(self):
        """
        Finds the mean squared error and the R2 score.
        """        
        z = self.data.ravel()
        ztilde = self.model.ravel()

        self.MSE = np.sum((z - ztilde)**2)/z.size
        self.R2 = 1 - np.sum((z - ztilde)**2) / np.sum((z - np.mean(z))**2)





class Training(LinearRegression):
    """
    Subclass as an extension specific for handling training data.
    """    

    def __init__(self, regressor, training_data, training_design_matrix):
        """
        Constructs a training set. 

        Parameters
        ----------
        regressor : LinearRegression
            the regressor for the original data for inheritance of information
        training_data : ndarray
            the z-points in the training set
        training_design_matrix : DesignMatrix
            the design matrix corresponding to the (x,y)-points that decides the z-points (i mangel på en bedre forklaring...)
        """        
        self.reg = regressor
        super().__init__(training_data, training_design_matrix, regressor.method, regressor.mode)

    def train(self, lmbda=0):
        """
        Alias for the motherclass's __call__-method.

        Parameters
        ----------
        lmbda : int/float, optional
            the tuning parameter for Ridge/Lasso Regression, by default 0

        Returns
        -------
        ParameterVector
            the computed β-parameter
        """        
        super().__call__(lmbda)
        self.setOptimalbeta(self.beta_current)
        return self.beta_current

    def randomShuffle(self, without_seed=True):
        """
        Resample data with replacement.

        Returns
        -------
        Training
            new object with reshuffeled data and corresponding design matrix
        """        
        z = self.data.ravel()
        X = self.dM.X
        if without_seed:
            np.random.seed()
        idx = np.random.randint(0, len(z), len(z))
        np.random.seed(ourSeed)
        zstar = z[idx]
        Xstar = X[idx]
        dMstar = self.dM.newObject(Xstar)
        newtrainer = Training(self, zstar, dMstar)
        return newtrainer



class Prediction(LinearRegression):
    """
    Subclass as an extension specific for handling test data.
    """
    def __init__(self, regressor, test_data, test_design_matrix):
        """
        Constructs a prediction set.

        Parameters
        ----------
        regressor : LinearRegression
            the regressor for the original data for inheritance of information
        test_data : ndarray
            the z-points in the prediction set
        test_design_matrix : DesignMatrix
            the design matrix corresponding to the (x,y)-points that decides the z-points (i mangel på en bedre forklaring her også, gitt...)
        """        
        self.reg = regressor
        super().__init__(test_data, test_design_matrix, regressor.method, regressor.mode)


