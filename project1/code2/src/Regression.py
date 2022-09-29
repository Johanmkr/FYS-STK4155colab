from src.utils import *

from sklearn import linear_model

from src.objects import parameterVector, targetVector, designMatrix

from copy import deepcopy


### Define some global varibles
## Valid 'codes' for settings:

manualModes = ['manual', 'MANUAL', 'Manual', 'own'] # own scripts
autoModes = ['auto', 'skl', 'SKL', 'sklearn', 'scikit'] # sklearn methods

olsMethods = ['ols', 'OLS', 'OrdinaryLinearRegression'] # OLS method
ridgeMethods = ['ridge', 'Ridge', 'RidgeRegression'] # Ridge method
lassoMethods = ['lasso', 'Lasso', 'LassoRegression'] # Lasso method


class linearRegression:

    """
    Class for performing linear regression fitting in 3d space.
    """    

    def __init__(self, train, test, method='ols', mode='manual'):
        #FIXME
        """
        Constructor that holds information necessary to the regression (or something of the sort...). 

        Parameters
        ----------
        target_vector : TargetVector
            the (input) data points 
        design_matrix : DesignMatrix
            the design/feature matrix
        method : str, optional
            the method (in (global) olsMethods, ridgeMethods or lassoMethods) with which to perform the least squares fitting, by default 'ols'
        mode : str, optional
            the setting (in (global) manualModes or autoModes) for performing least squares fitting, by default 'manual'
        """        
        self._setMode(mode)
        self._setMethod(method)

        self.trainer = train
        self.predictor = test
        self.npoints = self.trainer.npoints + self.predictor.npoints
        self.polydeg = self.trainer.polydeg
        
        self.trainer._setMode(mode)
        self.trainer._setMethod(method)
        self.predictor._setMode(mode)
        self.predictor._setMethod(method)

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
        #assert self.notPredictor
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
        #assert self.notPredictor and self.notTrainer
        self._setMethod(new_method)

    def fit(self, lmbda=0):
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
        #assert self.notPredictor

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
        
        if not isinstance(self, (Training, Prediction)):
            self.trainer.lmbda = lmbda
            beta = eval(f'self.trainer.{pre}{post}()')
            self.trainer.setOptimalbeta(beta)
            self.predictor.setOptimalbeta(beta)

        elif isinstance(self, Training):
            beta = eval(f'self.{pre}{post}()')
        else:
            raise TypeError("Cannot train on the test data.")

        self.setOptimalbeta(beta)

        return self.pV

    def setOptimalbeta(self, beta):
        """
        Set self-object's parameter β. 

        Parameters
        ----------
        beta : ParameterVector/ndarray/list
            the β-parameter
        """  
        
        self.pV = parameterVector(beta)


    def _sklOLS(self):
        """
        (intended local) The Ordinary Least Squares algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed β parameter
        """        
        reg = linear_model.LinearRegression()
        X = self.dM.X
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        return beta

    def _manOLS(self):
        """
        (intended local) The Ordinary Least Squares algorithm using own code.

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        
        X = self.dM.X
        XTX = X.T @ X
        z = self.tV.z
        beta = np.linalg.pinv(XTX) @ X.T @ z
        return beta
    
    def _sklRidge(self):
        """
        (intended local) The Ridge Regression algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        reg = linear_model.Ridge(alpha=self.lmbda)
        X = self.dM.X
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        #beta[0] = reg.intercept_
        return beta

    def _manRidge(self):
        """
        (intended local) The Ridge Regression algorithm using own code

        Returns
        -------
        ndarray
            the computed β parameter
        """   

        X = self.dM.X
        XTX = X.T @ X
        z = self.tV.z
        p = np.shape(X)[1]
        I = np.eye(p)
        beta = np.linalg.pinv(XTX+self.lmbda*I) @ X.T @ z
        return beta

    def _sklLasso(self):
        """
        (intended local) The Lasso Regression algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed β parameter
        """   
        reg = linear_model.Lasso(max_iter=int(1e6), alpha=self.lmbda)
        X = self.dM.X
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        #beta[0] = reg.intercept_

    def computeModel(self):
        """
        Computes the model.

        Returns
        -------
        ndarray
            the model
        """
        self.model = self.dM * self.pV  # see __mul__ and __rmul__ in respective classes
        #self.model = model.ravel()*zstd + zmean
        
        return self.model

    def computeExpectationValues(self):
        """
        Finds the mean squared error and the R2 score.
        """   
        self.mean_squared_error()
        self.R2_score()

    
    def mean_squared_error(self):
        self.MSE = np.sum((self.data - self.model)**2)/self.npoints
        return self.MSE

    def R2_score(self):
        self.R2 = 1 - np.sum((self.data - self.model)**2) / np.sum((self.data - np.mean(self.data))**2)
        return self.R2


    def subReg(self):
        # copy instance (temporary, for bootstrapping)
        # maybe delete

        trainer_star = self.trainer.randomShuffle()
        return linearRegression(trainer_star, self.predictor)










class Training(linearRegression):
    """
    Subclass as an extension specific for handling training data.
    """    

    def __init__(self, train_target_vector, train_design_matrix):
        """
        Constructs a training set. 

        Parameters
        ----------
        regressor : LinearRegression
            the regressor for the original data for inheritance of information
        training_target_vector : TargetVector
            the z-points in the training set
        training_design_matrix : DesignMatrix
            the design matrix corresponding to the (x,y)-points that decides the z-points (i mangel på en bedre forklaring...)
        """   


        self.tV = targetVector(train_target_vector)
        self.dM = designMatrix(train_design_matrix) 
        self.data = self.tV.z
        self.npoints = len(self.tV)
        self.polydeg = self.dM.polydeg

  
       
    def train(self, lmbda=0):
        """
        Alias for the motherclass's fit-method.

        Parameters
        ----------
        lmbda : int/float, optional
            the tuning parameter for Ridge/Lasso Regression, by default 0

        Returns
        -------
        ParameterVector
            the computed β-parameter
        """        
        super().fit(lmbda)
        self.setOptimalbeta(self.pV)
        return self.pV



    def randomShuffle(self, without_seed=True):
        """
        Resample data with replacement.

        Returns
        -------
        Training
            new object with reshuffeled data and corresponding design matrix
        """        
        idx = np.random.randint(0, self.npoints, self.npoints) # make sure this is with replacement

        z = self.tV[idx]
        X = self.dM[idx]
        tV = targetVector(z)
        dM = designMatrix(X)

        return Training(tV, dM)




class Prediction(linearRegression):
    """
    Subclass as an extension specific for handling test data.
    """
    def __init__(self, test_target_vector, test_design_matrix):
        """
        Constructs a prediction set.

        Parameters
        ----------
        regressor : LinearRegression
            the regressor for the original data for inheritance of information
        test_target_vector : TargetVector
            the z-points in the prediction set
        test_design_matrix : DesignMatrix
            the design matrix corresponding to the (x,y)-points that decides the z-points (i mangel på en bedre forklaring her også, gitt...)
        """        

  
    
        
        self.tV = targetVector(test_target_vector)
        self.dM = designMatrix(test_design_matrix)   
        self.data = self.tV.z 
        self.npoints = len(self.tV)
        self.polydeg = self.dM.polydeg
      
    def predict(self):
        super().computeModel()
        return self.model














