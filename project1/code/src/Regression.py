from src.utils import *

from sklearn import linear_model
from src.objects import parameterVector, targetVector, designMatrix, modelVector


## Valid 'codes' for settings:

manualModes = ['manual', 'MANUAL', 'Manual', 'own'] # own scripts
autoModes = ['auto', 'skl', 'SKL', 'sklearn', 'scikit'] # sklearn methods

olsMethods = ['ols', 'OLS', 'OrdinaryLinearRegression'] # OLS method
ridgeMethods = ['ridge', 'Ridge', 'RidgeRegression'] # Ridge method
lassoMethods = ['lasso', 'Lasso', 'LassoRegression'] # Lasso method

class linearRegression:

    """
    Class for performing linear regression fitting in 3D space.
    Currently assuming scaled training and validation data, as according to Z-score normalisation.
    """    

    def __init__(self, train, test, scheme='ols', mode='manual', shrinkage_parameter=0):
        """
        Constructor that holds information necessary to the regression (or something of the sort...). 

        Parameters
        ----------
        train : Training
            the SCALED training set 
        test : Prediction
            the SCALED prediction set
        method : str, optional
            the method (in (global) olsMethods, ridgeMethods or lassoMethods) with which to perform the fitting, by default 'ols'
        mode : str, optional
            the setting (in (global) manualModes or autoModes) for performing least squares fitting, by default 'manual'
        shinkage_parameter : float, optional
            value of shrinkage parameter in Ridge/Lasso regression, by default 0
        """ 

        self.__setMode(mode)
        self.__setMethod(scheme)

        for set in [train, test]:
            set.__setMode(mode)
            set.__setMethod(scheme)

        self.trainer = train
        self.predictor = test
        
        self.npoints = self.trainer.npoints + self.predictor.npoints
        self.polydeg = self.trainer.polydeg
        self.nfeatures = self.trainer.nfeatures

        self.lmbda = shrinkage_parameter

    def __setMode(self, mode):
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

    def __setMethod(self, scheme):
        """
        (intended local) Select method for regression.

        Parameters
        ----------
        scheme : str
            the method (in (global) olsMethods for Ordinary Least Squares, ridgeMethods for Ridge Regression or lassoMethods Lasso Regression)

        Raises
        ------
        ValueError
            the string "scheme" is not in any of the globally defined lists - maybe append?
        """        
        scheme = scheme.strip().lower()
        if scheme in olsMethods:
            self.scheme = 'ols'
        elif scheme in ridgeMethods:
            self.scheme = 'ridge'
        elif scheme in lassoMethods:
            self.scheme = 'lasso'
            self.__setMode('auto')
        else:
            raise ValueError('Not a valid scheme.')      

    def changeMode(self, new_mode='other'):
        """
        Change setting in self-object.

        Parameters
        ----------
        new_mode : str, optional
            the (new) setting (in *Modes or 'other' for the opposite of what it currently is), by default 'other' 
        """        
        
        old_mode = self.mode
        if new_mode == 'other':
            if old_mode in manualModes:
                new_mode = 'auto'
            elif old_mode in autoModes:
                new_mode = 'manual'
        self.__setMode(new_mode)

    def changeMethod(self, new_scheme):
        """
        Change the method of the self-object.

        Parameters
        ----------
        new_scheme : str
            the (new) method to use (in *Methods)
        """        
        #assert self.notPredictor and self.notTrainer
        self.__setMethod(new_scheme)

    def fit(self, shrinkage_parameter=None):
        """
        Minimise cost function to find ??-parameter.
        

        Parameters
        ----------
        lmbda : float, optional
            ??-parameter in Ridge/Lasso Regression, by default self.lmbda

        Returns
        -------
        parameterVector
            the resulting beta vector from the least squares fitting

        Raises
        ------
        TypeError
            if the user would be so crazy as to train on the validation data
        """

        self.lmbda = shrinkage_parameter or self.lmbda     

        if self.mode == 'auto':
            pre = '_skl'
        elif self.mode == 'manual':
            pre = '_man'
        # Why in the whole wide world does this not work with dunder???
        
        if self.scheme in olsMethods:
            post = f'OLS'
        elif self.scheme in ridgeMethods:
            post = f'Ridge'
        elif self.scheme in lassoMethods:
            post = f'Lasso'
    
        if not isinstance(self, (Training, Prediction)):
            self.trainer.setHyperParameter(self.lmbda)
            self.predictor.setHyperParameter(self.lmbda)
            beta = eval(f'self.trainer.{pre}{post}()')
            self.trainer.setOptimalbeta(beta)
            self.predictor.setOptimalbeta(parameterVector(beta))
            

        elif isinstance(self, Training):
            beta = eval(f'self.{pre}{post}()')
        else:
            raise TypeError("Cannot train on the test data.")

        self.setOptimalbeta(beta)

        return self.pV

    def setHyperParameter(self, lmbda):
        """
        Self-explanatory.

        Parameters
        ----------
        lmbda : float
            hyper parameter in Ridge/Lasso regression
        """
        self.lmbda = lmbda

    def setOptimalbeta(self, beta):
        """
        Set self-object's parameter ??. Compute its variance.

        Parameters
        ----------
        beta : ParameterVector/ndarray/list
            the ??-parameter
        """ 
        self.pV = parameterVector(beta)
        if not isinstance(self, Prediction):
            self.betaVariance()
  
    def _sklOLS(self):
        """
        (intended local) The Ordinary Least Squares algorithm using Scikit learn.

        Returns
        -------
        ndarray
            the computed ?? parameter
        """        
        reg = linear_model.LinearRegression(fit_intercept=False)
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
            the computed ?? parameter
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
            the computed ?? parameter
        """   
        reg = linear_model.Ridge(fit_intercept=False, alpha=self.lmbda)
        X = self.dM.X
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        return beta

    def _manRidge(self):
        """
        (intended local) The Ridge Regression algorithm using own code.

        Returns
        -------
        ndarray
            the computed ?? parameter
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
        Maximum number of iterations is set to 10^5. Feel free to increse if you have what in Norway is known as "fritidsproblemer".

        Returns
        -------
        ndarray
            the computed ?? parameter
        """   
        reg = linear_model.Lasso(fit_intercept=False, max_iter=int(1e5), alpha=self.lmbda)
        X = self.dM.X
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        return beta

    def computeModel(self):
        """
        Computes the model.

        Returns
        -------
        modelVector
            the model
        """
        self.model = self.dM * self.pV  # see __mul__ and __rmul__ in respective classes     
        self.mV = modelVector(self.model)
        return self.mV

    def computeExpectationValues(self):
        """
        Finds the mean squared error and the R2 score.
        """   
        self.mean_squared_error()
        self.R2_score()
   
    def mean_squared_error(self):
        """
        Calculate the mean squared error of the model.

        Returns
        -------
        float
            MSE-score (attribute 'MSE')
        """
        self.MSE = np.sum((self.data - self.model)**2)/self.npoints
        return self.MSE

    def R2_score(self):
        """
        Calculate the R^2-score of the model.

        Returns
        -------
        float
            R^2-score (attribute 'R2')
        """
        self.R2 = 1 - np.sum((self.data - self.model)**2) / np.sum((self.data - np.mean(self.data))**2)
        return self.R2

    def betaVariance(self):
        """
        Compute the variance of the beta parameter. Save it to the attribute 'pV'.

        Raises
        ------
        TypeError
            if you DARE use the test data for this
        """
        sigma2 = 1 # as data is scaled

        if not isinstance(self, (Training, Prediction)):
            X = self.trainer.dM.getMatrix()

        elif isinstance(self, Training):
            X = self.dM.getMatrix()

        else:
            raise TypeError("Cannot use test data for this.")
        
        XTX = X.T @ X

        if self.scheme == 'ols':
            var = sigma2 * np.linalg.pinv(XTX)
            var = np.diag(var)
        elif self.scheme == 'ridge':
            A = np.linalg.pinv(XTX + self.lmbda*np.eye(self.nfeatures))
            var = sigma2 * A @ XTX @ A.T
            var = np.diag(var)
        else:
            var = None

        self.pV.setVariance(var)

    def __str__(self):
        """
        Print the polynomial degree assumed in this regression
        and the hyper parameter if relevant. 

        Returns
        -------
        rawstring
            plt-friendly string
        """
        s = r'$d = %i$'%self.polydeg
        if self.scheme != 'ols':
            s += '\n'
            s += r'$\lambda = %.2e$'%self.lmbda
        return s



class Training(linearRegression):
    """
    Subclass as an extension specific for handling training data.
    """    

    def __init__(self, train_target_vector, train_design_matrix):
        """
        Construct a training set. 

        Parameters
        ----------
        training_target_vector : targetVector
            the z-points in the SCALED training set
        training_design_matrix : designMatrix
            the SCALED design matrix corresponding to the (x,y)-points that decides the z-points (i mangel p?? en bedre forklaring...)
        """   

        self.tV = targetVector(train_target_vector)
        self.dM = designMatrix(train_design_matrix) 
        self.data = self.tV.z
        self.npoints = len(self.tV)
        self.polydeg = self.dM.polydeg
        self.nfeatures = self.dM.nfeatures
       
    def train(self, lmbda=None):
        """
        Alias for the motherclass's fit-method.

        Parameters
        ----------
        lmbda : float, optional
            the tuning parameter for Ridge/Lasso Regression, by default self.lmbda

        Returns
        -------
        parameterVector
            the computed ??-parameter
        """       
        lmbda = lmbda or self.lmbda 
        super().fit(lmbda)
        self.setOptimalbeta(self.pV)
        return self.pV

    def randomShuffle(self):
        """
        Draw data randomly with replacement.

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
        Construct a prediction set.

        Parameters
        ----------
        test_target_vector : targetVector
            the z-points in the SCALED prediction set
        test__design_matrix : designMatrix
            the SCALED design matrix corresponding to the (x,y)-points that decides the z-points (i mangel p?? en bedre forklaring...)
        """       
        self.tV = targetVector(test_target_vector)
        self.dM = designMatrix(test_design_matrix)   
        self.data = self.tV.z 
        self.npoints = len(self.tV)
        self.polydeg = self.dM.polydeg
        self.nfeatures = self.dM.nfeatures
      
    def predict(self):
        """
        Alias for motherclass's 'computeModel()'.

        Returns
        -------
        modelVector
            the model
        """
        super().computeModel()
        return self.model














