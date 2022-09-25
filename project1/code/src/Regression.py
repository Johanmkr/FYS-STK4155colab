from cgi import test
from src.utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model



from src.parameterVector import ParameterVector
from src.targetVector import TargetVector
from src.designMatrix import DesignMatrix



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

    def __init__(self, target_vector, design_matrix, method='ols', mode='manual'):
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
            
        if isinstance(design_matrix, DesignMatrix):
            self.dM = design_matrix
        else:
            self.dM = DesignMatrix(design_matrix)
          
        if isinstance(target_vector, TargetVector):
            self.tV = target_vector
        else:
            self.tV = TargetVector(target_vector)
        
        self.data = self.tV.data

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

        self.pV = ParameterVector(eval(f'self.{pre}{post}()'), intercept=self.fit_intercept)

        return self.pV

    def setOptimalbeta(self, beta):
        """
        Set self-object's parameter β. OBS! Check the variance-thing.

        Parameters
        ----------
        beta : ParameterVector/ndarray/list
            the β-parameter
        """        
        self.pV = ParameterVector(beta, intercept=self.fit_intercept)
        # self.beta.computeVariance(self) # Check this!

    def split(self, test_size=0.2, scale=True):
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
        data = DataHandler(self.tV, self.dM)
        shuffled_data = data.shuffle()
        train_data, test_data = data.split(test_size)

        data = DataHandler(self.tV, self.dM)

        if scale:
            self.ref_data = train_data # save for later
            test_data = test_data.standardScaling(self.ref_data)
            train_data = train_data.standardScaling(self.ref_data)
            self.fit_intercept = False

        self.TRAINER = Training(self, train_data) 
        self.PREDICTOR = Prediction(self, test_data)
        self.TRAINER.scaled = not self.fit_intercept
        self.PREDICTOR.scaled = not self.fit_intercept
    
        return self.TRAINER, self.PREDICTOR


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
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        if self.fit_intercept:
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
        z = self.tV.z
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
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        if self.fit_intercept:
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
        z = self.tV.z
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
        z = self.tV.z
        reg.fit(X, z)
        beta = reg.coef_
        if self.fit_intercept:
            beta[0] = reg.intercept_

    def computeModel(self):
        """
        Computes the model.

        Returns
        -------
        ndarray
            the model
        """

        #unscale data?

        self.model = self.dM * self.pV # see __mul__ and __rmul__ in respective classes
        return self.model.ravel()

    def computeExpectationValues(self):
        """
        Finds the mean squared error and the R2 score.
        """        
        z = self.tV.z
        ztilde = self.model.ravel()

        self.MSE = np.sum((z - ztilde)**2)/z.size
        self.R2 = 1 - np.sum((z - ztilde)**2) / np.sum((z - np.mean(z))**2)

    

    def copy(self):
        LR = LinearRegression(self.tV, self.dM, self.method, self.mode)

        if hasattr(self, "pV"):
            LR.pV = self.pV
        if hasattr(self, "model"):
            LR.model = self.model
        if hasattr(self, "MSE"):
            LR.MSE = self.MSE
            LR.R2 = self.R2

        return LR





class Training(LinearRegression):
    """
    Subclass as an extension specific for handling training data.
    """    

    def __init__(self, regressor, train_target_vector, train_design_matrix=None):
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
        self.reg = regressor
        if isinstance(train_target_vector, DataHandler):
            tV = train_target_vector.tV
            dM = train_target_vector.dM
        else:
            tV = train_target_vector
            dM = train_design_matrix
        super().__init__(tV, dM, regressor.method, regressor.mode)

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
        data = DataHandler(self.tV, self.dM)
        shuffled_data = data.shuffle()

        newtrainer = Training(self, shuffled_data)
        return newtrainer

    def copy(self):

        T = Training(self.reg, self.tV, self.dM)

        if hasattr(self, "pV"):
            T.pV = self.pV
        if hasattr(self, "model"):
            T.model = self.model
        if hasattr(self, "MSE"):
            T.MSE = self.MSE
            T.R2 = self.R2

        return T




class Prediction(LinearRegression):
    """
    Subclass as an extension specific for handling test data.
    """
    def __init__(self, regressor, test_target_vector, test_design_matrix=None):
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
        self.reg = regressor
        if isinstance(test_target_vector, DataHandler):
            tV = test_target_vector.tV
            dM = test_target_vector.dM
        else:
            tV = test_target_vector
            dM = test_design_matrix
        super().__init__(tV, dM, regressor.method, regressor.mode)


    def predict(self):
        return super().computeModel()


    def copy(self):

        T = Prediction(self.reg, self.tV, self.dM)

        if hasattr(self, "pV"):
            T.pV = self.pV
        if hasattr(self, "model"):
            T.model = self.model
        if hasattr(self, "MSE"):
            T.MSE = self.MSE
            T.R2 = self.R2

        return T







class DataHandler:
    def __init__(self, target_vector, design_matrix):
        self.dM = design_matrix
        self.tV = target_vector

        self.z = self.tV.getVector()
        self.X = self.dM.getMatrix()

        #self.dM_org = self.dM.copy()
        #elf.tV_org = self.tV.copy()

    def shuffle(self):
        idx = np.random.randint(0, len(self.tV), len(self.tV))
        tV_ = self.tV.newObject(self.z[idx], is_scaled=self.tV.scaled)
        dM_ = self.dM.newObject(self.X[idx], is_scaled=self.dM.scaled, no_intercept=not self.dM.interceptColoumn)
        
        self.z = tV_.getVector()
        self.X = dM_.getMatrix()
        shuffled_data = DataHandler(tV_, dM_)

        self.tV = tV_
        self.dM = dM_
        
        return shuffled_data

    def standardScaling(self, reference_data=None):
        ref_data = reference_data or self
        zmean = np.mean(ref_data.z)
        zstd = np.std(ref_data.z)
        self.z_org = self.z.copy() # save original z
        self.z -= zmean
        self.z /= zstd

        Xmean = np.mean(ref_data.X, axis=0, keepdims=True) 
        Xstd = np.std(ref_data.X, axis=0, keepdims=True) 
        self.X_org = self.X.copy() # save original X  
        self.X -= Xmean
        self.X /= Xstd

        self.scaled = True

        tV_ = self.tV.newObject(self.z, is_scaled=True)
        dM_ = self.dM.newObject(self.X, is_scaled=True)

        scaled_data = DataHandler(tV_, dM_)

        self.tV = tV_
        self.dM = dM_

        return scaled_data

    def standardRescaling(self, reference_data=None):
        ref_data = reference_data

        zmean = np.mean(ref_data.z)
        zstd = np.std(ref_data.z)

        self.z *= zstd
        self.z += zmean

        Xmean = np.mean(ref_data.X, axis=0, keepdims=True) 
        Xstd = np.std(ref_data.X, axis=0, keepdims=True) 

        self.X *= Xstd
        self.X += Xmean


        # add intercept coloumn ? 
        X = np.ones((np.shape(self.X)[0]+1,np.shape(self.X)[1]))
        X[:,1:] = self.X
        self.X = X

        tV_ = self.tV.newObject(self.z, is_scaled=False)
        dM_ = self.dM.newObject(self.X, is_scaled=False)

        unscaled_data = DataHandler(tV_, dM_)

        self.tV = tV_
        self.dM = dM_

        return unscaled_data



    def split(self, test_size=0.2):

        s = int(test_size*len(self.tV))

        # train set
        tV_train = self.tV.newObject(self.z[s:])
        dM_train = self.dM.newObject(self.X[s:])
        train_data = DataHandler(tV_train, dM_train)

        # test set
        tV_test = self.tV.newObject(self.z[:s])
        dM_test = self.dM.newObject(self.X[:s])
        test_data = DataHandler(tV_test, dM_test)

        return train_data, test_data



