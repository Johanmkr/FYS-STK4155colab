from src.utils import *



class noneData:

    def __init__(self, data):
        """
        Superclass (not working superbly...) for targetVector and designMatrix.

        Parameters
        ----------
        data : ndarray
            the data array
        """
        self.data = data 

    def setScalingParams(self, mu, sigma):
        """
        Set the scaling parameters for standard scaling.

        Parameters
        ----------
        mu : float/ndarray
            mean value
        sigma : float/ndarray
            standard deviation
        """
        self.mu = mu
        self.sigma = sigma

    def __getitem__(self, index):
        """
        Use self-object as list-object.

        Parameters
        ----------
        index : int/slice
            the element(s) of the object

        Returns
        -------
        float/ndarray
            the data values of the given index/indices
        """
        return self.data[index]

    def scale(self):
        """
        Scale data (does not work!).
        """
        data = self.data.copy()
        data -= self.mu
        data /= self.sigma
        self.data = data

    def _scale(self, data):
        """
        For scaling data, to be used in subclass.

        Parameters
        ----------
        data : ndarray
            the 'data'-attribute of the object

        Returns
        -------
        ndarray
            scaled data
        """
        data -= self.mu
        data /= self.sigma
        return data

    def rescale(self):
        """
        Rescale data (does not work!). 
        """
        data = self.data.copy()
        
        data *= self.sigma
        data += self.mu
        self.data = data





class designMatrix(noneData):
    """
    Class for gathering properties of the design matrix of a 2D polynomial,
        z = f(x, y)  = c0 + c1*x + c2*y + c3*x^2 + c4*xy + c5*y^2 + ...,
    for a given polynomial degree.
    COMPATIBLE CLASSES:
        * parameterVector 
        * targetVector
        * modelVector
        * leastSquares (+ subclasses)
        * Bootstrap
    """

    def __init__(self, X):
        """
        Constructor for the 2D polynomial design matrix w/o intercept coloumn.

        Parameters
        ----------
        X : ndarray/pd.DataFrame/designMatrix
            the design matrix 

        Raises
        ------
        TypeError
            if the input is of wrong type
        """

        if isinstance(X, (pd.DataFrame, np.ndarray)):
            if isinstance(X, pd.DataFrame):
                self.Xpd = X
                self.X = X.to_numpy()
                self.npoints, self.nfeatures = np.shape(self.X) 
            
            else:
                self.X = X
                self.npoints, self.nfeatures = np.shape(self.X) 
                idx = [f"({X[i,0]:.3f}, {X[i,1]:.3f})" for i in range(np.shape(X)[0])]
                self.Xpd = pd.DataFrame(data=X, index=idx, columns=matrixColoumns[:self.nfeatures])
           
            self.polydeg = features2polydeg(self.nfeatures)

            if np.all(np.abs(np.mean(self.X, axis=1)) < 1e-10):
                self.scaled = True
            else:
                self.scaled = False

        elif isinstance(X, designMatrix):
            self.X = X.X
            self.Xpd = X.Xpd
            self.npoints, self.nfeatures, self.polydeg, self.scaled = X.npoints, X.nfeatures, X.polydeg, X.scaled

        else:
            raise TypeError("X must be ndarray, designMatrix or DataFrame.")

        super().__init__(self.X)
        self.X = self.data
        
    def getScalingParams(self, keepdims=True):
        """
        Retrieve the scaling parameters needed for standard scaling.

        Parameters
        ----------
        keepdims : bool, optional
            whether to keep the dimensions of the mean and standard deviation, by default True

        Returns
        -------
        ndarray, ndarray
            the mean per coloumn, the standard deviation per coloumn
        """
        assert not self.scaled
        mu = np.mean(self.X, axis=0, keepdims=keepdims)
        sigma = np.std(self.X, axis=0, keepdims=keepdims)
        self.setScalingParams(mu, sigma)
        return mu, sigma

    def scale(self):
        """
        Scale the data using standard scaling (works).
        """
        self.X = super()._scale(self.X)

    def getMatrix(self):
        """
        Get the actual matrix.
        (For when in doubt.)

        Returns
        -------
        ndarray
            the design matrix
        """        
        return self.X

    def __len__(self):
        """
        Get the number of features in the feature matrix.

        Returns
        -------
        int
            number of features
        """
        return np.shape(self.X)[1]

    def __str__(self):
        """
        Display X in a nice way using pandas. 

        Returns
        -------
        str
            terminal print
        """   
        l = 40; ind = ' '*5
        s = '\n' + '-'*l + '\n'
        s += f'design matrix X:\n'
        if self.scaled:
            s += '(scaled)\n'
        s += ind + f'polynomial degree  n = {self.polydeg:6.0f}\n'
        s += ind + f'number of features p = {self.nfeatures:6.0f} (+1)\n'
        s += ind + f'number of points   N = {self.npoints:6.0f}\n'
        s += '-'*l +'\n'
        s += self.Xpd.__str__() + '\n'
        s += '-'*l +'\n'
        return s

    def __mul__(self, other):
        """
        Special method for performing the matrix multiplication 
            z = Xβ,
        designed to work with instances of parameterVector.

        Parameters
        ----------
        other : ParameterVector
            the object that defines the β-parameter
        Returns
        -------
        ndarray
            matrix-vector product Xβ 
        """     
        return self.X @ other.beta




class targetVector(noneData):
    """
    Class for gathering properties of the z-data.
    COMPATIBLE CLASSES:
        * parameterVector
        * designMatrix 
        * modelVector
        * leastSquares (+ subclasses)
        * Bootstrap
    """

    def __init__(self, z) -> None:
        """
        Constructor for the z-data.

        Parameters
        ----------
        z : ndarray/targetVector
            the z-data

        Raises
        ------
        TypeError
            if the input is of wrong type
        """

        if isinstance(z, np.ndarray):
            self.z = z.ravel()
            self.npoints = z.size

            if np.abs(np.mean(self.z)) < 1e-10:
                self.scaled = True
            else:
                self.scaled = False
        
        elif isinstance(z, targetVector):
            self.z = z.z
            self.npoints, self.scaled = z.npoints, z.scaled

        else:
            raise ValueError("z must be ndarray or targetVector")

        super().__init__(self.z)
        self.z = self.data

    def __str__(self) -> str:
        """
        Simple terminal output.

        Returns
        -------
        str
            terminal print
        """
        s = '\n'
        for zi in self.z:
            s += f'{zi:8.4f}, '
        s.strip().strip(',')
        s += '\n'
        return s

    def __len__(self):
        """
        Get the number of data points.

        Returns
        -------
        int
            number of data points
        """
        return len(self.z)
    
    def scale(self):
        """
        Scale z-data using standard scaling.
        """
        self.z = super()._scale(self.z)

    def getScalingParams(self):
        """
        Retrieve the scaling parameters needed for standard scaling.

        Returns
        -------
        float, float
            the mean, the standard deviation
        """
        assert not self.scaled
        self.mu = np.mean(self.z)
        self.sigma = np.std(self.z)
        return self.mu, self.sigma

    def __sub__(self, other):
        if isinstance(other, targetVector):
            return self.z - other.z
        else: 
            return self.z - other


class modelVector(targetVector):
    def __init__(self, z) -> None:
        super().__init__(z)




class parameterVector:


    def __init__(self, beta) -> None:
      
        if isinstance(beta, (list, np.ndarray)):
            self.beta = np.asarray(beta)
            self.nfeatures = len(self.beta)
            self.polydeg = features2polydeg(self.nfeatures)
            
            self.idx = [f'β_{j}' for j in range(1, self.nfeatures+1)]
            self.idx_tex = [r'$\beta_{%i}$'%j for j in range(1, self.nfeatures+1)]

            self.stdv = None
            self.var = None
        
        elif isinstance(beta, parameterVector):
            self.beta = beta.beta
            self.nfeatures, self.polydeg, self.idx, self.idx_tex = beta.nfeatures, beta.polydeg, beta.idx, beta.idx_tex
            self.stdv, self.var = beta.stdv, beta.var
            

        else:
            raise TypeError("beta must be ndarray, list or parameterVector.")

    def getVector(self):
        """
        Yield the object as an ndarray. For when in doubt, so you can force the vector out.

        Returns
        -------
        ndarray
            entries equal entries of β
        """        
        return np.asarray(self.beta)

    def __str__(self):
        """
        Display β in a nice way using pandas. If variance is computed, the standard deviation is shown in the coloumn to the right.

        Returns
        -------
        str
            terminal print
        """       
        try:
            var = self.stdv**2
            sigma = self.stdv
            data = np.zeros((self.nfeatures, 2))
            data[:,0] = self.beta
            data[:,1] = sigma
            betapd = pd.DataFrame(data=data, index=self.idx, columns=['β', 'Δβ'])
        except TypeError:
            betapd = pd.DataFrame(data=self.beta, index=self.idx)
        l = 40; ind = ' '*5
        s = '\n' + '-'*l + '\n'
        s += f'β parameter:\n'
        s += ind + f'polynomial degree  d = {self.polydeg:6.0f}\n'
        s += ind + f'number of features p = {self.nfeatures:6.0f}\n'
        s += '-'*l +'\n'
        s += betapd.__str__() + '\n'
        s += '-'*l +'\n'
        return s

    def __getitem__(self, index):
        """
        Yield element of β.

        Parameters
        ----------
        i : int
            the entry number (index) of β

        Returns
        -------
        float/ndarray
            the element(s) of β in question
        """        
        return self.beta[index]

    def __len__(self):
        return len(self.beta)

    def __rmul__(self, other):
        """
        Special method for performing the matrix multiplication 
            z = Xβ,
        designed to work with instances of DesignMatrix.

        Parameters
        ----------
        other : DesignMatrix
            the object that defines the feature matrix we want

        Returns
        -------
        ndarray
            matrix-vector product Xβ 
        """        
        return other.X @ self.beta

    def setVariance(self, var):
        tol = 1e-14
        try: 
            self.var = np.where(var>tol, var, np.nan)
            self.stdv = self.var**(1/2)
        except TypeError:
            self.var = None
            self.stdv = None



    def group(self):
        self.pV_grouped = groupedVector(self)
        return self.pV_grouped

    def mean(self):
        return np.mean(self.beta)


class groupedVector(parameterVector):

    def __init__(self, beta) -> None:
        super().__init__(beta)

        polydegs = range(1, self.polydeg+1)
        beta = np.zeros(self.polydeg)
        i = 0
        if isinstance(self.var, np.ndarray):
            VAR = True
            var = np.zeros(self.polydeg)
        for d in range(1, self.polydeg+1):
            if d == self.polydeg:
                if VAR:
                    var[d-1] = np.mean(self.var[i:])
                beta[d-1] = np.mean(self.beta[i:])
            else:
                p = polydeg2features(d) - 1
                if VAR:
                    var[d-1] = np.mean(self.var[i:p])
                beta[d-1] = np.mean(self.beta[i:p])
                i = p

        self.beta = beta
        self.nfeatures = self.polydeg
        self.idx = [f'β^({d})' for d in polydegs]
        self.idx_tex = [r'$\beta^{(%i)}$'%d for d in polydegs]
        if VAR:
            self.setVariance(var)









class dataPrepper:
    """
    Very problem specific class for making x, y, z data ready for regression.
    Splitting and scaling has to happen before we can use the source files. 
    (Ideally, all should work for unscaled data as well, but when I tried to account for this,
    so any bugs arised, so I bailed. Maybe fix if time?)

    Will not spend time explaining this class. 
    Essentially, one can make an instance and call it, and it is all good.
    """

    def __init__(self, x, y, z, test_size=0.2) -> None:
        
        X = featureMatrix_2Dpolynomial(x, y)
        self.dM_org = designMatrix(X)
        self.tV_org = targetVector(z)

        self.original = {'design matrix':self.dM_org, 'target vector':self.tV_org}
        
        self.npoints = len(self.tV_org) # number of points
        self.idx = np.random.permutation(self.npoints) # shuffle indices
        self.test_size = test_size
        self.polydeg = self.dM_org.polydeg
        
        self.several = False

     
    def split(self, test_size=None):
        test_size = test_size or self.test_size
        s =  int(self.npoints*test_size)

        z = self.tV_org[self.idx]
        X = self.dM_org[self.idx]

        self.tV_train = targetVector(z[s:])
        self.dM_train = designMatrix(X[s:])
        self.tV_test = targetVector(z[:s])
        self.dM_test = designMatrix(X[:s])

    def scale(self, original=False):
        self.mu_z, self.sigma_z = self.tV_train.getScalingParams()
        self.mu_X, self.sigma_X = self.dM_train.getScalingParams()

        self.tV_test.setScalingParams(self.mu_z, self.sigma_z)
        self.dM_test.setScalingParams(self.mu_X, self.sigma_X)

        self.tV_org.setScalingParams(self.mu_z, self.sigma_z)
        self.dM_org.setScalingParams(self.mu_X, self.sigma_X)

        self.tV_train.scale()
        self.dM_train.scale()
        self.tV_test.scale()
        self.dM_test.scale()

        if original:
            self.tV_org.scale()
            self.dM_org.scale()

    def __call__(self):
        return self.tV_train, self.dM_train, self.tV_test, self.dM_test

    def getTrain(self, polydeg=None):
        if self.several and polydeg is not None:
            return self.tV_train, self.dM_train[polydeg]
        else:
            return self.tV_train, self.dM_train

    def getTest(self, polydeg=None):
        if self.several and polydeg is not None:
            return self.tV_test, self.dM_test[polydeg]
        else:
            return self.tV_test, self.dM_test

    def dump(self):
        return self.tV_org, self.dM_org
        

    def reduceOrderPolynomial(self, polydeg):
        if polydeg < self.polydeg:
            nfeatures = polydeg2features(polydeg)

            Xtrain = self.dM_train.getMatrix().copy()
            Xtrain = Xtrain[:,:nfeatures]
            dM_train = designMatrix(Xtrain)

            Xtest = self.dM_test.getMatrix().copy()
            Xtest = Xtest[:,:nfeatures]
            dM_test = designMatrix(Xtest)
        elif polydeg == self.polydeg:
            dM_train = self.dM_train
            dM_test = self.dM_test

        else:
            raise ValueError("Polynomial degree cannot be higher than the original one.")

        
        return dM_train, dM_test

    def genereteSeveralOrders(self, polydegs=None, save=True):
        polydegs = polydegs or range(1, self.polydeg+1)
        dMs_train = {}
        dMs_test = {}
        for d in polydegs:
            dM_train, dM_test = self.reduceOrderPolynomial(d)
            dMs_train[d] = dM_train
            dMs_test[d] = dM_test
        if save:
            dMs_train[self.polydeg] = self.dM_train
            dMs_test[self.polydeg] = self.dM_test
            self.dM_train = dMs_train
            self.dM_test = dMs_test
            self.several = True
        return dMs_train, dMs_test

    def __call__(self):
        self.prep()

    def prep(self, scale_all_data=False):
        self.split()
        self.scale(scale_all_data)
        self.genereteSeveralOrders()
