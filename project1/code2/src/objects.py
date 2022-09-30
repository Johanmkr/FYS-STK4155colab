
from copy import deepcopy
from multiprocessing.sharedctypes import Value
from src.utils import *





class dataType:

    def __init__(self, data) -> None:
        self.data = data #ndarray


    def setScalingParams(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __getitem__(self, index):
        return self.data[index]

    def scale(self):
        data = self.data.copy()
        data -= self.mu
        data /= self.sigma
        self.data = data

    def _scale(self, data):
        data -= self.mu
        data /= self.sigma
        return data

    def rescale(self):
        data = self.data.copy()
        
        data *= self.sigma
        data += self.mu
        self.data = data





class designMatrix(dataType):
    #FIXME
    #TODO
    """
    Class for gathering properties of the design matrix of a 2D polynomial,
        z = f(x, y)  = c0 + c1*x + c2*y + c3*x^2 + c4*xy + c5*y^2 + ...,
    for a given polynomial degree.
    COMPATIBLE CLASSES:
        * ParameterVector 
        * LeastSquares (+ subclasses)
        * Bootstrap
    """

    def __init__(self, X) -> None:
        """
        Constructor for the 2D polynomial design matrix.
        Parameters
        ----------
        polydeg : int
            2D polynomial degree
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
            raise ValueError("X must be ndarray, designMatrix or DataFrame.")

        super().__init__(self.X)
        self.X = self.data
        
    
    
    def getScalingParams(self, keepdims=True):
        assert not self.scaled
        mu = np.mean(self.X, axis=0, keepdims=keepdims)
        sigma = np.std(self.X, axis=0, keepdims=keepdims)
        self.setScalingParams(mu, sigma)
        return mu, sigma

    def scale(self):
        self.X = super()._scale(self.X)

 

    def getMatrix(self):
        """
        Get the actual matrix.
        Returns
        -------
        ndarray
            the design matrix of self.p features
        """        
        return self.X

    def __len__(self):
        return np.shape(self.X)[1] # self.nfeatures?


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
        designed to work with instances of ParameterVector.
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




class targetVector(dataType):

    def __init__(self, z) -> None:

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
        s = '\n'
        for zi in self.z:
            s += f'{zi:8.4f}, '
        s.strip().strip(',')
        s += '\n'
        return s

    def __len__(self):
        return len(self.z)
    
    def scale(self):
        self.z = super()._scale(self.z)


    def getScalingParams(self):
        assert not self.scaled
        self.mu = np.mean(self.z)
        self.sigma = np.std(self.z)
        return self.mu, self.sigma

    def __sub__(self, other):
        if isinstance(other, targetVector):
            return self.z - other.z
        else: 
            return self.z - other




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
            raise Value("beta must be ndarray, list or parameterVector.")

        

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
        
        if self.stdv != None:
            sigma = self.stdv
            data = np.zeros((self.p, 2))
            data[:,0] = self.beta
            data[:,1] = sigma
            betapd = pd.DataFrame(data=data, index=self.idx, columns=['β', 'Δβ'])
        else:
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
        self.var = var
        self.stdv = self.var**(1/2)



    def group(self):
        # FIXME

        polydegs = range(1, self.deg+1)
        beta_amp = np.zeros(self.deg)
        i = 0
        for d in range(1, self.deg):
            p = polydeg2features(d) - 1
            beta_amp[d-1] = np.mean(self.beta[i:p])
            i = p
        beta_amp[d] = np.mean(self.beta[i:])
            
        self.amp_betapd = pd.DataFrame(beta_amp, index=[f'β^({d})' for d in polydegs], columns=['β'])

        return self.amp_betapd


    def mean(self):
        return np.mean(self.beta)



        




class dataPrepper:

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

    def scale(self):
        self.mu_z, self.sigma_z = self.tV_train.getScalingParams()
        self.mu_X, self.sigma_X = self.dM_train.getScalingParams()

        self.tV_test.setScalingParams(self.mu_z, self.sigma_z)
        self.dM_test.setScalingParams(self.mu_X, self.sigma_X)

        self.tV_train.scale()
        self.dM_train.scale()
        self.tV_test.scale()
        self.dM_test.scale()

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
