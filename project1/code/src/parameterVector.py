from src.utils import *


class ParameterVector:
    """
    Class for gathering properties of the β-parameter, as fitting a least squares regression for some 2D polynomial basis.

    COMPATIBLE CLASSES:
        * DesignMatrix 
        * LeastSquares (+ subclasses)
        * Bootstrap
    """

    def __init__(self, beta, intercept=True):
        """
        Constructur for the β-parameter.

        Parameters
        ----------
        beta : list/ndarray/ParameterVector
            the (optimized) β-parameter
        """        
        if isinstance(beta, (list, np.ndarray)):
            self.beta = np.asarray(beta)
        elif isinstance(beta, ParameterVector):
            self.beta = beta.getVector()
        self.p = len(self.beta)
        self.intercept = intercept
        if self.intercept:
            self.deg = features2polydeg(self.p)
        else:
            self.deg = features2polydeg(self.p+1)

        self.n = self.deg

        self.stdv = None # standard deviation, to be calculated

        if self.intercept:
            self.idx = [f'β_{j}' for j in range(len(self.beta))]
            self.idx_tex = [r'$\beta_{%i}$'%j for j in range(len(self.beta))]
        else:
            self.idx = [f'β_{j}' for j in range(1, len(self.beta)+1)]
            self.idx_tex = [r'$\beta_{%i}$'%j for j in range(1, len(self.beta)+1)]
        

    def getVector(self):
        """
        Yield the object as an ndarray. For when in doubt, so you can force the vector out.

        Returns
        -------
        ndarray
            entries equal entries of β
        """        
        return np.asarray(self.beta)

    def computeVariance(self, regressor):
        """
        Find the variance in the β-parameter. OBS! Check this!!!!!

        Parameters
        ----------
        regressor : LeastSquares
            the regressor (trainer, predictor or the combiniation) that contains the data and the design matrix
        """        
        z = regressor.data.ravel()
        HInv = regressor.dM.Hinv #DUMMY SCALING SHOULD PROBABLY DO THIS DIFFERENTLY #FIXME

        HInv = np.linalg.pinv(regressor.dM.X.T @ regressor.dM.X)
        # self.var = np.diag(np.var(z) * HInv)[1:]
        # self.var = np.diag(HInv)
        print(f"Var(z): {np.var(z)}")
        self.var = np.diag(np.var(z) * HInv)

        # z_tilde = regressor.model.ravel()
        # N, p = regressor.dM.X.shape
        # sigma2 = 1 / (N-p-1) * np.sum((z-z_tilde)**2)
        # self.var = np.diag(HInv) * sigma2
        self.stdv = self.var**(1/2)

    def __str__(self):
        """
        Display β in a nice way using pandas. If variance is computed, the standard deviation is shown in the coloumn to the right.

        Returns
        -------
        str
            terminal print
        """       
        
        try:
            sigma = self.stdv
            data = np.zeros((self.p, 2))
            data[:,0] = self.beta
            data[:,1] = sigma
            betapd = pd.DataFrame(data=data, index=self.idx, columns=['β', 'Δβ'])
        except AttributeError:
            betapd = pd.DataFrame(data=self.beta, index=self.idx)
        l = 40; ind = ' '*5
        s = '\n' + '-'*l + '\n'
        s += f'β parameter:\n'
        s += ind + f'polynomial degree  n = {self.n:6.0f}\n'
        s += ind + f'number of features p = {self.p:6.0f}\n'
        s += '-'*l +'\n'
        s += betapd.__str__() + '\n'
        s += '-'*l +'\n'
        return s

    def __getitem__(self, i):
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
        return self.beta[i]

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


    def group(self):

        if self.intercept:
            polydegs = range(self.deg+1)
            beta_amp = np.zeros(self.deg+1)
            i = 0
            for d in range(self.deg):
                p = polydeg2features(d)
                beta_amp[d] = np.mean(self.beta[i:p])
                i = p
            beta_amp[d+1] = np.mean(self.beta[i:])
        
        else:
            polydegs = range(1, self.deg+1)
            beta_amp = np.zeros(self.deg)
            i = 0
            for d in range(1, self.deg):
                p = polydeg2features(d) - 1
                print(i, p-i)
                print(len(self.beta[i:p]))
                beta_amp[d-1] = np.mean(self.beta[i:p])
                i = p
            beta_amp[d] = np.mean(self.beta[i:])
        
        self.amp_betapd = pd.DataFrame(beta_amp, index=[f'β^({d})' for d in polydegs], columns=['β'])

        return self.amp_betapd
        




        
        
class betaCollection:
    """
    Meant as an extension of betaParamter to easily handle several β's when doing e.g. Bootstap.
    """

    def __init__(self, betas):
        """
        Constructs the collection of β-parameters.

        Parameters
        ----------
        betas : list/ndarray of ndarrays/ParameterVectors
            all β to collect
        """        

        if isinstance(betas, list):
            self.nbootstraps = len(betas)
            if isinstance(betas[0], (ParameterVector, ParameterVector)):
                self.p = betas[0].p
                self.betas = np.zeros((self.p,self.nbootstraps))            
                for i in range(self.nbootstraps):
                    self.betas[:,i] = betas[i].getVector()
            if isinstance(betas[0], np.ndarray):
                self.p = betas[0].size
                self.betas = np.zeros((self.p,self.nbootstraps))
                for i in range(self.nbootstraps):
                    self.betas[:,i] = betas[i]

        elif isinstance(betas, np.ndarray):
            self.betas = betas
            self.p, self.nbootstraps = np.shape(self.betas)
        
    def __str__(self):
        """
        Presents the collection nicely using pandas.

        Returns
        -------
        str
            terminal print
        """        
        betaspd = pd.DataFrame(data=self.betas, index=[f'β_{j}' for j in range(self.p)], columns=[f'({i+1})' for i in range(self.nbootstraps)])
        return betaspd.__str__()
    
    def __getitem__(self, index):
        """
        Get β-parameter from collection.

        Parameters
        ----------
        index : int
            the β-parameter number

        Returns
        -------
        ndarray
            the β-parameter number(s) [index]
        """        
        return self.betas[:,index]




