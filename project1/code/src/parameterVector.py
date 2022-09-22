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
            self.n = features2polydeg(self.p)
        else:
            self.n = features2polydeg(self.p+1)

        self.stdv = None # standard deviation, to be calculated

        if self.intercept:
            self.idx = [f'β_{j}' for j in range(len(self.beta))]
            self.idx_tex = [r'\beta_{%i}$'%j for j in range(len(self.beta))]
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
        HInv = regressor.dM.Hinv

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