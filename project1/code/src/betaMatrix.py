import numpy as np
import pandas as pd

class betaMatrix:

    def __init__(self, z, dM, beta='none'):
        self.dM = dM
        self.z = z
        self.beta = beta

    def setBeta(self, beta):
        self.beta = beta

    def compVariance(self):
        # from IPython import embed; embed()
        self.variance = np.diag(np.var(self.z) * self.dM.Hinv)

    def addColumns(self, addBeta):
        if len(addBeta.shape) == 1:
            addcolumns = 1
            addBeta = addBeta[:,np.newaxis]
        else:
            dummy, addcolumns = addBeta.shape
        if len(self.beta.shape) == 1:
            beta_new = np.zeros((int(len(self.beta)),1+addcolumns))
            beta_new[:,0] = self.beta 
            beta_new[:,1:] = addBeta 
            self.beta = beta_new
        else:
            nrow, ncol = self.beta.shape
            beta_new = np.zeros((nrow, ncol+addcolumns))
            beta_new[:, :ncol] = self.beta 
            beta_new[:, ncol:] = addBeta
            self.beta = beta_new

    def __str__(self):
        betapd = pd.DataFrame(data=self.beta, index=[f'β_{j}' for j in range(len(self.beta))])
        return betapd.__str__()
        
    



"""
SUGGESTION:
"""


def features2polydeg(features):
    """
    Function for finding the order of the 2D polynomial for a given length of β.

    Parameters
    ----------
    features : int
        number of features in model

    Returns
    -------
    int
        order of 2D polynomial
    """    
    p = features
    coeff = [1, 3, 2-2*p]
    ns = np.roots(coeff)
    n = int(np.max(ns))
    return n


class betaParameter:
    """
    Class for gathering properties of the β-parameter, as fitting a least squares regression for some 2D polynomial basis.

    COMPATIBLE CLASSES:
        * DesignMatrix 
        * LeastSquares (+ subclasses)
        * Bootstrap
    """

    def __init__(self, beta):
        """
        Constructur for the β-parameter.

        Parameters
        ----------
        beta : list/ndarray/betaParameter
            the (optimized) β-parameter
        """        
        if isinstance(beta, (list, np.ndarray)):
            self.beta = np.asarray(beta)
        elif isinstance(beta, betaParameter):
            self.beta = beta.getVector()
        self.p = len(self.beta)
        self.n = features2polydeg(self.p)

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
        # self.var = np.diag(1 * HInv)
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
            betapd = pd.DataFrame(data=data, index=[f'β_{j}' for j in range(len(self.beta))], columns=['β', 'Δβ'])
        except AttributeError:
            betapd = pd.DataFrame(data=self.beta, index=[f'β_{j}' for j in range(len(self.beta))])
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


class betaCollection:
    """
    Meant as an extension of betaParamter to easily handle several β's when doing e.g. Bootstap.
    """

    def __init__(self, betas):
        """
        Constructs the collection of β-parameters.

        Parameters
        ----------
        betas : list/ndarray of ndarrays/betaParameters
            all β to collect
        """        

        if isinstance(betas, list):
            self.nbootstraps = len(betas)
            if isinstance(betas[0], betaParameter):
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

'''    def getVector(self, index):
        return self.betas[:,index]'''