
from src.utils import *
from sklearn.preprocessing import StandardScaler


"""
Code for computing and storing a design matrix of a 2D polynomial of nth order.
"""

def matrixColoumns(polydeg):
    cols = ['x^0 y^0']
    for i in range(1, polydeg+1):
        for k in range(i+1):
            cols.append(f'x^{(i-k):1.0f} y^{k:1.0f}')

    return cols


class DesignMatrix:
    """
    Class for gathering properties of the design matrix of a 2D polynomial,

        z = f(x, y)  = c0 + c1*x + c2*y + c3*x^2 + c4*xy + c5*y^2 + ...,

    for a given polynomial degree.

    COMPATIBLE CLASSES:
        * ParameterVector 
        * LeastSquares (+ subclasses)
        * Bootstrap
    """

    def __init__(self, polydeg):
        """
        Constructor for the 2D polynomial design matrix.

        Parameters
        ----------
        polydeg : int
            2D polynomial degree
        """              
        self.n = polydeg
        self.p = polydeg2features(self.n)
        self.scaled = False

    def getMatrix(self):
        """
        Get the actual matrix.

        Returns
        -------
        np.array(*,self.p)
            the design matrix of self.p features
        """        
        return self.X
    
    def __call__(self, x, y):
        """
        Alias for self.createX(x, y).

        Parameters
        ----------
        x : np.array()
            the x-coordinates of the measurement points
        y : np.array()
            the y-coordinates of the measurement points
        """
        self.createX()
        

    def _setX(self, X):
        """
        Set X manually. 

        Parameters
        ----------
        X : ndarray(*,p) 
            the design matrix of p features
        """  
        self.X_org = X
        
        if np.all(np.isnan(X[:,0])) or np.all(np.abs(X[:,0])<1e-18):  
            self.X = X[:,1:]  # remove intercept coloumn
            self.interceptColoumn = False
            self.scaled = True # ? 
            cols = matrixColoumns(self.n)[1:]
        elif not self.interceptColoumn:
            self.X = X
            cols = matrixColoumns(self.n)[1:]
        else:
            self.X = X
            cols = matrixColoumns(self.n)
        self.Npoints, self.p = np.shape(self.X)
        self.Xpd = pd.DataFrame(self.X, columns=cols)
        self.Hessian()

    def createX(self, x, y):
        """
        Create the design matrix for a set of points.

        Parameters
        ----------
        x : ndarray
            the x-coordinates of the measurement points
        y : ndarray
            the y-coordinates of the measurement points
        """        
        xx = x.ravel(); yy = y.ravel()
       
        X = np.ones((len(xx), self.p))

        j = 1
        for i in range(1, self.n+1):
            for k in range(i+1):
                X[:,j] = (xx**(i-k))*(yy**k)
                j+=1
        self.interceptColoumn = True
        self._setX(X)
 
    def newObject(self, X, is_scaled=None, no_intercept=None):
        """
        Method that lets us create new objects that do not need computing.

        Parameters
        ----------
        X : ndarray(*,self.p)
            the design matrix as array
        is_scaled : bool
            whether X is scaled (=True) or not (=False) and is supposed to be None >> self.scaled, by default None

        Returns
        -------
        DesignMatrix
            the design matrix as object of this class
        """        
        newObject = DesignMatrix(self.n)
        newObject.scaled = is_scaled or self.scaled
        newObject.interceptColoumn = not no_intercept or self.interceptColoumn
        newObject._setX(X)
        return newObject

    def Hessian(self):
        """
        Compute the hessian matrix and its inverse.

        Returns
        -------
        ndarray(self.p, self.p)
            the hessian
        """
        self.H = self.X.T @ self.X
        self.Hinv = np.linalg.pinv(self.H)

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
        if not self.interceptColoumn:
            s += '(intercept coloumn removed)\n'
        s += ind + f'polynomial degree  n = {self.n:6.0f}\n'
        s += ind + f'number of features p = {self.p:6.0f}\n'
        s += ind + f'number of points   N = {self.Npoints:6.0f}\n'
        s += '-'*l +'\n'
        s += self.Xpd.__str__() + '\n'
        s += '-'*l +'\n'
        return s

    def __getitem__(self, index):
        """
        Yield row vector of X.

        Parameters
        ----------
        index : int
            the row number (index) of X

        Returns
        -------
        ndarray
            the row vecotrs(s) of X in question
        """
        return self.X[index]

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

