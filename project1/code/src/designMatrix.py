import numpy as np
import pandas as pd


"""
Code for computing and storing a design matrix of a 2D polynomial of nth order.
"""

def matrixColoumns(polydeg):
    cols = ['x^0 y^0']
    for i in range(1, polydeg+1):
        for k in range(i+1):
            cols.append(f'x^{(i-k):1.0f} y^{k:1.0f}')

    return cols

def polydeg2features(polydeg):
    """
    Function for finding the length of β for a given polynomial degree in 2 dimensions.

    Parameters
    ----------
    polydeg : int
        order of 2D polynomial 

    Returns
    -------
    int
        number of features in model
    """
    n = polydeg
    p = int((n+1)*(n+2)/2)
    return p

def feature2polydeg(features):
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


class DesignMatrix:

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

    def _set_X(self, X):
        """
        Set X manually. 

        Parameters
        ----------
        X : np.array(*,p) 
            the design matrix of p features
        """        
        self.X = X
        self.Npoints = np.shape(self.X)[0]
        self.Xpd = pd.DataFrame(self.X, columns=matrixColoumns(self.n))
        self.Hessian()

    def create_X(self, x, y):
        """
        Create the design matrix for a set of points.

        Parameters
        ----------
        x : np.array()
            the x-coordinates of the measurement points
        y : np.array()
            the y-coordinates of the measurement points
        """        
        xx = x.ravel(); yy = y.ravel()
       
        X = np.ones((len(xx), self.p))

        j = 1
        for i in range(1, self.n+1):
            for k in range(i+1):
                X[:,j] = (xx**(i-k))*(yy**k)
                j+=1

        self._set_X(X)

    def newObject(self, X):
        """
        Method that lets us create new objects that do not need computing.

        Parameters
        ----------
        X : np.array(*,self.p)
            the design matrix as array

        Returns
        -------
        DesignMatrix
            the design matrix as object of this class
        """        
        newObject = DesignMatrix(self.n)
        newObject._set_X(X)
        return newObject

    def Hessian(self):
        """
        Compute the hessian matrix and its inverse.

        Returns
        -------
        np.array(self.p, self.p) #???
            the hessian
        """        
        self.H = self.X.T @ self.X
        self.Hinv = np.linalg.pinv(self.H)

    def __str__(self):
        """
        Display design matrix using pandas
        """
        l = 40; ind = ' '*5
        s = '-'*l + '\n'
        s += f'design matrix X:\n'
        s += ind + f'polynomial degree  n = {self.n:6.0f}\n'
        s += ind + f'number of features p = {self.p:6.0f}\n'
        s += ind + f'number of points   N = {self.Npoints:6.0f}\n'
        s += '-'*l +'\n'
        s += self.Xpd.__str__()
        return s

    def __mul__(self, other):
        #uncertain about this
        return self.X @ other.X