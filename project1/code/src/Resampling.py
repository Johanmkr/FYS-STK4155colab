import numpy as np

from src.Regression import LeastSquares



class Bootstrap:

    """
    _summary_
    """    

    def __init__(self, method, dM, Beta):
        self.dM = dM
        self.method = method
        self.Beta = Beta

    def perform(self, data, no_bootstrap):
        newbeta = np.zeros((len(self.Beta.beta), no_bootstrap))
        for i in range(no_bootstrap):
            idx = np.random.randint(0,len(data), len(data))
            newdata = data[idx]
            LS = LeastSquares(newdata, self.dM)
            LS.split(test_size=0)
            LS("OLS")
            newbeta[:,i] = LS.beta
        self.Beta.addColumns(newbeta)

            



   
        



class CrossValidation:

    def __init__(self):
        pass