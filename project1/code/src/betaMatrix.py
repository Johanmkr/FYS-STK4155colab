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
    


    