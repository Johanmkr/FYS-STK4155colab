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



class betaParameter:

    def __init__(self, beta):
        self.beta = beta
        self.p = len(beta)

    def getVector(self):
        return self.beta

    def compVariance(self, data, dM):
        self.var = np.diag(np.var(data) * dM.Hinv)
        return self.var

    def __str__(self):
        betapd = pd.DataFrame(data=self.beta, index=[f'β_{j}' for j in range(len(self.beta))])
        return betapd.__str__()



class betaCollection:

    def __init__(self, betas):

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

        if isinstance(betas, np.ndarray):
            self.betas = betas
            self.p, self.nbootstraps = np.shape(self.betas)

        
    def __str__(self):
        betaspd = pd.DataFrame(data=self.betas, index=[f'β_{j}' for j in range(self.p)], columns=[f'({i+1})' for i in range(self.nbootstraps)])
        return betaspd.__str__()
    

    def __getitem__(self, index):
        return self.betas[:,index]

'''    def getVector(self, index):
        return self.betas[:,index]'''