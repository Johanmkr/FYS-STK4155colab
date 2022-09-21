
import numpy as np
from src.betaMatrix import betaCollection
from src.designMatrix import DesignMatrix

from src.Regression import LeastSquares
from src.betaMatrix import betaParameter, betaCollection




class Bootstrap1:

    """
    _summary_
    """    

    def __init__(self, method, dM, Beta):
        self.dM = dM
        self.method = method
        self.Beta = Beta

    def perform(self, data, no_bootstrap):
        data = data.ravel()
        newbeta = np.zeros((len(self.Beta.beta), no_bootstrap))
        for i in range(no_bootstrap):
            idx = np.random.randint(0,len(data), len(data))
            newdata = data[idx]
            LS = LeastSquares(newdata, self.dM, self.method)
            newbeta = LS()
            newbeta[:,i] = newbeta.getVector()
        self.Beta.addColumns(newbeta)


class Bootstrap:

    def __init__(self, trainer, predictor):
        self.trainer = trainer
        self.predictor = predictor
        self.polydeg = self.trainer.polydeg
        
    def __call__(self, no_bootstraps=100, comparison_mode=False):
        self.nBS = no_bootstraps

        beta_list = []
        train_list =[]
        for i in range(self.nBS):
            newtrainer = self.trainer.randomShuffle()
            betastar = newtrainer.train()
            beta_list.append(betastar)
            train_list.append(newtrainer)

        if comparison_mode:
            beta_list2 = []
            for i in range(self.nBS):
                newtrainer = train_list[i]
                newtrainer.changeMode()
                betastar = newtrainer.train()
                beta_list2.append(betastar)

            self.betas2 = betaCollection(beta_list2)



        self.betas = betaCollection(beta_list)

    # dont touch this
    def bias_varianceDecomposition(self):
        z_test = self.predictor.data.ravel()
        z_pred = np.empty((z_test.shape[0], self.nBS))
        diff = np.zeros_like(z_pred)

        for i in range(self.nBS):
            self.predictor.setOptimalbeta(self.betas[i])
            z_pred[:,i] = self.predictor.fit()
            diff[:,i] = z_test - z_pred[:,i]


        Errors = np.mean(diff**2, axis=1, keepdims=True)[:,0]
        Bias2s = (z_test - np.mean(z_pred, axis=1, keepdims=True)[:,0])**2
        Vars = np.var(z_pred, axis=1, keepdims=True)[:,0]

        self.MSE = np.mean(Errors)
        self.bias2 = np.mean(Bias2s)
        self.var = np.mean(Vars)

        tol = 1e-8
        assert abs(self.bias2 + self.var - self.MSE) < tol

   
        



class CrossValidation:

    def __init__(self):
        pass