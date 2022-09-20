from hashlib import new
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
        

    def __call__(self, no_bootstraps=100, comparison_mode=False):
        self.nBS = no_bootstraps
        trainer2 = self.trainer

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
    
    def prediction(self):
        z_test = self.predictor.data.ravel()
        z_pred = np.empty((z_test.shape[0], self.nBS))
        z_test2 = np.empty((z_test.shape[0], self.nBS))
        for i in range(self.nBS):
            beta = self.betas[i]
            self.predictor.setOptimalbeta(beta)
            z_pred[:,i] = self.predictor.fit()
            z_test2[:,i] = z_test
        
      
        error = np.mean( np.mean((z_test2 - z_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        print(error, bias, variance, bias+variance)
        

            



   
        



class CrossValidation:

    def __init__(self):
        pass