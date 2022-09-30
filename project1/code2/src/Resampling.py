
from src.Regression import linearRegression, Training, Prediction
from src.utils import *


from src.objects import parameterVector, targetVector, designMatrix

from copy import deepcopy




class resampleTechnique:
    def __init__(self, trainer, predictor, no_iterations=100, mode='manual', method='OLS', hyper_param=0) -> None:
        self.trainer = trainer
        self.predictor = predictor
        self.polydeg = self.trainer.polydeg
        self.nfeatures = self.trainer.nfeatures
        self.Niter = no_iterations
        self.mode, self.method = mode, method
        self.lmbda = hyper_param

    def __call__(self, no_iterations=None, hyper_param=None):
        self.Niter = no_iterations or self.Niter
        self.lmbda = hyper_param or self.lmbda
        
        self.trainings = []
        self.predictions = []

        for i in range(self.Niter):
            self.advance(i)

        return self.trainings, self.predictions

    def mean_squared_error(self):

        MSEs = np.zeros((2, self.Niter)) # (train, test)
        for i in range(self.Niter):
            MSEs[0, i] = self.trainings[i].MSE
            MSEs[1, i] = self.predictions[i].MSE
  
        return MSEs

    def __len__(self):
        return self.Niter

    def getOptimalParameters(self):
        pVs = []
        mu_beta = np.zeros(self.Niter)
        for i in range(self.Niter):
            pV = self.trainings[i].pV
            pVs.append(pV)
            mu_beta[i] = pV.mean()

        return pVs, mu_beta



class Bootstrap(resampleTechnique):

    def __init__(self, trainer, predictor, no_bootstraps=100, mode='manual', method='OLS', hyper_param=0):
        super().__init__(trainer, predictor, no_bootstraps, mode, method, hyper_param)
        self.B = self.Niter

    def advance(self, i):
        train = self.trainer.randomShuffle()
        pred = deepcopy(self.predictor)
        reg = linearRegression(train, pred, mode=self.mode, method=self.method)
        reg.fit(self.lmbda)
        train.computeModel()
        pred.predict()

        train.computeExpectationValues()
        pred.computeExpectationValues()

        self.trainings.append(train)
        self.predictions.append(pred)


    def bias_varianceDecomposition(self):
        z_test = self.predictor.data.ravel()
        z_pred = np.empty((z_test.shape[0], self.B))
        diff = np.zeros_like(z_pred)

        for i in range(self.B):
            z_pred[:,i] = self.predictions[i].model.ravel()
            diff[:,i] = z_test - z_pred[:,i]

        Errors = np.mean(diff**2, axis=1, keepdims=True)[:,0]
        Bias2s = (z_test - np.mean(z_pred, axis=1, keepdims=True)[:,0])**2
        Vars = np.var(z_pred, axis=1, keepdims=True)[:,0]

        self.error = np.mean(Errors)
        self.bias2 = np.mean(Bias2s)
        self.var = np.mean(Vars)

        tol = 1e-8
        assert abs(self.bias2 + self.var - self.error) < tol

        return self.error, self.bias2, self.var

    
   
    


class CrossValidation(resampleTechnique):

    def __init__(self, trainer, predictor, k_folds=5, mode='manual', method='OLS', hyper_param=0):
        super().__init__(trainer, predictor, k_folds, mode, method, hyper_param)
        self.k = self.Niter


    def advance(self, i):

        tV = deepcopy(self.trainer.tV)
        dM = deepcopy(self.trainer.dM)

        all_idx = np.arange(0, len(tV), 1)
        part_size = int(len(tV)/self.k)

        test_idx = slice(i*part_size,(i+1)*part_size)
        train_idx = np.delete(all_idx, test_idx)
        test_idx = np.delete(all_idx, train_idx)

        tog = Training(tV[train_idx], dM[train_idx])
        quiz = Prediction(tV[test_idx], dM[test_idx])

        reg = linearRegression(tog, quiz, mode=self.mode, method=self.method)
        reg.fit(self.lmbda)

        tog.computeModel()
        tog.computeExpectationValues()
        
        quiz.predict()
        quiz.computeExpectationValues()

        self.trainings.append(tog)
        self.predictions.append(quiz)

