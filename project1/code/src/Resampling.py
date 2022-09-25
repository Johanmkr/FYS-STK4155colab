from src.utils import *

from src.designMatrix import DesignMatrix
from src.Regression import LinearRegression, Training, Prediction
from src.parameterVector import betaCollection

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
            LS = LinearRegression(newdata, self.dM, self.method)
            newbeta = LS()
            newbeta[:,i] = newbeta.getVector()
        self.Beta.addColumns(newbeta)


class Bootstrap:

    def __init__(self, trainer, predictor, no_bootstraps=100):
        self.trainer = trainer
        self.predictor = predictor
        self.polydeg = self.trainer.polydeg

        self.B = no_bootstraps
        
    def __call__(self, no_bootstraps=None):
        self.B = no_bootstraps or self.B
        beta_list = []
        
        trainings = []
        predictions = []
        for i in range(self.B):
            newtrainer = self.trainer.randomShuffle()
            betastar = newtrainer.train()
            beta_list.append(betastar)

            tr = self.trainer.copy()
            tr.setOptimalbeta(betastar)
            tr.computeModel()
            tr.computeExpectationValues()

            pr = self.predictor.copy()
            pr.setOptimalbeta(betastar)
            pr.predict()
            pr.computeExpectationValues()

            trainings.append(tr)
            predictions.append(pr)

        self.betas = betaCollection(beta_list)

        self.trainings = trainings
        self.predictions = predictions

        return self.trainings, self.predictions

    # dont touch this
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

    def mean_squared_error(self):

        MSEs = np.zeros((2, self.B)) # (train, test)
        for i in range(self.B):
            MSEs[0, i] = self.trainings[i].MSE
            MSEs[1, i] = self.predictions[i].MSE
  
      
        return MSEs

   
        



class CrossValidation:

    def __init__(self, regressor):
        self.reg = regressor
        self.data = regressor.data
        self.dM = regressor.dM
        self.trainings = []
        self.predictions = []

        # self.MSE_train = []
        # self.MSE_cv = []
        # self.R2_train = []
        # self.R2_cv = []
        # self.var_cv = []
        # self.bias_cv = []

    def __call__(self, k_folds=5):
        beta_list = []
        rav_data = self.data.ravel()
        all_idx = np.arange(0, len(rav_data), 1)
        k_size = int(rav_data.size / k_folds)
        for i in range(k_folds): 
            test_idx = slice(i*k_size,(i+1)*k_size)
            train_idx = np.delete(all_idx, test_idx)

            z_train = rav_data[train_idx]
            z_test = rav_data[test_idx]
            X_train = self.dM[train_idx,:]
            X_test = self.dM[test_idx, :]

            trainer = Training(self.reg, z_train, self.dM.newObject(X_train))
            predictor = Prediction(self.reg, z_test, self.dM.newObject(X_test))

            beta = trainer.train()
            trainer.computeModel()
            trainer.computeExpectationValues()
            
            predictor.setOptimalbeta(beta)
            predictor.computeModel()
            predictor.computeExpectationValues()

            # self.MSE_train.append(trainer.MSE)
            # self.MSE_cv.append(predictor.MSE)
            # self.R2_train.append(trainer.R2)
            # self.R2_cv.append(predictor.R2)
            # self.var_cv.append(np.var(predictor.model.ravel()))
            # self.bias_cv.append((predictor.data.ravel() - np.mean(predictor.model.ravel())**2))

            z_test = self.predictor.data.ravel()


            self.trainings.append(trainer)
            self.predictions.append(predictor)
            beta_list.append(beta)
        self.betas = betaCollection(beta_list)
        
        return [np.mean(self.MSE_train), np.mean(self.MSE_cv), np.mean(self.R2_train), np.mean(self.R2_cv), np.mean(self.var_cv), np.mean(self.bias_cv)]
        

