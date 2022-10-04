from src.utils import *
from src.Regression import linearRegression, Training, Prediction



class noneResampler:

    """
    Motherclass for resampling methods. Has daughters
        * Bootstrap
        * CrossValidation
    """

    def __init__(self, trainer, predictor, no_iterations, mode='manual', scheme='OLS', hyper_param=0) -> None:
        """
        Initiate resampling with data sets and model specs. 

        Parameters
        ----------
        trainer : Training
            the SCALED training set
        predictor : Prediction
            the SCALED prediction set
        no_iterations : int
            number of iterations in resampling algorithm
        mode : str, optional
            (see linearRegression) the mode setting in regression, by default 'manual'
        scheme : str, optional
            (see linearRegression) the method with which to perform regression, by default 'OLS'
        hyper_param : int, optional
            hyper parameter to tune, by default 0
        """
        self.trainer = trainer
        self.predictor = predictor
        self.polydeg = self.trainer.polydeg
        self.nfeatures = self.trainer.nfeatures
        self.Niter = no_iterations
        self.mode, self.scheme = mode.lower(), scheme.lower()
        self.lmbda = hyper_param
        self.method = self.scheme # alias
    
    
    def advance(self, i):
        raise NotImplementedError("Implement advance() in subclass.")

    def __call__(self, no_iterations=None, hyper_param=None):
        """
        Perform the resampling algorithm.

        Parameters
        ----------
        no_iterations : int, optional
            number of iterations in resampling algorithm, by default self.Niter
        hyper_param : _type_, optional
            hyper parameter to tune, by default self.lmbda

        Returns
        -------
        list(*Training), list(*Prediction)
            list with trained objects, list with predicted objects
        """
        self.Niter = no_iterations or self.Niter
        self.lmbda = hyper_param or self.lmbda
        
        self.trainings = []
        self.predictions = []

        for i in range(self.Niter):
            self.advance(i)

        return self.trainings, self.predictions

    def mean_squared_error(self):
        """
        Get MSEs for each iteration.

        Returns
        -------
        ndarray
            array with [train MSE, test MSE] for each iteration alone
        """

        MSEs = np.zeros((2, self.Niter)) # (train, test)
        for i in range(self.Niter):
            MSEs[0, i] = self.trainings[i].MSE
            MSEs[1, i] = self.predictions[i].MSE
  
        return MSEs

    def __len__(self):
        return self.Niter

    def getOptimalParameters(self):
        """
        Retrieve computed parameter vectors and the mean of their elements

        Returns
        -------
        list(*parameterVector), list(*groupedVector),  ndarray
            list of all parameter vectors, their corresponding mean element value
        """
        pVs = []
        amp_pVs = []
        mu_beta = np.zeros(self.Niter)
        for i in range(self.Niter):
            pV = self.trainings[i].pV
            pVs.append(pV)
            amp_pVs.append(pV.group())
            mu_beta[i] = pV.mean()

        return pVs, amp_pVs, mu_beta

    def __str__(self):
        s = r'$d = %i$'%self.polydeg
        if self.scheme != 'ols':
            s += '\n'
            s += r'$\lambda = %.2e$'%self.lmbda
        return s



class Bootstrap(noneResampler):

    def __init__(self, trainer, predictor, no_bootstraps=100, mode='manual', scheme='OLS', hyper_param=0):
        """
        Initiate bootstrap with data sets and model specs. 

        Parameters
        ----------
        trainer : Training
            the SCALED training set
        predictor : Prediction
            the SCALED prediction set
        no_bootstraps : int, optional
            number of iterations in bootstrap, by default 100
        mode : str, optional
            (see linearRegression) the mode setting in regression, by default 'manual'
        scheme : str, optional
            (see linearRegression) the method with which to perform regression, by default 'OLS'
        hyper_param : int, optional
            hyper parameter to tune, by default 0
        """
        super().__init__(trainer, predictor, no_bootstraps, mode, scheme, hyper_param)
        self.B = self.Niter

    def advance(self, i):
        """
        Algorithm for bootstrap.

        Parameters
        ----------
        i : int
            iteration number
        """
        train = self.trainer.randomShuffle()
        pred = deepcopy(self.predictor)
        reg = linearRegression(train, pred, mode=self.mode, scheme=self.scheme)
        reg.fit(self.lmbda)
        train.computeModel()
        pred.predict()

        train.computeExpectationValues()
        pred.computeExpectationValues()

        self.trainings.append(train)
        self.predictions.append(pred)

    def bias_varianceDecomposition(self):
        """
        Decompose bootstrap prediction error in bias^2 and variance.

        Returns
        -------
        float, float, float
            predicition error, bias^2, variance
        """
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

    def ID(self):
        return 'BS'

    
   


class CrossValidation(noneResampler):

    def __init__(self, trainer, predictor, k_folds=5, mode='manual', scheme='OLS', hyper_param=0):
        """
        Initiate k-fold cross validation with data sets and model specs. 

        Parameters
        ----------
        trainer : Training
            the SCALED training set
        predictor : Prediction
            the SCALED prediction set
        k_folds : int, optional
            number of folds, by default 5
        mode : str, optional
            (see linearRegression) the mode setting in regression, by default 'manual'
        scheme : str, optional
            (see linearRegression) the method with which to perform regression, by default 'OLS'
        hyper_param : int, optional
            hyper parameter to tune, by default 0
        """
        super().__init__(trainer, predictor, k_folds, mode, scheme, hyper_param)
        self.k = self.Niter

    def advance(self, i):
        """
        Algorithm for k-fold cross validation.

        Parameters
        ----------
        i : int
            iteration number
        """

        tV = deepcopy(self.trainer.tV)
        dM = deepcopy(self.trainer.dM)

        all_idx = np.arange(0, len(tV), 1)
        part_size = int(len(tV)/self.k)

        test_idx = slice(i*part_size,(i+1)*part_size)
        train_idx = np.delete(all_idx, test_idx)
        test_idx = np.delete(all_idx, train_idx)

        tog = Training(tV[train_idx], dM[train_idx])
        quiz = Prediction(tV[test_idx], dM[test_idx])

        reg = linearRegression(tog, quiz, mode=self.mode, scheme=self.scheme)
        reg.fit(self.lmbda)

        tog.computeModel()
        tog.computeExpectationValues()
        
        quiz.predict()
        quiz.computeExpectationValues()

        self.trainings.append(tog)
        self.predictions.append(quiz)


    def ID(self):
        return 'CV'

