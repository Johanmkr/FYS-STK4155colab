try:
    from utils import *
except ModuleNotFoundError:
    from src.utils import *

import autograd.numpy as np
from autograd import elementwise_grad as egrad


import numpy as np0




class noneRULE:
    """
    Mother-class for update rules.
    """
    def __init__(self, theta0:np0.ndarray, eta:float=0.0, gamma:float=0.0, rho1:float=0.0, rho2:float=0.0, epsilon:float=0.0):
        """
        Contruct the update rule (optimiser) with initial vector and parameter values.

        Parameters
        ----------
        theta0 : ndarray
            initial parameter vector
        eta : float, optional
            global learning rate, by default 0.0
        gamma : float, optional
            momentum parameter, by default 0.0
        rho1 : float, optional
            hyperparameter, by default 0.0
        rho2 : float, optional
            hyperparameter, by default 0.0
        epsilon : float, optional
            small number for numerical stability, by default 0.0
        """
        self.theta = theta0
        self.eta = eta
        self.params = {'eta':eta, 'gamma':gamma, 'rho1':rho1, 'rho2':rho2, 'epsilon':epsilon}
        self.idx = 0
        self.v = np.zeros_like(self.theta)
        self.schedule = lambda k: self.eta

    def set_params(self, eta:float=None, gamma:float=None, rho1:float=None, rho2:float=None, epsilon:float=None):
        """
        Set parameters needed for update rule

        Parameters
        ----------
        eta : float, optional
            global learning rate, by default self.params['eta']
        gamma : float, optional
            momentum parameter, by default self.params['gamma']
        rho1 : float, optional
            hyperparameter, by default self.params['rho1']
        rho2 : float, optional
            hyperparameter, by default self.params['rho2']
        epsilon : float, optional
            small number for numerical stability, by default self.params['epsilon']
        """
        for param in self.params:
            val = eval(param) or self.params[param]
            self.params[param] = val
        self.eta = self.params['eta']
    
    def __call__(self, grad:np0.ndarray, theta:np0.ndarray):
        """
        Find next θ.

        Parameters
        ----------
        grad : ndarray
            gradient of loss function for correct argument θ
        theta : ndarray
            current parameter vector

        Returns
        -------
        ndarray
            updated parameter vector
        """
        self.update(theta, grad)
        return self.theta

    def next(self, it:int=None):
        """
        Update iteration number of instance.

        Parameters
        ----------
        it : int, optional
            current iteration, by default self.idx+1
        """
        self.idx = self.idx+1 or it
        self.schedule(self.idx) # not sure this is necessary

    def set_learning_schedule(self, tau:int, eta_0:float=None, eta_tau:float=None):
        """
        Set a learning schedule for the learning parameter η.

        Parameters
        ----------
        tau : int
            max. iteration that schedule cares about
        eta_0 : float, optional
            initial η, by default self.eta
        eta_tau : float, optional
            last η, by default 0.01*eta_0
        """
        eta0 = eta_0 or self.eta
        etatau = eta_tau or 0.01*eta0
        self.schedule = lambda k: self.standard_learning_schedule(k, eta0, etatau, tau)
        
    def standard_learning_schedule(self, k:int, eta_0:float, eta_tau:float, tau:int):
        """
        Apply the standard learning schedule.

        Parameters
        ----------
        k : int
            _description_
        eta_0 : float
            initial η
        eta_tau : float
            last η
        tau : int
            max. iteration that schedule cares about
        """
        kt = k/tau
        if kt <= 1:
            eta = (1-kt) * eta_0 + kt*eta_tau
        else:
            eta = eta_tau
        self.set_learning_rate(eta)
    
    def set_learning_rate(self, eta:float):
        """
        Set instance's learning rate to some value.

        Parameters
        ----------
        eta : float
            global learning rate
        """
        self.eta = eta

    def update(self, theta:np0.ndarray, grad:np0.ndarray):
        """
        Update θ by momentum (plain if γ=0) method.

        Parameters
        ----------
        theta : ndarray
            current θ
        grad : ndarray
            gradient of loss function at befitting arguments
        """
        # standard:
        self.v = self.gamma*self.v - self.eta*grad
        self.theta = theta + self.v

    def get_diff(self):
        """
        Return the change in θ from last iteration to current.

        Returns
        -------
        ndarray
            change in θ
        """
        return self.v

class noneRULEadaptive(noneRULE):
    """
    Mother-class for adaptive update rules.
    """
    def __init__(self, theta0:np0.ndarray, eta:float=0, rho1:float=0, rho2:float=0, epsilon:float=1e-7):
        """
        Initialise an adaptive optimiser

        Parameters
        ----------
        theta0 : ndarray
            initial parameter vector
        eta : float, optional
            global learning rate, by default 0.0
        rho1 : float, optional
            hyperparameter, by default 0.0
        rho2 : float, optional
            hyperparameter, by default 0.0
        epsilon : float, optional
            small number for numerical stability, by default 1e-7
        """
        super().__init__(theta0, eta, epsilon=epsilon, rho1=rho1, rho2=rho2)
        self.r = np.zeros_like(self.theta)
        self.epsilon = epsilon # numerical stability

    def set_params(self, eta:float=None, rho1:float=None, rho2:float=None, epsilon:float=None):
        """
        _summary_

        Parameters
        ----------
        eta : float, optional
            global learning rate, by default self.params['eta']
        rho1 : float, optional
            hyperparameter, by default self.params['rho1']
        rho2 : float, optional
            hyperparameter, by default self.params['rho2']
        epsilon : float, optional
            small number for numerical stability, by default self.params['epsilon']
        """
        super().set_params(eta, 0, rho1, rho2, epsilon)
        self.epsilon = self.params['epsilon']



class rule_ClassicalSGD(noneRULE):
    def __init__(self, theta0:np0.ndarray, eta0:float=0.01):
        """
        Construct the plain (S)GD scheme.

        Parameters
        ----------
        theta0 : ndarray
            initial parameter vector
        eta0 : float, optional
            global learning rate, by default 0.01
        """
        super().__init__(theta0, eta0, gamma=0)
        self.params = {key:self.params[key] for key in ['eta']}
        
    def set_params(self, eta:float=None):
        """
        Set learning rate.

        Parameters
        ----------
        eta : float, optional
            global learning rate, by default None
        """
        super().set_params(eta, 0)
        self.gamma = 0

class rule_MomentumSGD(noneRULE):
    def __init__(self, theta0:np0.ndarray, eta0:float=0.01, gamma:float=0.9):
        """
        Construct the momentum (S)GD scheme.

        Parameters
        ----------
        theta0 : ndarray
            initial parameter vector
        eta0 : float, optional
            global learning rate, by default 0.01
        gamma : float, optional
            momentum parameter, by default 0.9
        """
        super().__init__(theta0, eta0, gamma=gamma)
        self.params = {key:self.params[key] for key in ['eta', 'gamma']}

    def set_params(self, eta0:float=None, gamma:float=None):
        """
        Set learning rate and momentum parameter.

        Parameters
        ----------
        eta0 : float, optional
            global learning rate, by default None
        gamma : float, optional
            momentum parameter, by default None
        """
        super().set_params(eta0, gamma)
        self.gamma = self.params['gamma']

class rule_AdaGrad(noneRULEadaptive):
    def __init__(self, theta0:np0.ndarray, eta:float=0.01, epsilon:float=1e-7):
        """
        Construct the AdaGrad scheme.

        Parameters
        ----------
        theta0 : ndarray
            initial parameter vector
        eta : float, optional
            global learning rate, by default 0.01
        epsilon : float, optional
            small number for numerical stability, by default 1e-7
        """
        super().__init__(theta0, eta, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'epsilon']}
    
    def update(self, theta:np0.ndarray, grad:np0.ndarray):
        """
        Update θ by the AdaGrad rule.

        Parameters
        ----------
        theta : ndarray
            current θ
        grad : ndarray
            gradient of loss function at befitting arguments
        """
        self.r = self.r + grad**2
        self.v = - self.eta*(self.epsilon+np.sqrt(self.r))**(-1) * grad # check if this is element-wise
        self.theta = theta + self.v 

    def set_params(self, eta:float=None, epsilon:float=None):
        """
        Set parameters for AdaGrad.

        Parameters
        ----------
        eta : float, optional
            global learning rate, by default None
        epsilon : float, optional
            small number for numerical stability, by default None
        """
        super().set_params(eta, None, None, epsilon)

class rule_RMSProp(noneRULEadaptive):
    def __init__(self, theta0:np0.ndarray, eta:float=0.01, rho:float=0.9, epsilon:float=1e-7):
        """
        Construct the RMSProp scheme.

        Parameters
        ----------
        theta0 : np0.ndarray
            initial parameter vector
        eta : float, optional
            global learning rate, by default 0.01
        rho : float, optional
            hyperparameter ρ in RMSProp, by default 0.9
        epsilon : float, optional
           small number for numerical stability, by default 1e-7
        """
        super().__init__(theta0, eta, rho2=rho, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho2', 'epsilon']}

    def update(self, theta:np0.ndarray, grad:np0.ndarray):
        """
        Update θ by the RMSProp rule.

        Parameters
        ----------
        theta : ndarray
            current θ
        grad : ndarray
            gradient of loss function at befitting arguments
        """
        self.r = self.rho*self.r + (1-self.rho)*grad**2
        self.v = - self.eta / (np.sqrt(self.epsilon + self.r))*grad # is this element-wise? should be...
        self.theta = theta + self.v

    def set_params(self, eta:float=None, rho:float=None,  epsilon:float=None):
        """
        Set parameters for RMSProp.

        Parameters
        ----------
        eta : float, optional
            global learning rate, by default None
        rho : float, optional
            hyperparameter ρ in RMSProp
        epsilon : float, optional
            small number for numerical stability, by default None
        """
        super().set_params(eta, None, rho, epsilon)
        self.rho = self.params['rho2']

class rule_Adam(noneRULEadaptive):
    def __init__(self, theta0:np0.ndarray, eta:float=0.001, rho1:float=0.9, rho2:float=0.999, epsilon:float=1e-8):
        """
        Construct the Adam scheme.

        Parameters
        ----------
        theta0 : ndarray
            initial parameter vector
        eta : float, optional
            global learning rate, by default 0.001
        rho1 : float, optional
            hyperparameter ρ1 in Adam, by default 0.9
        rho2 : float, optional
            hyperparameter ρ2 in Adam,, by default 0.999
        epsilon : float, optional
            _description_, by default 1e-8
        """
        super().__init__(theta0, eta, rho1=rho1, rho2=rho2, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho1', 'rho2', 'epsilon']}
        self.s = np.zeros_like(self.theta)
            
    def update(self, theta:np0.ndarray, grad:np0.ndarray, k:int=None):
        """
        Update θ by the Adam rule.

        Parameters
        ----------
        theta : ndarray
            current θ
        grad : ndarray
            gradient of loss function at befitting arguments
        k : int
            current iteration
        """
        k = k or self.idx; k+=1
        self.s = self.rho1*self.s + (1-self.rho1)*grad
        self.r = self.rho2*self.r + (1-self.rho2)*grad**2
        s_hat = self.s * (1-self.rho1**k)**(-1)
        r_hat = self.r * (1-self.rho2**k)**(-1)
        self.v = - self.eta*s_hat * (np.sqrt(r_hat) + self.epsilon)**(-1) # is this element-wise? should be...
        self.theta = theta + self.v

    def set_params(self, eta:float=None, rho1:float=None, rho2:float=None, epsilon:float=None):
        """
        _summary_

        Parameters
        ----------
        eta : float, optional
            global learning rate, by default None
        rho1 : float, optional
            hyperparameter ρ1 in Adam, by default None
        rho2 : float, optional
            hyperparameter ρ2 in Adam,, by default None
        epsilon : float, optional
            _description_, by default None
        """
        super().set_params(eta, rho1, rho2, epsilon)
        self.rho1 = self.params['rho1']
        self.rho2 = self.params['rho2']








class noneGradientDescent:
    """
    Mother-class for GD and SGD.
    """
    def __init__(self, X:np0.ndarray, y:np0.ndarray, eta:float, theta0:np0.ndarray, no_epochs:int, tolerance:float):
        """
        Set up a (S)GD algorithm.

        Parameters
        ----------
        X : np0.ndarray
            design matrix
        y : np0.ndarray
            target data
        eta : float
            global learning rate
        theta0 : np0.ndarray
            initial parameter vector
        no_epochs : int
            number of complete iterations (epochs)
        tolerance : float
            tolerance for when to stop iterations
        """
        self.X = X
        self.y = y
        self.eta = eta  #Learning rate
        if isinstance(theta0, (float, int)):
            self.theta = np.ones(1)*theta0
        else:
            self.theta = np.asarray(theta0)
        self.no_epochs = no_epochs
        self.n_obs = np.shape(X)[0]
        self.tol = tolerance
        self.v = np.zeros_like(self.theta)

        self.indices = np.arange(len(y))
        self.current_epoch = 0

    @classmethod
    def simple_initialise(cls, eta:float, tolerance:float=1e-7):
        """
        Class method for using update rules with NeuralNetwork.

        Parameters
        ----------
        eta : float
            gloabl learning rate
        tolerance : float, optional
            (unnecessary tolerance), by default 1e-7

        Returns
        -------
        SGD or GD
            the instance resulting from the dummy initialisation 
        """
        return cls(np.zeros((10,1)), np.zeros(10), eta, 0, 100, tolerance)

    def simple_update(self, gradient:np0.ndarray, theta:np0.ndarray):
        """
        Method for updating correctly with NeuralNetwork.

        Parameters
        ----------
        gradient : ndarray
            gradient of loss function at befitting arguments
        theta : ndarray
            current θ
            
        Returns
        -------
        ndarray
            updated θ
        """
        theta_new = self.update_rule(gradient, theta)
        return theta_new

    def per_batch(self, grad:np0.ndarray):
        """
        Algortihm for a batch.

        Parameters
        ----------
        grad : ndarray
            gradient of loss function at befitting arguments
        """
        self.theta = self.update(grad)
        self.update_rule.next()
       
    def set_params(self, **params):
        """
        Set parameters in update rule.
        """
        # set params in update rule
        self.update_rule.set_params(**params)

    def add_nag(self):
        """
        Apply Nesterov momentum.
        """
        # try-except?
        self.grad = lambda k=0: self.compute_gradient(self.theta + self.update_rule.gamma+self.v)

    def set_update_rule(self, scheme:str, params:dict={}, NAG=False):
        """
        Define update rule.

        Parameters
        ----------
        scheme : str
            name of optimiser
        params : dict, optional
            relevant hyperparameters, by default {}
        NAG : bool, optional
            whether to apply NAG (True) or not (False), by default False

        Raises
        ------
        ValueError
            if the scheme is not recognised
        """
        rule = scheme.strip().lower()
        momentum = False
        self.adaptive = False
        if rule in ['plain', 'none', 'manual']:
            # params : eta0
            update_rule = rule_ClassicalSGD(self.theta, **params)
            self.optimiser = 'plain'
        elif rule in ['mom', 'momentum']:
            # params: eta0, gamma
            update_rule = rule_MomentumSGD(self.theta, **params)
            momentum = True
            self.optimiser = 'momentum'
        elif rule in ['ada', 'adagrad']:
            # params: eta, epsilon
            update_rule = rule_AdaGrad(self.theta, **params)
            self.adaptive = True
            self.optimiser = 'AdaGrad'
        elif rule in ['rms', 'rmsprop']:
            # params: eta, rho, epsilon
            update_rule = rule_RMSProp(self.theta, **params)
            self.adaptive = True
            self.optimiser = 'RMSProp'
        elif rule in ['adam']:
            # params: eta, rho1, rho2, epsilon
            update_rule = rule_Adam(self.theta, **params)
            self.adaptive = True
            self.optimiser = 'Adam'
        else: 
            raise ValueError(f"The library does not have functionalities for {rule} optimiser.")
        

        update_rule.set_params(self.eta)
        self.update_rule = update_rule
        self.update = lambda A: update_rule(A, self.theta)
        if momentum and NAG:
            self.add_nag()

    def apply_learning_schedule(self, eta_0:float=None, eta_tau:float=None, tau:int=None):
        """
        Apply standard learning schedyke.

        Parameters
        ----------
        eta_0 : float, optional
            initial learning rate, by default None
        eta_tau : float, optional
            last learning rate, by default None
        tau : int, optional
            number of iterations the schedule considers, by default self.n_obs*200
        """
        self.update_rule.set_learning_schedule(tau or self.n_obs*200, eta_0 or self.eta, eta_tau)

    def regression_setting(self, regularisation:float=0):
        """
        Use the MSE as loss function.

        Parameters
        ----------
        regularisation : float, optional
            penalty parameter λ (in Ridge regression), by default 0
        """
        self.lmbda = regularisation

        def gradient(x, y, theta):
            lf = lambda theta: np.mean((x@theta - y)**2)/2 + self.lmbda*np.mean(theta**2)
            return egrad(lf)(theta)

        self.grad = gradient

    def classification_setting(self, regularisation:float=0):
        """
        Use the cross entropy as loss function. OBS! Not yet tested.

        Parameters
        ----------
        regularisation : float, optional
            small regularisation λ, by default 0
        """
        def gradient(x, y, theta):
            # FIXME
            def lf(theta):
                xt = x@theta # is this ok? What about intercept??
                return - np.mean( y*(xt )  - np.log10(1 +np.exp(xt) )) # cross entropy
            return egrad(lf)(theta)
        self.grad = gradient
        
    def __call__(self, no_epochs:int=None, deregestration_every:int=None):
        """
        Find minimium of loss function.

        Parameters
        ----------
        no_epochs : int, optional
            number of iterations, by default self.no_epochs
        deregestration_every : int, optional
            print MSE every ... iteration, by default self.no_epochs//4

        Returns
        -------
        ndarray
            the optimal θ
        """
        n_iter = no_epochs or self.no_epochs
        say = deregestration_every or self.no_epochs//4
        say = int(say)
        start = self.current_epoch
        for k in trange(start, start+n_iter):
            self.current_epoch = k
            self.per_epoch(self.grad)
            diff = np.abs(self.update_rule.get_diff())
            if np.all(diff) < self.tol:
                print(f"Stopped after {k} epochs as |v| < {self.tol:.0e}.")
                break
            if (k-1) % say == 0 and k > say-1:
                print(f'MSE = {self.mean_squared_error():.4f}')
        self.current_epoch = k
        self.no_epochs = self.current_epoch
        return self.theta

    def set_loss_function(self, loss_function:Callable):
        """
        Manually set loss function.

        Parameters
        ----------
        loss_function : Callable(x:ndarray, y:ndarray, theta:ndarray)
            loss function as function of the features (x), the target value (y) and the parameter vector (theta)
        """
        self.grad = lambda x, y, theta: egrad(lambda theta: loss_function(x, y, theta))(theta)

    def mean_squared_error(self, X:ndarray=None, y:ndarray=None, theta:ndarray=None):
        """
        Return the mean squared error.

        Parameters
        ----------
        X : ndarray, optional
            design matrix, by default self.X
        y : ndarray, optional
            target values, by default self.y
        theta : ndarray, optional
            parameter vector, by default self.theta

        Returns
        -------
        float
            the MSE
        """
        # wack solution 
        try:
            X = X or self.X
            y = y or self.y
        except ValueError:
            X = X; y = y
        try:
            theta = theta or self.theta
        except ValueError:
            theta = theta 
        return np.mean((X@theta - y)**2)

    def get_params(self, terminal_print=True):
        """
        Give information related to update rule.

        Parameters
        ----------
        terminal_print : bool, optional
            whether to print (True) the information or not (False), by default True

        Returns
        -------
        dict
            hyperparameters in update rule
        """
        if terminal_print:
            for param in self.update_rule.params:
                val = self.update_rule.params[param]
                print(f'{param:10s} = {val:10.2e}')

        return self.update_rule.params

        



class GD(noneGradientDescent):
    def __init__(self, X:np0.ndarray, y:np0.ndarray, eta:float, theta0:np0.ndarray, no_epochs:int=500, tolerance=1e-6):
        """
        Set up a GD algorithm.

        Parameters
        ----------
        X : np0.ndarray
            design matrix
        y : np0.ndarray
            target data
        eta : float
            global learning rate
        theta0 : ndarray
            initial parameter vector
        no_epochs : int
            number of complete iterations (epochs)
        tolerance : float
            tolerance for when to stop iterations
        """
        super().__init__(X, y, eta, theta0, no_epochs, tolerance)
    
    def __str__(self):
        """
        Return name of scheme.

        Returns
        -------
        str
            name of update rule
        """
        return self.optimiser + " GD"

    def per_epoch(self, grad:Callable):
        """
        Algorithm for a single epoch.

        Parameters
        ----------
        grad : Callable(X:ndarray, y:ndarray, theta:ndarray)
            the gradient as a function of the design matrix (X), the target values (y) and the parameter vector (theta)
        """
        self.per_batch(grad(self.X, self.y, self.theta))


class SGD(noneGradientDescent):
    def __init__(self, X, y, eta:float, theta0:np0.ndarray, no_epochs:int=500, no_minibatches:int=5, tolerance=1e-6):
        """
        Set up an SGD algorithm.

        Parameters
        ----------
        X : np0.ndarray
            design matrix
        y : np0.ndarray
            target data
        eta : float
            global learning rate
        theta0 : np0.ndarray
            initial parameter vector
        no_epochs : int
            number of complete iterations (epochs)
        no_minibatches : int
            number of minibatches (subsets)
        tolerance : float
            tolerance for when to stop iterations
        """
        super().__init__(X, y, eta, theta0, no_epochs, tolerance)
        self.m = int(no_minibatches)
        assert self.m <= self.n_obs

    def __str__(self):
        """
        Return name of scheme.

        Returns
        -------
        str
            name of update rule
        """
        if self.m == 1:
            return self.optimiser + " GD"
        else:
            return self.optimiser + " SGD"
   
    def per_epoch(self, grad):
        """
        Algorithm for a single epoch.

        Parameters
        ----------
        grad : Callable(X:ndarray, y:ndarray, theta:ndarray)
            the gradient as a function of the design matrix (X), the target values (y) and the parameter vector (theta)
        """
        indices = np.arange(self.n_obs)
        np.random.shuffle(indices)
        batches = np.array_split(indices, self.m)

        for k in range(self.m):
            x_batch = self.X[batches[k]]
            y_batch = self.y[batches[k]]
            self.per_batch(grad(x_batch, y_batch, self.theta))








