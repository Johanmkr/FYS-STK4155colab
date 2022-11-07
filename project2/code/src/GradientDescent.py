import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad as agrad
import matplotlib.pyplot as plt
from IPython import embed






"""
Need following update rules and initialisations:
* plain
* momentum
* Adam
* AdaGrad
* RMSProp
"""

"""
Will be used inside of class:

Example: (idea)
def __init__(self, optimiser:str):
    self.perturbate = parseOptimiser(optimiser)

def __call__(self):
    for epoch in epochs:
        ...
        for batch in batches:
            grad_k = ...
            theta_k = self.perturbate(theta_k, grad_k, eta_k )
            ...


"""




class noneRULE:
    def __init__(self, theta0, eta:float=0.0, gamma:float=0.0, rho1:float=0.0, rho2:float=0.0, epsilon:float=0.0):
        self.theta = theta0
        self.eta = eta
        self.params = {'eta':eta, 'gamma':gamma, 'rho1':rho1, 'rho2':rho2, 'epsilon':epsilon}
        self.idx = 0
        self.v = np.zeros_like(self.theta)
        self.schedule = lambda k: self.eta

    '''@classmethod
    def init(cls):
        return cls(0, )'''

    def set_params(self, eta:float=None, gamma:float=None, rho1:float=None, rho2:float=None, epsilon:float=None):
        for param in self.params:
            val = eval(param) or self.params[param]
            self.params[param] = val
        self.eta = self.params['eta']
    
    def __call__(self, grad, theta):
        self.update(theta, grad)
        return self.theta

    def next(self, it:int=None):
        self.idx = self.idx+1 or it
        self.schedule(self.idx) # not sure this is necessary

    def set_learning_schedule(self, tau:int, eta_0:float=None, eta_tau:float=None):
        eta0 = eta_0 or self.eta
        etatau = eta_tau or 0.01*eta0
        self.schedule = lambda k: self.standard_learning_schedule(k, eta0, etatau, tau)
        
    def standard_learning_schedule(self, k, eta_0, eta_tau, tau):
        kt = k/tau
        if kt <= 1:
            eta = (1-kt) * eta_0 + kt*eta_tau
        else:
            eta = eta_tau
        self.set_learning_rate(eta)
    
    def set_learning_rate(self, eta:float):
        self.eta = eta

    def update(self, theta, grad):
        # standard:
        self.v = self.gamma*self.v - self.eta*grad
        self.theta = theta + self.v


class noneRULEadaptive(noneRULE):
    def __init__(self, theta0, eta:float=0, rho1:float=0, rho2:float=0, epsilon:float=1e-7):
        super().__init__(theta0, eta, epsilon=epsilon, rho1=rho1, rho2=rho2)
        self.r = np.zeros_like(self.theta)
        self.epsilon = epsilon # numerical stability

    
    def set_params(self, eta: float = None, rho1: float = None, rho2: float = None, epsilon: float = None):
        super().set_params(eta, 0, rho1, rho2, epsilon)
        self.epsilon = self.params['epsilon']

class rule_ClassicalSGD(noneRULE):
    def __init__(self, theta0, eta0:float=0.01):
        super().__init__(theta0, eta0, gamma=0)
        self.params = {key:self.params[key] for key in ['eta']}
        
    def set_params(self, eta:float=None):
        super().set_params(eta, 0)
        self.gamma = 0

class rule_MomentumSGD(noneRULE):
    def __init__(self, theta0, eta0:float=0.01, gamma:float=0.9):
        super().__init__(theta0, eta0, gamma=gamma)
        self.params = {key:self.params[key] for key in ['eta', 'gamma']}

    def set_params(self, eta0:float=None, gamma:float=None):
        super().set_params(eta0, gamma)
        self.gamma = self.params['gamma']
class rule_AdaGrad(noneRULEadaptive):
    def __init__(self, theta0, eta:float=0.01, epsilon:float=1e-7):
        super().__init__(theta0, eta, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'epsilon']}
    
    def update(self, theta, grad):
        self.r = self.r + grad**2
        self.v = - self.eta*(self.epsilon+np.sqrt(self.r))**(-1) * grad # check if this is element-wise
        self.theta = theta + self.v 

    def set_params(self, eta: float = None, epsilon: float = None):
        super().set_params(eta, None, None, epsilon)

class rule_RMSProp(noneRULEadaptive):
    def __init__(self, theta0, eta:float=0.01, rho:float=0.1, epsilon:float=1e-7):
        super().__init__(theta0, eta, rho2=rho, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho2', 'epsilon']}

    def update(self, theta, grad):
        self.r = self.rho*self.r + (1-self.rho)*grad**2
        self.v = - self.eta / (np.sqrt(self.epsilon + self.r[:]))*grad # is this element-wise? should be...
        self.theta = theta + self.v

    def set_params(self, eta: float = None, rho:float=None,  epsilon:float=None):
        super().set_params(eta, None, rho, epsilon)
        self.rho = self.params['rho2']
class rule_Adam(noneRULEadaptive):
    def __init__(self, theta0, eta:float=0.001, rho1:float=0.9, rho2:float=0.999, epsilon:float=1e-8):
        super().__init__(theta0, eta, rho1=rho1, rho2=rho2, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho1', 'rho2', 'epsilon']}
        self.s = np.zeros_like(self.theta)
            
    def update(self, theta, grad, k=None):
        k = k or self.idx
        self.s = self.rho1*self.s + (1-self.rho1)*grad
        self.r = self.rho2*self.r + (1-self.rho2)*grad**2
        s_hat = self.s * (1-self.rho1**k)**(-1)
        r_hat = self.r * (1-self.rho2**k)**(-1)
        self.v = - self.eta*s_hat * (np.sqrt(r_hat) + self.epsilon)**(-1) # is this element-wise? should be...
        self.theta = theta + self.v

    def set_params(self, eta:float=None, rho1:float=None, rho2:float=None, epsilon:float=None):
        super().set_params(eta, rho1, rho2, epsilon)
        self.rho1 = self.params['rho1']
        self.rho2 = self.params['rho2']











class noneGradientDescent:
    def __init__(self, X, y, eta:float, theta, no_epochs:int, tolerance:float):
        self.X = X
        self.y = y
        self.eta = eta  #Learning rate
        if isinstance(theta, (float, int)):
            self.theta = np.ones(1)*theta
        else:
            self.theta = theta
        self.no_epochs = no_epochs
        self.tol = tolerance
        self.v = np.zeros_like(theta)

        self.indices = np.arange(len(y))

    @classmethod
    def simple_initialise(cls, eta:float, tolerance:float=1e-7):
        return cls(np.zeros((10,1)), np.zeros(10), eta, 100, tolerance)

    def simple_update(self, gradient, theta):
        theta_new = self.update_rule(gradient, theta)
        return theta_new

    def epoch_calculation(self, grad):
        self.per_epoch(grad)
        self.update_rule.next()

    
    def per_batch(self, grad):
        self.theta = self.update(grad)
        # if test that breaks loop if self.v is less than self.tol


    def set_params(self, **params):
        # set params in update rule
        self.update_rule.set_params(**params)

    def add_nag(self):
        self.grad = lambda k=0: self.compute_gradient(self.theta + self.update_rule.gamma+self.v)
        # try-except?

    def set_update_rule(self, scheme:str, params:dict={}, NAG=False):
        rule = scheme.strip().lower()
        momentum = False
        self.adaptive = False
        if rule in ['plain', 'none', 'manual']:
            # params : eta0
            update_rule = rule_ClassicalSGD(self.theta, **params)
        elif rule in ['mom', 'momentum']:
            # params: eta0, gamma
            update_rule = rule_MomentumSGD(self.theta, **params)
            momentum = True
        elif rule in ['ada', 'adagrad']:
            # params: eta, epsilon
            update_rule = rule_AdaGrad(self.theta, **params)
            self.adaptive = True
        elif rule in ['rms', 'rmsprop']:
            # params: eta, rho, epsilon
            update_rule = rule_RMSProp(self.theta, **params)
            self.adaptive = True
        elif rule in ['adam']:
            # params: eta, rho1, rho2, epsilon
            update_rule = rule_Adam(self.theta, **params)
            self.adaptive = True
        else: 
            raise ValueError(f"The library does not have functionalities for {rule} optimiser.")
        
        update_rule.set_params()
        self.update_rule = update_rule
        self.update = lambda A: update_rule(A, self.theta)
        if momentum and NAG:
            self.add_nag()

    def apply_learning_schedule(self, eta_0:float=None, eta_tau:float=None, tau:int=None):
        self.update_rule.set_learning_schedule(tau or self.no_epochs*200, eta_0 or self.eta, eta_tau)




class GD(noneGradientDescent):
    def __init__(self, X, y, eta:float, theta, no_epochs:int, tolerance=1e-6):
        super().__init__(X, y, eta, theta, no_epochs, tolerance)
    
    # def compute_gradient(self, theta_k):
    #     self.A = agrad(self.LF)(theta_k)

    def per_epoch(self, grad):
        self.per_batch(grad(slice(0,len(self.X))))




class SGD(noneGradientDescent):
    def __init__(self, X, y, eta:float, theta, no_epochs:int, tolerance=1e-6):
        super().__init__(X, y, eta, theta, no_epochs, tolerance)

    # def compute_gradient(self, theta_k):
    #     self.A = egrad(self.LF)(theta_k)
    #     return self.A

        
    def per_epoch(self, grad):
        m = 5
        M = len(self.X)/m
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        batches = np.array_split(indices, m)

        for batch in batches:
            #print(batch, grad(batch))
            self.per_batch(grad(batch))




if __name__=="__main__":
    x = np.linspace(-10,10, 1000)
    f = lambda x, theta: theta[0]*x + theta[1]*x**2
    y = f(x, (0.1, 0.9)) 

    def LF(idx):
        return lambda theta: np.sum((f(x[idx], theta)- y[idx])**2)  /len(idx)


    lf = LF([9, 10, 100])
    print(lf((0.1, 0.9)))
    X = np.zeros((len(x), 2))
    X[:,0] = x
    X[:,1] = x**2
    sgd = SGD(X, y, 0.05, np.array([0.16, 0.8]), 500)
    sgd2 = SGD.simple_initialise(0.05)
    sgd2.set_update_rule('momentum')
    sgd2.set_params(gamma=0.1)

    sgd.set_update_rule('plain')
    sgd.apply_learning_schedule(tau=len(x)*100)
    
    def grad(theta_k):
        return lambda idx: egrad(LF(idx))(theta_k)


    # for _ in range(100):
        # theta = sgd.theta
        # sgd.epoch_calculation(grad(theta))
