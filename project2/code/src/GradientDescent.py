import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad as agrad
import matplotlib.pyplot as plt
from IPython import embed
import sys




class noneRULE:
    def __init__(self, theta0, eta:float=0.0, gamma:float=0.0, rho1:float=0.0, rho2:float=0.0, epsilon:float=0.0):
        self.theta = theta0
        self.eta = eta
        self.params = {'eta':eta, 'gamma':gamma, 'rho1':rho1, 'rho2':rho2, 'epsilon':epsilon}
        self.idx = 0
        self.v = np.zeros_like(self.theta)
        self.schedule = lambda k: self.eta

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
    def __init__(self, theta0, eta:float=0.01, rho:float=0.9, epsilon:float=1e-7):
        super().__init__(theta0, eta, rho2=rho, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho2', 'epsilon']}

    def update(self, theta, grad):
        self.r = self.rho*self.r + (1-self.rho)*grad**2
        self.v = - self.eta / (np.sqrt(self.epsilon + self.r))*grad # is this element-wise? should be...
        self.theta = theta + self.v

    def set_params(self, eta:float=None, rho:float=None,  epsilon:float=None):
        super().set_params(eta, None, rho, epsilon)
        self.rho = self.params['rho2']

class rule_Adam(noneRULEadaptive):
    def __init__(self, theta0, eta:float=0.001, rho1:float=0.9, rho2:float=0.999, epsilon:float=1e-8):
        super().__init__(theta0, eta, rho1=rho1, rho2=rho2, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho1', 'rho2', 'epsilon']}
        self.s = np.zeros_like(self.theta)
            
    def update(self, theta, grad, k=None):
        k = k or self.idx; k+=1
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
    def __init__(self, X, y, eta:float, theta0, no_epochs:int, tolerance:float):
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

    @classmethod
    def simple_initialise(cls, eta:float, tolerance:float=1e-7):
        return cls(np.zeros((10,1)), np.zeros(10), eta, 0, 100, tolerance)

    def simple_update(self, gradient, theta):
        theta_new = self.update_rule(gradient, theta)
        return theta_new

    
    def per_batch(self, grad):
        self.theta = self.update(grad)
        self.update_rule.next()
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
        

        update_rule.set_params(self.eta)
        self.update_rule = update_rule
        self.update = lambda A: update_rule(A, self.theta)
        if momentum and NAG:
            self.add_nag()

    def apply_learning_schedule(self, eta_0:float=None, eta_tau:float=None, tau:int=None):
        self.update_rule.set_learning_schedule(tau or self.n_obs*200, eta_0 or self.eta, eta_tau)

    def regression_setting(self, f):
        # Use MSE loss func

        def gradient(x, y, theta):
            lf = lambda theta: np.sum((f(x, theta) - y)**2)/len(x)/2
            return egrad(lf)(theta)

        self.grad = gradient

    def classification_setting(self, f):
        def gradient(x, y, theta):
            # FIXME
            tol = 1e-12
            I = lambda theta: np.mean(np.where(np.abs(f(x,theta)-y)<tol, 1, 0), axis=0 )
            lf = lambda theta: 1 - I(theta)
            return egrad(lf)(theta)
        self.grad = gradient
        
    
    def __call__(self):
        for _ in range(self.no_epochs):
            self.per_epoch(self.grad)
        return self.theta

    def set_loss_function(self, loss_function):
        self.grad = lambda x, y, theta: egrad(lambda theta: loss_function(x, y, theta))(theta)





class GD(noneGradientDescent):
    def __init__(self, X, y, eta:float, theta0, no_epochs:int=500, tolerance=1e-6):
        super().__init__(X, y, eta, theta0, no_epochs, tolerance)
    

    def per_epoch(self, grad):
        self.per_batch(grad(self.X, self.y, self.theta))




class SGD(noneGradientDescent):
    def __init__(self, X, y, eta:float, theta0, no_epochs:int=500, no_minibatches:int=5, tolerance=1e-6):
        super().__init__(X, y, eta, theta0, no_epochs, tolerance)
        self.m = int(no_minibatches)
        assert self.m <= self.n_obs


        
    def per_epoch(self, grad):
        indices = np.arange(self.n_obs)
        np.random.shuffle(indices)
        batches = np.array_split(indices, self.m)

        for k in range(self.m):
            x_batch = self.X[batches[k]]
            y_batch = self.y[batches[k]]
            self.per_batch(grad(x_batch, y_batch, self.theta))




if __name__=="__main__":
    x = np.linspace(-1,1, 1000)
    f = lambda x, theta: theta[0]*x + theta[1]*x**2 + theta[2]*x**3
    f = lambda x, theta: theta[0]*x *np.cos(theta[1]*x) + theta[2]*x**2
    y = f(x, (2, 1.7, -0.4)) + np.random.randn(len(x))*0.05

    X = np.zeros((len(x),1))
    X[:,0] = x

    NN = 100
    theta0 = [0.23, 0.31, 1]
    eta0 = 0.2
    lmbda = 0.1
    LF_R = lambda x, y, theta: (np.sum((f(x, theta) - y)**2)+ lmbda * np.sum(theta**2) )/ (2*len(y))
    
    sgd1 = SGD.simple_initialise(0.1)
    

    sgd = SGD(x, y, eta0, theta0, NN)
    sgd2 = SGD(x, y, eta0, theta0, NN)
    sgd.set_update_rule('plain')

    sgd2.set_update_rule('momentum')
    sgd2.set_params(gamma=0.5)
    #sgd.apply_learning_schedule(tau=len(x)*100)
    sgd.set_loss_function(LF_R)
    sgd2.set_loss_function(LF_R)

    theta = sgd()
    yhat = f(x, theta)

    theta2 = sgd2()
    yhat2 = f(x, theta2)

    gd = GD(x, y, eta0, theta0, NN)
    gd.set_update_rule('plain')
    gd.set_loss_function(LF_R)

    sgd3 = SGD(x, y, eta0, theta0, NN)
    sgd3.set_update_rule('AdaGrad')
    sgd3.set_loss_function(LF_R)

    theta3 = sgd3()
    yhat3 = f(x, theta3)

    sgd4 = SGD(x, y, eta0, theta0, NN)
    sgd4.set_update_rule('RMSProp')
    sgd4.set_loss_function(LF_R)

    theta4 = sgd4()
    yhat4 = f(x, theta4)

    sgd5 = SGD(x, y, eta0, theta0, NN)
    sgd5.set_update_rule('Adam')
    sgd5.set_loss_function(LF_R)

    theta5 = sgd5()
    yhat5 = f(x, theta5)



    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', c='k', alpha=0.5)
    ax.plot(x, yhat, '-', label='plain SGD')
    ax.plot(x, yhat2, '--', label='momentum SGD')
    ax.plot(x, yhat3, ':', label='AdaGrad')
    ax.plot(x, yhat4, '-', label='RMSProp')
    ax.plot(x, yhat5, '--', label='Adam')
    ax.legend()
    plt.show()
    

    # for _ in range(100):
        # theta = sgd.theta
        # sgd.epoch_calculation(grad(theta))
