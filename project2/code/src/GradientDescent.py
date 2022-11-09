try:
    from utils import *
except ModuleNotFoundError:
    from src.utils import *

import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt






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

    def get_diff(self):
        return self.v

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
        self.current_epoch = 0

    @classmethod
    def simple_initialise(cls, eta:float, tolerance:float=1e-7):
        return cls(np.zeros((10,1)), np.zeros(10), eta, 0, 100, tolerance)

    def simple_update(self, gradient, theta):
        theta_new = self.update_rule(gradient, theta)
        return theta_new

    def per_batch(self, grad):
        # if test that breaks loop if self.v is less than self.tol
        self.theta = self.update(grad)
        self.update_rule.next()
       
    def set_params(self, **params):
        # set params in update rule
        self.update_rule.set_params(**params)

    def add_nag(self):
        # try-except?
        self.grad = lambda k=0: self.compute_gradient(self.theta + self.update_rule.gamma+self.v)

    def set_update_rule(self, scheme:str, params:dict={}, NAG=False):
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
        self.update_rule.set_learning_schedule(tau or self.n_obs*200, eta_0 or self.eta, eta_tau)

    def regression_setting(self, regularisation:float=0):
        self.lmbda = regularisation

        def gradient(x, y, theta):
            lf = lambda theta: np.mean((x@theta - y)**2)/2 + self.lmbda*np.mean(theta**2)
            return egrad(lf)(theta)

        self.grad = gradient

    def classification_setting(self, regularisation:float=0):
        def gradient(x, y, theta):
            # FIXME
            def lf(theta):
                xt = x@theta # is this ok? What about intercept??
                return - np.mean( y*(xt )  - np.log10(1 +np.exp(xt) )) # cross entropy
            return egrad(lf)(theta)
        self.grad = gradient
        
    def __call__(self, no_epochs=None, deregestration_every=None):
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

    def set_loss_function(self, loss_function):
        self.grad = lambda x, y, theta: egrad(lambda theta: loss_function(x, y, theta))(theta)

    def mean_squared_error(self, X:ndarray=None, y:ndarray=None, theta:ndarray=None):
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

    def accuracy_score(self):
        # FIXME
        X = self.X
        y = self.y
        theta = self.theta
        tol = 1e-12
        xt = x@theta
        I = np.where(np.abs(xt-y)<tol, 1, 0)
        return np.mean(I)

    def get_params(self, terminal_print=True):
        if terminal_print:
            for param in self.update_rule.params:
                val = self.update_rule.params[param]
                print(f'{param:10s} = {val:10.2e}')

        return self.update_rule.params

        



class GD(noneGradientDescent):
    def __init__(self, X:ndarray, y:ndarray, eta:float, theta0:ndarray, no_epochs:int=500, tolerance=1e-6):
        super().__init__(X, y, eta, theta0, no_epochs, tolerance)
    
    def __str__(self):
        return self.optimiser + " GD"

    def per_epoch(self, grad):
        self.per_batch(grad(self.X, self.y, self.theta))


class SGD(noneGradientDescent):
    def __init__(self, X, y, eta:float, theta0, no_epochs:int=500, no_minibatches:int=5, tolerance=1e-6):
        super().__init__(X, y, eta, theta0, no_epochs, tolerance)
        self.m = int(no_minibatches)
        assert self.m <= self.n_obs

    def __str__(self):
        if self.adaptive:
            return self.optimiser
        else:
            return self.optimiser + " SGD"


        
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
    #f = lambda x, theta: theta[0]*x *np.cos(theta[1]*x) + theta[2]*x**2
    y = f(x, (2, 1.7, -0.4)) + np.random.randn(len(x))*0.05

    X = np.zeros((len(x),3))
    X[:,0] = x
    X[:,1] = x**2
    X[:,2] = x**3

    y = X@np.array([2,1.7,-0.4]) + np.random.randn(len(x))*0.05


    NN = 100
    theta0 = [0.23, 0.31, 1]


    #print(X@theta0)
    eta0 = 0.2
    lmbda = 0.1
    LF_R = lambda x, y, theta: (np.sum((f(x, theta) - y)**2)+ lmbda * np.sum(theta**2) )/ (2*len(y))
 

    sgd = SGD(X, y, eta0, theta0, NN)
    # #sgd2 = SGD(x, y, eta0, theta0, NN)
    sgd.set_update_rule('plain')

    # # sgd2.set_update_rule('momentum')
    # # sgd2.set_params(gamma=0.5)
    # #sgd.apply_learning_schedule(tau=len(x)*100)
    sgd.regression_setting()
    # # sgd2.set_loss_function(LF_R)

    theta = sgd()
    yhat = X@theta

    # theta2 = sgd2()
    # yhat2 = f(x, theta2)

    # gd = GD(x, y, eta0, theta0, NN)
    # gd.set_update_rule('plain')
    # gd.set_loss_function(LF_R)

    # sgd3 = SGD(x, y, eta0, theta0, NN)
    # sgd3.set_update_rule('AdaGrad')
    # sgd3.set_loss_function(LF_R)

    # theta3 = sgd3()
    # yhat3 = f(x, theta3)

    # sgd4 = SGD(x, y, eta0, theta0, NN)
    # sgd4.set_update_rule('RMSProp')
    # sgd4.set_loss_function(LF_R)

    # theta4 = sgd4()
    # yhat4 = f(x, theta4)

    # sgd5 = SGD(x, y, eta0, theta0, NN)
    # sgd5.set_update_rule('Adam')
    # sgd5.set_loss_function(LF_R)

    # theta5 = sgd5()
    # yhat5 = f(x, theta5)



    # fig, ax = plt.subplots()
    # ax.plot(x, y, 'o', c='k', alpha=0.5)
    # ax.plot(x, yhat, '-', label='plain SGD')
    # # # ax.plot(x, yhat2, '--', label='momentum SGD')
    # # # ax.plot(x, yhat3, ':', label='AdaGrad')
    # # # ax.plot(x, yhat4, '-', label='RMSProp')
    # # # ax.plot(x, yhat5, '--', label='Adam')
    # ax.legend()
    # plt.show()
    

    space = np.linspace(-1,1,20)
    xx, yy = np.meshgrid(space,space)

    def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    zz = FrankeFunction(xx, yy)
    zzr = zz.ravel()
    zzr = (zzr-np.mean(zzr))/np.std(zzr)

    dd = 20
    FrankeX = np.zeros((len(zzr),dd))
    FrankeX[:,0] = xx.ravel()
    FrankeX[:,1] = yy.ravel()
    FrankeY = zzr#[:,np.newaxis]

    FrankeX[:,2] = xx.ravel()**2
    FrankeX[:,3] = xx.ravel()*yy.ravel()
    FrankeX[:,4] = yy.ravel()**2

    FrankeX[:,5] = xx.ravel()**3
    FrankeX[:,6] = xx.ravel()**2*yy.ravel()
    FrankeX[:,7] = xx.ravel()*yy.ravel()**2
    FrankeX[:,8] = yy.ravel()**3

    FrankeX[:,9] = xx.ravel()**4
    FrankeX[:,10] = xx.ravel()**3*yy.ravel()
    FrankeX[:,11] = xx.ravel()**2*yy.ravel()**2
    FrankeX[:,12] = xx.ravel()*yy.ravel()**3
    FrankeX[:,13] = yy.ravel()**4

    FrankeX[:,14] = xx.ravel()**5
    FrankeX[:,15] = xx.ravel()**4*yy.ravel()
    FrankeX[:,16] = xx.ravel()**3*yy.ravel()**2
    FrankeX[:,17] = xx.ravel()**2*yy.ravel()**3
    FrankeX[:,18] = xx.ravel()*yy.ravel()**4
    FrankeX[:,19] = yy.ravel()**5

    for i in range(dd):
        FrankeX[:,i] = (FrankeX[:,i]-np.mean(FrankeX[:,i]))/np.std(FrankeX[:,i])


    Sgd_F = SGD(FrankeX, FrankeY, 0.08, np.random.randn(dd), no_epochs=2000, no_minibatches=30)
    Sgd_F.set_update_rule("rms")
    #Sgd_F.set_params(gamma=0.4)
    Sgd_F.regression_setting()
    beta = Sgd_F()
    print(Sgd_F.mean_sqared_error())

    FrankeY_hat = FrankeX@beta
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    ax.plot_trisurf(FrankeX[:,0], FrankeX[:,1], FrankeY_hat, cmap='coolwarm')
    ax.scatter(FrankeX[:,0], FrankeX[:,1], FrankeY, color='k')

    plt.show()

