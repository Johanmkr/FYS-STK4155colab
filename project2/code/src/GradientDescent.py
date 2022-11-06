import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad 
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




class noneSGD:
    def __init__(self, theta0, eta:float=0.0, gamma:float=0.0, rho1:float=0.0, rho2:float=0.0, epsilon:float=0.0):
        self.theta = theta0
        #self.eta = eta
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
            eval(f'self.{param} = {val}')
    
    def __call__(self, grad, theta=None):
        theta = theta or self.theta
        return self.update(theta, grad)

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


class noneSGDadaptive(noneSGD):
    def __init__(self, theta0, eta:float=0, rho1:float=0, rho2:float=0, epsilon:float=1e-7):
        super().__init__(theta0, eta, epsilon=epsilon, rho1=rho1, rho2=rho2)
        self.r = np.zeros_like(self.theta)
        self.epsilon = epsilon # numerical stability




class rule_ClassicalSGD(noneSGD):
    def __init__(self, theta0, eta0:float):
        super().__init__(theta0, eta0, gamma=0)
        self.params = {key:self.params[key] for key in ['eta']}
        

class rule_MomentumSGD(noneSGD):
    def __init__(self, theta0, eta0:float, gamma:float=0.9):
        super().__init__(theta0, eta0, gamma=gamma)
        self.params = {key:self.params[key] for key in ['eta', 'gamma']}


class rule_AdaGrad(noneSGDadaptive):
    def __init__(self, theta0, eta:float, epsilon:float=1e-7):
        super().__init__(theta0, eta, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'epsilon']}
    
    def update(self, theta, grad):
        self.r = self.r + grad**2
        self.theta = theta - self.eta*(self.epsilon+np.sqrt(self.r))**(-1) * grad # check if this is element-wise

class rule_RMSProp(noneSGDadaptive):
    def __init__(self, theta0, eta:float, rho:float, epsilon:float=1e-7):
        super().__init__(theta0, eta, rho2=rho, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho2', 'epsilon']}

    def update(self, theta, grad):
        self.r = self.rho2*self.r + (1-self.rho2)*grad**2
        self.theta = theta - self.eta / (np.sqrt(self.epsilon + self.r[:]))*grad # is this element-wise? should be...

class rule_Adam(noneSGDadaptive):
    def __init__(self, theta0, eta:float=0.001, rho1:float=0.9, rho2:float=0.999, epsilon:float=1e-8):
        super().__init__(theta0, eta, rho1=rho1, rho2=rho2, epsilon=epsilon)
        self.params = {key:self.params[key] for key in ['eta', 'rho1', 'rho2', 'epsilon']}
        self.s = np.zeros_like(self.theta)
        
    
    def update(self, theta, grad):
        self.s = self.rho1*self.s + (1-self.rho1)*grad
        self.r = self.rho2*self.r + (1-self.rho2)*grad**2
        s_hat = self.s * (1-self.rho1**self.idx)**(-1)
        r_hat = self.r * (1-self.rho2**self.idx)**(-1)
        self.theta = theta - self.eta*s_hat * (np.sqrt(r_hat) + self.epsilon)**(-1) # is this element-wise? should be...





class noneGradientDescent:
    def __init__(self, loss_function, eta:float, theta, no_epochs:int):
        self.LF = loss_function  #Loss function as function of theta
        self.eta = eta  #Learning rate
        self.theta = theta
        self.no_epochs = no_epochs

        self.grad = lambda k=0: self.compute_gradient(self.theta) # check if this works properly
        
        self.momentum = False


    def set_params(self, eta:float=None, gamma:float=0.0, rho:float=0.0, rho1:float=0.0, rho2:float=0.0, epsilon:float=0.0):
        # set params in update rule
        self.eta = eta or self.eta
        if gamma > 0.0:
            self.momentum = True
            self.gamma = gamma

        # idk

        self.update_rule.set_params(eta, gamma, rho, rho1, rho2, epsilon)

    def add_nag(self, gamma=None):
        assert self.momentum
        self.grad = lambda k=0: self.compute_gradient(self.theta + self.update_rule.params['gamma']+self.v)

    def set_update_rule(self, scheme:str, params:dict):
        rule = scheme.strip().lower()

        if rule in ['plain', 'none', 'manual']:
            # params : eta
            update_rule = rule_ClassicalSGD(self.theta, **params)
        elif rule in ['mom', 'momentum']:
            # params: eta, gamma
            update_rule = rule_MomentumSGD(self.theta, **params)
        elif rule in ['ada', 'adagrad']:
            # params: eta, epsilon
            update_rule = rule_AdaGrad(self.theta, **params)
        elif rule in ['rms', 'rmsprop']:
            # params: eta, rho2, epsilon
            update_rule = rule_RMSProp(self.theta, **params)
        elif rule in ['adam']:
            # params: eta, ro1, rho2, epsilon
            update_rule = rule_Adam(self.theta, **params)
        else: 
            raise ValueError(f"The library does not have functionalities for {rule} optimiser.")
        
        self.update_rule = lambda k=0: update_rule(self.theta, self.grad())



class GD(noneGradientDescent):
    def __init__(self, loss_function, eta:float, theta, no_epochs:int):
        super().__init__(loss_function, eta, theta, no_epochs)
    
    def compute_gradient(self, theta_k):
        self.A = grad(self.LF)(theta_k)


class SGD(noneGradientDescent):
    def __init__(self, loss_function, eta:float, theta, no_epochs:int):
        super().__init__(loss_function, eta, theta, no_epochs)

    def compute_gradient(self, theta_k):
        self.A = egrad(self.LF)(theta_k)
    



class GradientDescent:
    def __init__(self, loss_function, eta:float, theta, no_epochs:int):
        self.LF = loss_function  #Loss function as function of theta
        self.eta = eta  #Learning rate
        self.theta = theta
        self.no_epochs = no_epochs

        self.grad = lambda k=0: self.compute_gradient(self.theta) # check if this works properly
        


class GD2(GradientDescent):
    def __init__(self, LF, eta, theta, nr_epochs):
        GradientDescent.__init__(self, LF=LF, eta=eta, theta=theta, nr_epochs=nr_epochs)


    def find_A(self, LFargument):
        # grd = grad(self.LF)
        # self.A = grd(LFargument)
        self.A = grad(self.LF)(LFargument)

    def find_eta(self):
        self.eta = self.eta

    def find_v(self):
        self.v = -self.eta * self.A

    #does not update eta yet
    def __call__(self, theta):
        self.find_A(theta)
        # self.find_eta() # If tunable learning rate
        self.find_v()
        theta = theta + self.v
        return theta #could consider making theta a class variabel

class MGD(GD):
    def __init__(self, LF, eta=0.01, gamma=0.1, NAG=False):
        GD.__init__(self, LF=LF, eta=eta)
        self.gamma = gamma
        self.v_prev = 0
        self.NAG = NAG

    def find_v(self):
        self.v = self.gamma*self.v_prev - self.eta*self.A 

    def __call__(self, theta):
        if self.NAG:
            self.find_A(theta+self.gamma*self.v_prev)
        else:
            self.find_A(theta)
        self.find_v()
        theta = theta + self.v
        self.v_prev = self.v
        return theta

    
class plain_SGD(GradientDescent):
    def __init__(self, X, LF, eta=0.01, gamma=0.9, nr_epochs=1000, nr_minibatches=10):
        GradientDescent.__init__(self, LF=LF, eta=eta)
        self.gamma = gamma
        self.nr_epochs = nr_epochs
        self.m = nr_minibatches
        self.X = X
        self.M = int(len(X)/self.m)
        self.tau = 200* len(self.X)
        self.eta0 = eta
        self.eta_tau = 0.01 * self.eta0
        # self.v_prev = 0

    def learning_schedule(self, k):
        if k>self.tau:
            self.eta = self.eta_tau
        else:
            self.eta = (1-k/self.tau)*self.eta0 + k/self.tau*self.eta_tau
    
    def find_A(self, LFargument):
        self.A = egrad(self.LF)(LFargument)

    def find_v(self):
        self.v = - self.eta*self.A

    def __call__(self, theta):
        #   This is a wack way of doing this, needs improvement soon
        k = 1
        # embed()
        for epoch in range(self.nr_epochs):
            indecies = np.arange(self.M*self.m)
            np.random.shuffle(indecies)
            batches = np.array_split(indecies, self.m)
            # for i in range(self.m):
            for batch in batches:
                xi = self.X[batch]
                # random_indecies = np.random.randint(0, high=self.M*self.m, size=self.M)
                # random_indecies = np.random.choice(self.M*self.m, size=self.M, replace=False)
                # xi = self.X[random_indecies]
                self.find_A(xi)
                self.learning_schedule(k)
                self.find_v()
                # embed()
                theta = theta + self.v
                # self.v_prev = self.v
                k += 1
                # print(self.v)
            # plt.scatter(theta, f(theta), marker="1")
        # embed()
        
        theta_out = theta[np.argmin(self.LF(theta))]
        return theta_out







def parseOptimiser(theta0, optimiser:str, params:dict):
    rule = optimiser.strip().lower()

    if rule in ['plain', 'none', 'manual']:
        return rule_ClassicalSGD(theta0, **params)
    elif rule in ['mom', 'momentum']:
        return rule_MomentumSGD(theta0, **params)
    elif rule in ['ada', 'adagrad']:
        return rule_AdaGrad(theta0, **params)
    elif rule in ['rms', 'rmsprop']:
        return rule_RMSProp(theta0, **params)
    elif rule in ['adam']:
        return rule_Adam(theta0, **params)
    else: 
        raise ValueError(f"The library does not have functionalities for {rule} optimiser.")
    





if __name__=="__main__":
    x = np.linspace(-10,10, 1000)
    # def f(x):
    #     return 0.1 * x * np.cos(x)
    f = lambda x: 0.1 *x * np.cos(x)
    # f = lambda x: x**2
    y = f(x)
    # fx = 0.1*x*np.cos(x)
    # Lf = lambda y, fx: (y-fx)**2
    plt.plot(x,f(x))

    # gd = MGD(f, eta=0.01, gamma=0.9, NAG=True)
    # gd = GD(f, eta=0.01)
    gd = plain_SGD(X=x, LF=f)

    x0 = 0
    plt.scatter(x0, f(x0), marker="x")
    # MAXiter = 20
    # Niter = 0
    # for _ in range(MAXiter):
    #     xnew = gd(x0)
    #     if abs(xnew-np.min(f(x))) < 1e-1:
    #         break
    #     else:
    #       x0 = xnew
    #       plt.scatter(x0,  f(x0), marker="1")
    #     # x0 = xnew
    #     Niter += 1
    # print(Niter)
    x0 = gd(x0)
    plt.scatter(x0, f(x0), marker="o")
    plt.show()
    # print(x0)



    # def f(x):
    #     return np.sin(x)
    # fder = egrad(f)
    # plt.plot(x,f(x), label="f")
    # plt.plot(x, fder(x), label="fder")
    # plt.plot(x, np.cos(x), ls="--")
    # plt.show()