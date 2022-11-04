import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad 
import matplotlib.pyplot as plt
from IPython import embed

class GradientDescent:
    def __init__(self, LF, eta, theta, nr_epochs):
        self.LF = LF  #Loss function as function of theta
        self.eta = eta  #Learning rate
        self.theta = theta
        self.nr_epochs = nr_epochs





class GD(GradientDescent):
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


class noneAlgo:
    eps0 = 1e-7 # for numerical stability
    params0 = {'eta':0.0, 'gamma':0.0, 'epsilon':eps0, 'rho':0.0, 'rho1':0.0, 'rho2':0.0}
    def __init__(self, theta0):
        self.theta = theta0
        self.idx = 0
        self.epsilon = 1e-7
    
    def __call__(self, theta, grad, params=params0):
        return self.update(theta, grad, **params)

    def next(self, it=None):
        self.idx = self.idx+1 or it

    def learning_schedule(self):
        pass


class PlainRule(noneAlgo): # lack of better names... 
    def __init__(self, theta0):
        super().__init__(theta0)
    
    def update(self, theta, grad, eta, gamma=0.0, epsilon=1e-7, rho=0.0, rho1=0.0, rho2=0.0):
        self.theta = theta - eta * grad

class MomentumPerturbation(noneAlgo):
    def __init__(self, theta0):
        super().__init__(theta0)
        self.v = np.zeros_like(theta0)

    def update(self, theta, grad, eta, gamma, epsilon=1e-7, rho=0.0, rho1=0.0, rho2=0.0):
        self.v = gamma*self.v + eta*grad
        self.theta = theta - self.v



class AdaGrad(noneAlgo):
    def __init__(self, theta0):
        super().__init__(theta0)
        self.r = np.zeros_like(theta0)
    
    def update(self, theta, grad, eta, gamma=0.0, epsilon=1e-6, rho=0.0, rho1=0.0, rho2=0.0):
        self.r = self.r + grad**2
        self.theta = theta - eta*(epsilon+np.sqrt(self.r))**(-1) * grad # check if this is element-wise

class RMSProp(noneAlgo):
    def __init__(self, theta0):
        super().__init__(theta0)
        self.r = np.zeros_like(theta0)
        self.epsilon = 1e-6
    
    def update(self, theta, grad, eta, gamma=0.0, epsilon=1e-6, rho=0.0, rho1=0.0, rho2=0.0):
        self.r = rho*self.r + (1-rho)*grad**2
        self.theta = theta - eta / (np.sqrt(epsilon + self.r[:]))*grad # is this element-wise? should be...

class Adam(noneAlgo):
    def __init__(self, theta0):
        super().__init__(theta0)
        self.s = np.zeros_like(theta0)
        self.r = np.zeros_like(theta0)
        self.epsilon = 1e-8
    
    def update(self, theta, grad, eta, gamma=0.0, epsilon=1e-8, rho=0.0, rho1=0.9, rho2=0.999):
        self.s = rho1*self.s + (1-rho1)*grad
        self.r = rho2*self.r + (1-rho2)*grad**2
        s_hat = self.s * (1-rho1**self.idx)**(-1)
        r_hat = self.r * (1-rho2**self.idx)**(-1)
        self.theta = theta - eta*s_hat * (np.sqrt(r_hat) + epsilon)**(-1) # is this element-wise? should be...



def parseOptimiser(theta0, optimiser='none'):
    opt = optimiser.strip().lower()

    if opt in ['plain', 'none', 'manual']:
        return PlainRule(theta0)
    elif opt in ['mom', 'momentum']:
        return MomentumPerturbation(theta0)
    elif opt in ['ada', 'adagrad']:
        return AdaGrad(theta0)
    elif opt in ['rms', 'rmsprop']:
        return RMSProp(theta0)
    elif opt in ['adam']:
        return Adam(theta0)
    else: 
        raise ValueError(f"The library does not have functionalities for {opt} optimiser.")
    





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