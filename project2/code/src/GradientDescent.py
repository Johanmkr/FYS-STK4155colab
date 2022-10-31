import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad 
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, LF, eta):
        self.LF = LF  #Loss function as function of theta
        self.eta = eta  #Learning rate

    
class GD(GradientDescent):
    def __init__(self, LF, eta=0.01):
        GradientDescent.__init__(self, LF=LF, eta=eta)

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
    def __init__(self, LF, eta=0.01, gamma=0.9, nr_epochs=10, nr_minibatches=5, t0=5, t1=50):
        GradientDescent.__init__(LF, eta)
        self.gamma = gamma
        self.nr_epochs = nr_epochs
        self.m = nr_minibatches 
        self.t0 = t0
        self.t1 = t1
    
    def find_A



if __name__=="__main__":
    x = np.linspace(-10,10, 1000)
    # def f(x):
    #     return 0.1 * x * np.cos(x)
    f = lambda x: 0.1 *x * np.cos(x)
    # fx = 0.1*x*np.cos(x)
    # Lf = lambda y, fx: (y-fx)**2
    plt.plot(x,f(x))

    gd = MGD(f, eta=0.01, gamma=0.9, NAG=True)
    # gd = GD(f, eta=0.01)

    x0 = -9.4
    plt.scatter(x0, f(x0), marker="x")
    MAXiter = 10000
    Niter = 0
    for _ in range(MAXiter):
        xnew = gd(x0)
        if abs(gd.A) < 1e-2:
            break
        else:
            x0 = xnew
        Niter += 1
    plt.scatter(x0, f(x0), marker="o")
    print(Niter)
    plt.show()
    # print(x0)



    # def f(x):
    #     return np.sin(x)
    # fder = egrad(f)
    # plt.plot(x,f(x), label="f")
    # plt.plot(x, fder(x), label="fder")
    # plt.plot(x, np.cos(x), ls="--")
    # plt.show()