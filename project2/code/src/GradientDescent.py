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
    def __init__(self, X, LF, eta=0.01, gamma=0.9, nr_epochs=1000, nr_minibatches=40):
        GradientDescent.__init__(self, LF=LF, eta=eta)
        self.gamma = gamma
        self.nr_epochs = nr_epochs
        self.m = nr_minibatches
        self.X = X
        self.M = int(len(X)/self.m)
        self.tau = 1* self.nr_epochs * self.m
        self.eta0 = eta
        self.eta_tau = 0.01 * self.eta0

    def learning_schedule(self, k):
        self.eta = (1-k/self.tau)*self.eta0 + k/self.tau*self.eta_tau
    
    def find_A(self, LFargument):
        self.A = egrad(self.LF)(LFargument)

    def find_v(self):
        self.v = -self.eta*self.A

    def __call__(self, theta):
        #   This is a wack way of doing this, needs improvement soon
        k = 1
        for epoch in range(self.nr_epochs):
            for i in range(self.m):
                random_indecies = np.random.randint(0, high=self.M*self.m, size=self.M)
                xi = self.X[random_indecies]
                self.find_A(xi)
                self.learning_schedule(k)
                self.find_v()
                theta = theta + self.v
                k += 1
            # plt.scatter(theta, f(theta), marker="1")
        return np.min(theta)






if __name__=="__main__":
    x = np.linspace(-10,10, 1000)
    # def f(x):
    #     return 0.1 * x * np.cos(x)
    f = lambda x: 0.1 *x * np.cos(x)
    y = f(x)
    # fx = 0.1*x*np.cos(x)
    # Lf = lambda y, fx: (y-fx)**2
    plt.plot(x,f(x))

    # gd = MGD(f, eta=0.01, gamma=0.9, NAG=True)
    # gd = GD(f, eta=0.01)
    gd = plain_SGD(X=x, LF=f)

    x0 = 0.0
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