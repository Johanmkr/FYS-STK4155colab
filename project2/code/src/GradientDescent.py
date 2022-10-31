import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad 

class GradientDescent:
    def __init__(self, LossFunction, eta):
        self.LF = LossFunction  #Loss function as function of theta
        self.eta = eta  #Learning rate

    
class GD(GradientDescent):
    def __init__(self, LossFunction, eta=0.1):
        GradientDescent.__init__(LossFunction=LossFunction, eta=eta)

    def find_A(self, LFargument)
        self.A = grad(self.LF(LFargument))

    #does not update eta yet
    def __call__(self, theta):
        self.find_A(theta)
        self.v = self.eta*self.A
        theta = theta - self.v
        return theta #could consider making theta a class variabel

class MGD(GD):
    def __init__(self, LossFunction, eta=0.1, gamma=0.1):
        GD.__init__(LossFunction=LossFunction, eta=eta)
        self.gamma = gamma
        self.v_prev = 0

    def __call__(self, theta):
        self.find_A(theta+gamma*self.v_prev)
        self.v = gamma*self.v_prev + self.eta*self.A
        theta = theta - self.v
        self.v_prev = self.v
        return theta

    

class plain_SGD(GradientDescent):
    def __init__(self, LossFunction, eta, gamma):
        GradientDescent.__init__(LossFunction, eta)
        self.gamma = gamma



