import autograd.numpy as np
from autograd import elementwise_grad as egrad 
from autograd import grad 

class GradientDescent:
    def __init__(self, CostFunction, eta):
        self.CostFunction = CostFunction    #Cost function as function of theta
        self.eta = eta  #Learning rate

    
class GD(GradientDescent):
    def __init__(self, CostFunction, eta=0.1):
        GradientDescent.__init__(CostFunction=CostFunction, eta=eta)
        self.N = grad(self.CostFunction)

    #does not update eta yet
    def __call__(self, theta):
        theta = theta - self.eta*self.g
        return theta

class plain_SGD(GradientDescent):
    def __init__(self, CostFunction, eta, gamma):
        GradientDescent.__init__(CostFunction, eta)
        self.gamma = gamma



