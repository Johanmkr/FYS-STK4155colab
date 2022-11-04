import numpy as np

class ActivationFunction:
    def __init__(self,
                activationFunction = "sigmoid"):
        self.activationFunction = activationFunction
        self.parseFunction(self.activationFunction)

    def parseFunction(self, stringToParse):
        g = stringToParse.strip().lower() # case insensitive
        if g == 'sigmoid':
            self.activationFunction = self.sigmoid
            self.derivativeFunction = self.sigmoidDerivative
        elif q in ['relu', 'rectified linear unit']:
            self.activationFunction = self.ReLU
            self.derivativeFunction = self.ReLUDerivative
        elif q in ['leaky relu', 'lrelu', 'relu*']: #idk
            self.activationFunction = self.leakyReLU
            self.derivativeFunction = self.leakyReLUDerivative
        elif g in ['tanh', 'hyperbolic tangent']:
            self.activationFunction = self.hyperbolicTangent
            self.derivativeFunction = self.hyperbolicTangentDerivative
        
        # add other functions

    def derivative(self, a):
        return self.derivativeFunction(a)

    def __call__(self, a):
        return self.activationFunction(a)

    def sigmoid(self, a):
        return 1/(1+np.exp(-a))

    def sigmoidDerivative(self, a):
        return (self.sigmoid(a) * (1-self.sigmoid(a)))

    def ReLU(self, a):
        return np.max([0, a])

    def ReLUDerivative(self, a):
        # undefined in x=0, do we need to care about this?
        #return self.ReLU(a)/a 
        return np.where(a>0, 1, 0)
    
    def leakyReLU(self, a):
        return self.leakyReLUDerivative(a)*a

    def leakyReLUDerivative(self, a):
        # undefined in x=0, do we need to care about this?
        return np.where(a>0, 1, 0.1)

    def hyperbolicTangent(self, a):
        return np.tanh(a)
    
    def hyperbolicTangentDerivative(self, a):
        return 1 - np.tanh(a)**2

    def siftmac(self, a):
        return np.exp(a)/np.sum(np.exp(a))

    #add other functions