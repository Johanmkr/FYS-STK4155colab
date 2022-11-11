import numpy as np

class ActivationFunction:
    def __init__(self,
                activationFunction = "sigmoid"):
        self.parseFunction(activationFunction)

    def parseFunction(self, stringToParse):
        g = stringToParse.strip().lower() # case insensitive
        if g in ['sigmoid', 'sigmiod']: # ...
            self.activationFunction = self.sigmoid
            self.derivativeFunction = self.sigmoidDerivative
        elif g in ['relu', 'rectified linear unit', 'rectifier']:
            self.activationFunction = self.ReLU
            self.derivativeFunction = self.ReLUDerivative
        elif g in ['leaky relu', 'lrelu', 'relu*', 'leaky rectifier']:
            self.activationFunction = self.leakyReLU
            self.derivativeFunction = self.leakyReLUDerivative
        elif g in ['tanh', 'hyperbolic tangent']:
            self.activationFunction = self.hyperbolicTangent
            self.derivativeFunction = self.hyperbolicTangentDerivative
        elif g in ['softmax']:
            self.activationFunction = self.softmax
            self.derivativeFunction = self.softmaxDerivative
        elif g in ['linear', 'lin']:
            self.activationFunction = self.linear
            self.derivativeFunction = self.linearDerivative
        else:
            raise ValueError(f"The library does not have functionalities for {g} activation function.")

        
        # add other functions

    def derivative(self, a):
        return self.derivativeFunction(a)

    def __call__(self, a):
        return self.activationFunction(a)

    def sigmoid(self, a):
        try:
            # print(a)
            return 1/(1+np.exp(-a))
        except RuntimeWarning:
            print(a)
    def sigmoidDerivative(self, a):
        return (self.sigmoid(a) * (1-self.sigmoid(a)))

    def ReLU(self, a):
        # return np.max([0, a])
        return np.where(a > 0, a, 0)

    def ReLUDerivative(self, a):
        # undefined in x=0, do we need to care about this?
        #return self.ReLU(a)/a 
        return np.where(a > 0, 1, 0)
    
    def leakyReLU(self, a):
        # return self.leakyReLUDerivative(a)*a
        return np.where(a>0, a, 0.1*a)

    def leakyReLUDerivative(self, a):
        # undefined in x=0, do we need to care about this?
        return np.where(a>0, 1, 0.01)

    def hyperbolicTangent(self, a):
        return np.tanh(a)
    
    def hyperbolicTangentDerivative(self, a):
        return 1 - np.tanh(a)**2

    def softmax(self, a):
        exps = np.exp(a)
        return exps/np.sum(exps)

    def softmaxDerivative(self, a):
        sm = self.softmax(a)

        # do not think this actually works...
        # https://e2eml.school/softmax.html
        return sm*np.identity(sm.size) - sm.transpose() @ sm

    def linear(self, a):
        return a
        # return a

    def linearDerivative(self, a):
        return 1

    #add other functions