import numpy as np

class ActivationFunction:
    def __init__(self,
                activationFunction = "sigmoid"):
        self.activationFunction = activationFunction
        self.parseFunction(self.activationFunction)

    def parseFunction(self, stringToParse):
        if stringToParse == 'sigmoid':
            self.activationFunction = self.sigmoid
            self.derivativeFunction = self.sigmoidDerivative
        # add other functions

    def derivative(self, a):
        return self.derivativeFunction(a)

    def __call__(self, a):
        return self.activationFunction(a)

    def sigmoid(self, a):
        return 1/(1+np.exp(-a))

    def sigmoidDerivative(self, a):
        return (self.sigmoid(a) * (1-self.sigmoid(a)))

    #add other functions