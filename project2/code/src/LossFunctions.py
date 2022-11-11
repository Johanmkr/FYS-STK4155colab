import numpy as np

class LossFunctions:
    def __init__(self,
                lossFunction = 'mse'):
        self.lossFunction = lossFunction
        self.parseFunction(self.lossFunction)
    
    def parseFunction(self, stringToParse):
        if stringToParse == 'mse':
            self.lossFunction = self.mse 
            self.derivativeFunction = self.mseDerivative
        elif stringToParse == "crossentropy":
            self.lossFunction = self.crossentropy
            self.derivativeFunction = self.crossentropyDerivative

    def derivative(self, prediction, target):
        return self.derivativeFunction(prediction, target)

    def __call__(self, prediction, target):
        return self.lossFunction(prediction, target)

    def mse(self, prediction, target):
        return ((prediction-target)**2) / 2

    def mseDerivative(self, prediction, target):
        return (prediction-target)

    def crossentropy(self, prediction, target):
        return -(target*np.log(prediction) + (1-target)*np.log(1-prediction))

    def crossentropyDerivative(self, prediction, target):
        return (prediction-target)/(1-prediction)/prediction

