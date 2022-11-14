import numpy as np
from sys import float_info
epsilon = float_info.epsilon

class LossFunctions:
    def __init__(self,
                lossFunction:str = 'mse') -> None:
        """Collection of loss functions and their derivatives.

        Args:
            lossFunction (str, optional): String to be parsed to set the function. Defaults to 'mse'.
        """
        self.lossFunction = lossFunction
        self.parseFunction(self.lossFunction)
    
    def parseFunction(self, stringToParse:str) -> None:
        """_summary_

        Args:
            stringToParse (str): _description_
        """
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
        return -(target*np.log(prediction+epsilon) + (1-target)*np.log(1-prediction+epsilon))

    def crossentropyDerivative(self, prediction, target):
        return (prediction-target)/(1-prediction+epsilon)/(prediction+epsilon)

