import numpy as np
from Layer import Layer
from LossFunctions import LossFunctions
from WeightMatrix import WeightMatrix
from ActivationFunction import ActivationFunction
import GradientDescent as GD
from IPython import embed


class devNeuralNetwork:
    def __init__(self,
                inputData  = None,
                outputData = None,
                hiddenLayers = 0,
                neuronsInEachLayer = 4,
                activationFunction = None,
                outputFunction = None,
                outputNeurons = None,
                lossFunctionType = 'mse'):
        self.inputData = inputData
        self.outputData = outputData
        self.hiddenLayers = hiddenLayers
        self.neuronsInEachLayer = neuronsInEachLayer
        self.activationFunction = activationFunction
        self.outputFunction = outputFunction
        if outputNeurons is not None:
            self.outputNeurons = outputNeurons
        else:
            self.outputNeurons = len(outputData)
        self.lossFunction = LossFunctions(lossFunctionType)

        #   Make list of layer objects
        self.layers = []
        
        self.setInitialisedLayers()
        self.nrOfLayers = len(self.layers)

        #   Initialise the weights
        self.initialiseWeights()


    def __str__(self):
        stringToReturn = 'Feed Forward Neural Network\n'
        stringToReturn      += f'Input layer: {self.layers[0].neurons} neurons\n'
        for i in range(1, len(self.layers)-1):
            stringToReturn  += f'H-layer {i}: {self.layers[i].neurons} neurons\n'
        stringToReturn      += f'Output layer: {self.layers[-1].neurons} neurons\n'
        
        return stringToReturn

    
    def setInitialisedLayers(self):
        #   Append input layer
        self.layers.append(Layer(activationFunction=self.activationFunction, inputData=self.inputData))
        #   Append hidden layers
        for _ in range(self.hiddenLayers):
            self.layers.append(Layer(neurons=self.neuronsInEachLayer, activationFunction=self.activationFunction))
        #   Append output layers
        self.layers.append(Layer(self.outputNeurons, activationFunction=self.outputFunction))


    def addLayer(self,
                layer = None,
                idx = -1,
                neurons = None):

        if layer is not None:
            LayerToAdd = layer
        else:
            LayerToAdd = Layer(neurons=neurons, activationFunction=self.activationFunction)
        self.layers.insert(idx, LayerToAdd)
        self.nrOfLayers += 1
        self.initialiseWeights()


    def initialiseWeights(self):
        self.weights = []
        for i in range(self.nrOfLayers-1):
            nIn = self.layers[i].neurons
            nOut = self.layers[i+1].neurons
            self.weights.append(WeightMatrix(nIn, nOut))

    def feedForward(self):
        for i in range(1, self.nrOfLayers):
            activator = self.weights[i-1].w.T @ self.layers[i-1].h + self.layers[i].bias
            self.layers[i].a = activator
            self.layers[i].h = self.activationFunction(activator)
        
    def backPropagation(self):
        eta = 0.01
        self.layers[-1].delta = self.activationFunction.derivative(self.layers[-1].a) * self.lossFunction.derivative(self.layers[-1].h, self.outputData)

        for l in range(self.nrOfLayers, 1, -1):
            self.layers[l].delta = self.layers[l+1].delta @ self.weights[l].w * self.activationFunction.derivative(self.layers[l].a)
            self.weights[l] = self.weights[l] - eta * self.layers[l].delta @ self.layers[l-1].h
            self.layers[l].bias = self.layers[l] -eta * self.layers[l].delta

            #this must be tied together with the gradient descent code.
    def train(self, N):
        for _ in range(N):
            self.feedForward()
            self.backPropagation()


if __name__=="__main__":
    x = np.linspace(-10,10, 5)
    y = x**2
    dummy = devNeuralNetwork(x, y, hiddenLayers=3, neuronsInEachLayer=4, activationFunction=ActivationFunction('sigmoid'))

    dummy.train(1)
    # embed()
    # print(dummy.weights[0].w.shape)
