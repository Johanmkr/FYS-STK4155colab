import numpy as np
import matplotlib.pyplot as plt
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
                activationFunction = 'sigmoid',
                outputFunction = 'linear',
                outputNeurons = None,
                lossFunction = 'mse',):
        self.inputData = inputData
        self.outputData = outputData
        self.hiddenLayers = hiddenLayers
        self.neuronsInEachLayer = neuronsInEachLayer
        self.activationFunction = ActivationFunction(activationFunction)
        self.outputFunction = ActivationFunction(outputFunction)
        self.lossFunction = LossFunctions(lossFunction)

        self.setInputFeatures(self.inputData)
        self.outputNeurons = outputNeurons or self.features

        #   Make list of layer objects
        self.layers = []
        
        self.setInitialisedLayers()
        self.nrOfLayers = len(self.layers)

        #   Initialise the weights
        self.initialiseWeights()
        # embed()


    def UpdateInputLayer(self, data):
        self.setInputFeatures(data)
        self.layers[0].UpdateLayer(self.inputs, self.features)
        if data.ndim == 1:
            self.layers[0].h[:,0] = data
        else:
            self.layers[0].h = data
        # print(self.layers[0].h[:,0])

    def setInputFeatures(self, data):
        if data.ndim == 1:
            self.inputs = len(data)
            self.features = 1 
        else:
            self.inputs, self.features = data


    def __str__(self):
        stringToReturn = 'Feed Forward Neural Network\n'
        stringToReturn += f"Number of layers: {self.nrOfLayers}\n"
        stringToReturn      += f'Input layer: {self.layers[0].neurons} neurons\n'
        for i in range(1, len(self.layers)-1):
            stringToReturn  += f'H-layer {i}: {self.layers[i].neurons} neurons\n'
        stringToReturn      += f'Output layer: {self.layers[-1].neurons} neurons\n'
        
        return stringToReturn

    def __call__(self, X=None):
        if X is not None:
            self.UpdateInputLayer(X)
        self.feedForward()
        return self.layers[-1].h

    
    def setInitialisedLayers(self):
        #   Append input layer
        self.layers.append(Layer(activationFunction=self.activationFunction, inputs=self.inputs, features=self.features))
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
            # embed()
            w = self.weights[i-1].w 
            h = self.layers[i-1].h
            b = self.layers[i].bias
            # activator = self.weights[i-1].w.T @ self.layers[i-1].h + self.layers[i].bias
            activator = h @ w.T + b
            self.layers[i].a = activator
            self.layers[i].h = self.activationFunction(activator)
        self.layers[-1].h = self.outputFunction(self.layers[-1].h)

        
    def backPropagation(self):
        eta = 0.1
        self.layers[-1].delta = self.activationFunction.derivative(self.layers[-1].a) * self.lossFunction.derivative(self.layers[-1].h, self.outputData[:,np.newaxis])

        # embed()
        for l in range(self.nrOfLayers-2, 0, -1):
            # embed()
            self.layers[l].delta =  self.layers[l+1].delta @ self.weights[l].w * self.activationFunction.derivative(self.layers[l].a)

            self.weights[l].w = self.weights[l].w - eta * self.layers[l+1].delta.T @ self.layers[l].h
            self.layers[l].bias = self.layers[l].bias -eta * self.layers[l].delta
        self.weights[0].w = self.weights[0].w - eta * self.layers[1].delta.T @ self.layers[0].h
    
            #this must be tied together with the gradient descent code.
    def train(self, N=1):
        trainingData = self.inputData
        self.setInputLayerData(trainingData)
        for _ in range(N):
            self.feedForward()
            self.backPropagation()


if __name__=="__main__":
    x = np.linspace(-5,5, 100)
    y = x*np.cos(x) + 0.5*np.random.randn(len(x))
    ynorm = (y-np.mean(y))/np.std(y)
    plt.plot(x,ynorm, ".", label="data")
    # plt.show()
    dummy = devNeuralNetwork(x, ynorm, hiddenLayers=1, neuronsInEachLayer=3)
    print(dummy)
    # dummy.train()
    # dummy()
    Niter = 10
    LF = LossFunctions()
    for i in range(Niter):
        dummy.train(1000)
        pred = dummy()
        #print(LF(pred, y))
        loss = np.mean(LF(pred, ynorm))
        # embed()
        print(f"Loss for N = {i+1}: {loss:.2f}")
    plt.plot(x,pred, label=f"N={i+1}")

    xtest = 3.5
    ypredtest = dummy(xtest)
    # print(ypredtest)

    plt.plot(xtest, ypredtest, ".", label="Test point")


    plt.legend()
    plt.show()



    # print(dummy.weights[0].w.shape)
