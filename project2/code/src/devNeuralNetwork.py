import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer
from LossFunctions import LossFunctions
# from WeightMatrix import WeightMatrix
from ActivationFunction import ActivationFunction
import GradientDescent as GD
from DataHandler import DataHandler
from IPython import embed

np.random.seed(1)



class devNeuralNetwork:
    def __init__(self,
                inputData  = None,
                targetData = None,
                hiddenLayers = 0,
                neuronsInEachLayer = 4,
                activationFunction = 'sigmoid',
                outputFunction = 'linear',
                outputNeurons = None,
                lossFunction = 'mse',):
        self.inputData = inputData
        self.targetData = targetData
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


    def updateInputLayer(self, data):
        self.setInputFeatures(data)
        self.layers[0].UpdateLayer(self.inputs, self.features)
        if data.ndim == 1 or data.ndim:
            self.layers[0].h = data
        else:
            self.layers[0].h = data
        # print(self.layers[0].h[:,0])

    def setInputFeatures(self, data):
        self.inputs = len(data)
        self.features = 1 
        # if data.ndim == 1 or data.ndim == 0:
        #     self.inputs = len(data)
        #     self.features = 1 
        # else:
        #     self.inputs, self.features = data.shape


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
            self.updateInputLayer(np.asarray(X))
        self.feedForward()
        return self.outputData

    
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
        # self.weights = []
        for i in range(1, self.nrOfLayers): #looping over hidden layers + output layer
            nIn = self.layers[i-1].neurons
            nOut = self.layers[i].neurons
            # self.weights.append(WeightMatrix(nIn, nOut))
            self.layers[i].setWmatrix(nIn,nOut)

    def feedForward(self, train=False):
        for i in range(1, self.nrOfLayers):
            # embed()

            # w = self.weights[i-1].w 

            w = self.layers[i].w
            h = self.layers[i-1].h
            b = self.layers[i].bias
            # activator = self.weights[i-1].w.T @ self.layers[i-1].h + self.layers[i].bias
            activator = h @ w.T + b
            self.layers[i].a = activator
            self.layers[i].h = self.activationFunction(activator)
        # self.outputData = DataHandler(self.layers[-1].h)
        self.outputData = self.layers[-1].h 
        if train:
            self.mu = np.mean(self.outputData)
            self.sigma = np.std(self.outputData)
        self.outputData = (self.outputData - self.mu)/self.sigma
        self.layers[-1].h = self.outputData
            # self.layers[-1].h = self.outputFunction(self.outputData.dataScaled)

        
    def backPropagation(self):
        eta = 0.001
        self.layers[-1].delta = self.activationFunction.derivative(self.layers[-1].a) * self.lossFunction.derivative(self.outputData, self.targetData)
        #   Find deltas
        for l in range(self.nrOfLayers-2, 0, -1):
            delta = self.layers[l+1].delta

            # w = self.weights[l].w
            
            w = self.layers[l+1].w
            a = self.layers[l].a

            self.layers[l].delta = delta @ w * self.activationFunction.derivative(a)

            # self.layers[l].delta =  self.layers[l+1].delta @ self.weights[l].w * self.activationFunction.derivative(self.layers[l].a)



        # for l in range(self.nrOfLayers-2,-1,-1):
        #     self.weights[l].w = self.weights[l].w - eta * self.layers[l+1].delta.T @ self.layers[l].h
        # for l in range(self.nrOfLayers-1, 0, -1):
        #     self.layers[l].bias = self.layers[l].bias -eta * np.sum(self.layers[l].delta, axis=0)


        # Testing new approach
        for l in range(self.nrOfLayers-1, 0, -1):
            w = self.layers[l].w
            delta = self.layers[l].delta 
            h = self.layers[l-1].h
            bias = self.layers[l].bias 

            self.layers[l].w = w - eta * delta.T @ h 
            self.layers[l].bias = bias - eta * np.sum(delta, axis=0)

        # self.weights[0].w = self.weights[0].w - eta * self.layers[1].delta.T @ self.layers[0].h
    
            #this must be tied together with the gradient descent code.
    def train(self, N=1):
        trainingData = self.inputData
        self.updateInputLayer(trainingData)
        for _ in range(N):
            self.feedForward(train=True)
            self.backPropagation()


if __name__=="__main__":
    x = np.linspace(-5,5, 100)
    y = x*np.cos(x) + 0.5*np.random.randn(len(x))
    ynorm = (y-np.mean(y))/np.std(y)
    x = x[:,np.newaxis]
    ynorm = ynorm[:,np.newaxis]
    plt.plot(x,ynorm, ".", label="data")
    # plt.show()
    dummy = devNeuralNetwork(x, ynorm, hiddenLayers=3, neuronsInEachLayer=4)
    print(dummy)
    # dummy.train()
    # dummy()
    Niter = 16
    dummy.train()
    pred = dummy()
    plt.plot(x, pred, label="Untrained")
    LF = LossFunctions()
    for i in range(1, Niter+1):
        N =  (250*i)
        dummy.train(N)
        pred = dummy()
        # embed()
        loss = np.mean(LF(pred, ynorm))
        # embed()
        print(f"Loss for N = {i}: {loss:.2f}  with {N} iterations")
        if i % 4 == 0:
            plt.plot(x,pred, label=f"N={i}")
    

    # xtest = np.linspace(-10,10,1000)

    # ypredtest = dummy(xtest)
    # plt.plot(xtest, ypredtest, label="Test point")


    plt.legend()
    plt.show()



    # print(dummy.weights[0].w.shape)
