import numpy as np
import matplotlib.pyplot as plt
try:
    from src.Layer import Layer
    from src.LossFunctions import LossFunctions
    from src.ActivationFunction import ActivationFunction
    import src.GradientDescent as GradientDescent
    from src.ExtendAndCollapse import ExtendAndCollapse as EAC
    from src.utils import *
except ModuleNotFoundError:
    from Layer import Layer
    from LossFunctions import LossFunctions
    from ActivationFunction import ActivationFunction
    import GradientDescent as GradientDescent
    from ExtendAndCollapse import ExtendAndCollapse as EAC
    from utils import *
from IPython import embed
from time import time
from tqdm import trange
from sys import float_info

np.random.seed(69)

maxVal = 1/float_info.epsilon

class devNeuralNetwork:
    def __init__(self,
                inputData:np.ndarray  = None,
                targetData:np.ndarray = None,
                hiddenLayers:int = 0,
                neuronsInEachLayer:int = 4,
                activationFunction:str = 'sigmoid',
                outputFunction:str = 'linear',
                outputNeurons:int = None,
                lossFunction:str = 'mse',
                optimizer:str = 'plain',
                epochs:int = None,
                batchSize:int = None,
                nrMinibatches:int = None,
                eta:float = 0.01,
                lmbda:float = 0.0,
                testSize:float = 0.2,
                terminalUpdate:bool = False,
                classification:bool = False) -> None:
        """Initialises Feed Forward Neural network with the following parameters:

        Args:
            inputData (ndarray, optional): Must be multidimensional: dims [n,k] with n data points and k>0. Defaults to None.
            targetData (ndarray, optional): Will be used to determine error of network output. Defaults to None.
            hiddenLayers (int, optional): Number of hidden layers in network. Defaults to 0.
            neuronsInEachLayer (int, optional): Number of neurons per hidden layer. Defaults to 4.
            activationFunction (str, optional): Activation function (used in hidden layers) as string. Will be parsed from activationFunction.py. Defaults to 'sigmoid'.
            outputFunction (str, optional): Output function  (used in output layer only) as string. Will be parsed from activationFunction.py. Defaults to 'linear'.
            outputNeurons (int, optional): Number of output neurons. If not set, output neurons will be features of target data. Defaults to None.
            lossFunction (str, optional): Loss function as string. Will be parsed from lossFunction.py. Defaults to 'mse'.
            optimizer (str, optional): Optimizer to use when performing gradient descent. Defaults to 'plain'.
            epochs (int, optional): Number of epochs over which we train the network. Defaults to None.
            batchSize (int, optional): Number of data points per batch used is stochastic gradient descent. Defaults to None.
            nrMinibatches (int, optional): Number of minibatches to generate. Cannot be combined with batchSize. Defaults to None.
            eta (float, optional): Learning rate. Defaults to 0.01.
            lmbda (float, optional): Regularisation parameter. Defaults to 0.0.
            testSize (float, optional): Fraction of input data used for testing. Defaults to 0.2.
            terminalUpdate (bool, optional): If true, progress is written  to the terminal during training. Defaults to False.
            classification (bool, optional): If true, the networks deals with a classification problem. Defaults to False.
        """

        #   Set self variables
        self.inputData = inputData
        self.targetData = targetData
        self.comparisonData = self.targetData
        self.hiddenLayers = hiddenLayers
        self.neuronsInEachLayer = neuronsInEachLayer
        self.activationFunctionString = activationFunction
        self.outputFunctionString = outputFunction
        self.lossFunctionString = lossFunction
        self.activationFunction = ActivationFunction(activationFunction)
        self.outputFunction = ActivationFunction(outputFunction)
        self.lossFunction = LossFunctions(lossFunction)
        self.optimizer = optimizer
        self.terminalUpdate = terminalUpdate
        self.testSize = testSize
        self.classification = classification
        self.epochs = epochs
        self.eta = eta
        self.lmbda = lmbda
        self.Niter = 0  # Keep track of training iterations

        #   Set weight initialisation according to the given activation function
        if activationFunction in ["sigmoid", "tanh"]:
            self.weightInitialiser = "Xavier"
        elif activationFunction in ["relu", "lrelu", "relu*"]:
            self.weightInitialiser = "He"

        #   Split and normalise data. Classification analysis does not include feature scaling. 
        if self.classification:
            self.inputData, self.targetData, self.trainData, self.trainOut, self.testData, self.testOut = feature_scale_split(self.inputData, self.targetData, train_size=1-self.testSize)
        else:
            self.inputData, self.targetData, self.trainData, self.trainOut, self.testData, self.testOut = Z_score_normalise_split(self.inputData, self.targetData, train_size=1-self.testSize)
        
        #   Set input and feature
        self.setInputFeatures(self.trainData)
        self.outputNeurons = outputNeurons or self.features

        #   Set batch size or nr. minibatches depending on what information is given in the initialisation
        if batchSize is not None:
            self.batchSize = batchSize
            self.nrMinibatches = self.inputs//self.batchSize
        elif nrMinibatches is not None:
            self.nrMinibatches = nrMinibatches
            self.batchSize = self.inputs//self.nrMinibatches
        else:
            self.batchSize = self.inputs 
            self.nrMinibatches = 1
        
        #   Make list of layer objects
        self.layers = []
        
        #   Initialise layers objects
        self.setInitialisedLayers()
        self.nrOfLayers = len(self.layers)

        #   Initialise the weights
        self.initialiseWeights()

    def __str__(self):
        """Generates print of the network structure.

        Returns:
            str: Network structure.
        """
        stringToReturn = 'Feed Forward Neural Network\n'
        stringToReturn += f"Number of layers: {self.nrOfLayers}\n"
        stringToReturn      += f'Input layer: {self.layers[0].neurons} neurons\n'
        for i in range(1, len(self.layers)-1):
            stringToReturn  += f'H-layer {i}: {self.layers[i].neurons} neurons\n'
        stringToReturn      += f'Output layer: {self.layers[-1].neurons} neurons\n\n'

        stringToReturn += f"Nr. epochs: {self.epochs}\n"
        stringToReturn += f"Batch size: {self.batchSize}\n"
        stringToReturn += f"Nr batches: {self.nrMinibatches}\n"
        stringToReturn += f"Test size: {self.testSize}\n"
        stringToReturn += f"Total iterations: {self.nrMinibatches * self.epochs}\n"
        stringToReturn += f"Lambda: {self.lmbda}\n"
        stringToReturn += f"Eta: {self.eta}\n\n"

        stringToReturn += f"Activation function: {self.activationFunctionString}\n"
        stringToReturn += f"Output function: {self.outputFunctionString}\n"
        stringToReturn += f"Loss function: {self.lossFunctionString}\n"
        stringToReturn += f"Optimizer: {self.optimizer}\n\n"
        return stringToReturn

    def __call__(self, X:np.ndarray=None) -> None:
        """Performs one forward pass, either with the given data, or with the default data already initialised in input layer.

        Args:
            X (np.ndarray, optional): Data to initialise first layer. Defaults to None.

        Returns:
            np.ndarray: Output data, values of the output layer after output function after one forward pass.
        """
        if X is not None:
            # self.layers[0].h = X
            self.updateInputLayer(X)
        self.feedForward()
        return self.outputData

    def updateInputLayer(self, data:np.ndarray) -> None:
        """Updates input layer. Sets new inputs and features according to the updated layer.

        Args:
            data (np.ndarray): Data to initialise the input layer. 
        """
        self.setInputFeatures(data)
        self.layers[0].UpdateLayer(self.inputs, self.features)
        self.layers[0].h = data

    def sliceInputAndComparison(self, idx:np.ndarray) -> None:
        """Slices the train data. If indices are random, this correspond to random resampling of train data. 

        Args:
            idx (np.ndarray): Indices of train data that we set as the first layer values.
        """
        self.layers[0].h = self.trainData[idx]
        self.comparisonData = self.trainOut[idx]

    def setInputFeatures(self, data:np.ndarray) -> None:
        """Initialises the inputs and feature variables. This could be done if we change the input data set for instance.

        Args:
            data (np.ndarray): Data to initialise first layer. 
        """
        if data.ndim == 1 or data.ndim == 0:
            self.inputs = len(data)
            self.features = 1 
        else:
            self.inputs, self.features = data.shape

    def setInitialisedLayers(self) -> None:
        """Initialises the layers according to the number of hidden layers and neurons in each layer.
        """
        #   Append input layer
        self.layers.append(Layer(activationFunction=self.activationFunction, inputs=self.inputs, features=self.features))
        #   Append hidden layers
        for i in range(self.hiddenLayers):
            if isinstance(self.neuronsInEachLayer, (tuple, list, np.ndarray)):
                    try:
                        self.layers.append(Layer(neurons=self.neuronsInEachLayer[i], activationFunction=self.activationFunction))
                    except IndexError:
                        self.layers.append(Layer(neurons=self.neuronsInEachLayer[-1], activationFunction=self.activationFunction))
            else:
                self.layers.append(Layer(neurons=self.neuronsInEachLayer, activationFunction=self.activationFunction))
        #   Append output layers
        self.layers.append(Layer(self.outputNeurons, activationFunction=self.outputFunction))

    def addLayer(self, layer:Layer = None, idx:int = -1, neurons:int = None) -> None:
        """Adds a layer to the list of layer object at a given index.

        Args:
            layer (Layer, optional): Layer object to be added. Defaults to None.
            idx (int, optional): Index at which to add the layer in the layer list. Defaults to -1.
            neurons (int, optional): Number of neurons in the added layer. Defaults to None.
        """
        if layer is not None:
            LayerToAdd = layer
        else:
            LayerToAdd = Layer(neurons=neurons, activationFunction=self.activationFunction)
        self.layers.insert(idx, LayerToAdd)
        self.nrOfLayers += 1
        self.initialiseWeights()

    def initialiseWeights(self) -> None:
        """Initialises the weights.
        """
        for i in range(1, self.nrOfLayers): #looping over hidden layers + output layer
            nIn = self.layers[i-1].neurons
            nOut = self.layers[i].neurons
            self.layers[i].setWmatrix(nIn,nOut, initialisation=self.weightInitialiser)

    def feedForward(self) -> None:
        """Feed forward pass
        """
        #   Loop through all the layers except the input layer.
        for i in range(1, self.nrOfLayers):
            w = self.layers[i].w
            h = self.layers[i-1].h
            b = self.layers[i].bias

            activator = h @ w.T + b
            self.layers[i].a = activator
            self.layers[i].h = self.activationFunction(activator)
        #   Override last output -> run last output through the output function
        self.layers[-1].h = self.outputFunction(self.layers[-1].a)
        self.outputData = self.layers[-1].h

    def backPropagation(self) -> None:
        """Back propagation pass
        """
        #   Find delta (error) of output layer
        self.layers[-1].delta = self.activationFunction.derivative(self.layers[-1].a) * self.lossFunction.derivative(self.outputData, self.comparisonData)
        #   Find and set deltas for the remaining layers
        for l in range(self.nrOfLayers-2, 0, -1):
            delta = self.layers[l+1].delta
            w = self.layers[l+1].w
            a = self.layers[l].a
            self.layers[l].delta = (delta @ w) * self.activationFunction.derivative(a)

        F = EAC(self.layers) # Generating full matrices W, B, D, H in order to update all weights and biases in one computation. 

        gradW = F.regGradW()
        gradB = F.regGradB()
        if self.lmbda > 0.0:
            gradW += self.lmbda * F.W

        #   Update weight and biases, avoid too large gradients.
        if np.any(gradW > maxVal) or np.any(gradB > maxVal):
            pass
        else:
            F.W = self.sgdW.simple_update(gradW, F.W)
            F.B = self.sgdB.simple_update(gradB, F.B)

        #   Collapse weights and biases back down in to layer form
        newWeights = F.collapseWeights()
        newBiases = F.collapseBiases()

        #   Update weights and biases for each laye
        for i in range(1, self.nrOfLayers):
            self.layers[i].w = newWeights[i-1]
            self.layers[i].bias = newBiases[i-1]

    def setRandomIndecies(self):
        return np.random.choice(np.arange(self.inputs), size=self.batchSize, replace=False)

    def accuracy(self, prediction, target, tol=1e-5):
        prediction = np.where(prediction > 0.5, 1, 0)
        accuracy = np.where(np.abs(prediction-target) < tol, 1, 0)
        return np.mean(accuracy)

    def get_testLoss(self):
        if self.classification:
            return self.accuracy(self.__call__(self.testData), self.testOut)
        else:
            return np.mean(self.lossFunction(self.__call__(self.testData), self.testOut))
    
    def get_trainLoss(self):
        if self.classification:
            return self.accuracy(self.__call__(self.trainData), self.trainOut)
        else:
            return np.mean(self.lossFunction(self.__call__(self.trainData), self.trainOut))

    def printTrainingInfo(self, epoch):
        trainLoss = self.get_trainLoss()
        testLoss = self.get_testLoss()
        stringToPrint = f"Epochs: {epoch}\n"
        if self.classification:
            stringToPrint += f"Train accuracy:  {trainLoss:.2f}\n"
            stringToPrint += f"Test accuracy:   {testLoss:.2f}\n"
        else:
            stringToPrint += f"Train loss:   {trainLoss:.2f}\n"
            stringToPrint += f"Test loss:    {testLoss:.2f}\n"
        print(stringToPrint)

    def train(self, extractInfoPerXEpoch = None):
        self.sgdW = GradientDescent.SGD.simple_initialise(eta=self.eta)
        self.sgdB = GradientDescent.SGD.simple_initialise(eta=self.eta)
        self.sgdW.set_update_rule(self.optimizer)
        self.sgdB.set_update_rule(self.optimizer)

        self.updateInputLayer(self.trainData)
        if self.terminalUpdate:
            for epoch in trange(1, self.epochs+1):
                for i in range(self.nrMinibatches):
                    self.sliceInputAndComparison(self.setRandomIndecies())
                    self.feedForward()
                    self.backPropagation()
                    self.Niter += 1
                if epoch % 25 == 0:
                    self.printTrainingInfo(epoch)
        elif extractInfoPerXEpoch is not None:
            self.testLossPerEpoch = [self.get_testLoss()]
            self.lossEpochs = [0]
            for epoch in range(1, self.epochs+1):
                for i in range(self.nrMinibatches):
                    self.sliceInputAndComparison(self.setRandomIndecies())
                    self.feedForward()
                    self.backPropagation()
                    self.Niter += 1
                if epoch % extractInfoPerXEpoch == 0:
                    self.testLossPerEpoch.append(self.get_testLoss())
                    self.lossEpochs.append(epoch)
        else:
            for epoch in range(1, self.epochs + 1):
                for i in range(self.nrMinibatches):
                    self.sliceInputAndComparison(self.setRandomIndecies())
                    self.feedForward()
                    self.backPropagation()
                    self.Niter += 1


if __name__=="__main__":
    pass
