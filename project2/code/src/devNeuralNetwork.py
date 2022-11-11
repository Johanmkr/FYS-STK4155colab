import numpy as np
import matplotlib.pyplot as plt
try:
    from src.Layer import Layer
    from src.LossFunctions import LossFunctions
    from src.ActivationFunction import ActivationFunction
    import src.GradientDescent as GradientDescent
    from src.DataHandler import DataHandler
    from src.ExtendAndCollapse import ExtendAndCollapse as EAC
    from src.utils import *
except ModuleNotFoundError:
    from Layer import Layer
    from LossFunctions import LossFunctions
    from ActivationFunction import ActivationFunction
    import GradientDescent as GradientDescent
    from DataHandler import DataHandler
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
                inputData  = None,
                targetData = None,
                hiddenLayers = 0,
                neuronsInEachLayer = 4,
                activationFunction = 'sigmoid',
                outputFunction = 'linear',
                outputNeurons = None,
                lossFunction = 'mse',
                optimizer = 'plain',
                epochs = None,
                batchSize = None,
                nrMinibatches = None,
                eta = 0.01,
                lmbda = 0.0,
                testSize = 0.2,
                terminalUpdate = False,
                classification = False):
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
        # self.ttSplit(self.testSize)
        self.classification = classification
        if self.classification:
            self.inputData, self.targetData, self.trainData, self.trainOut, self.testData, self.testOut = feature_scale_split(self.inputData, self.targetData, train_size=1-self.testSize)
        else:
            self.inputData, self.targetData, self.trainData, self.trainOut, self.testData, self.testOut = Z_score_normalise_split(self.inputData, self.targetData, train_size=1-self.testSize)
        self.setInputFeatures(self.trainData)
        self.outputNeurons = outputNeurons or self.features
        self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
            self.nrMinibatches = self.inputs//self.batchSize
        elif nrMinibatches is not None:
            self.nrMinibatches = nrMinibatches
            self.batchSize = self.inputs//self.nrMinibatches
        else:
            self.batchSize = self.inputs 
            self.nrMinibatches = 1
        self.eta = eta
        self.lmbda = lmbda
        self.Niter = 0  # Keep track of training iterations

        #   Make list of layer objects
        self.layers = []
        
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

    def __call__(self, X=None):
        if X is not None:
            self.layers[0].h = X
        self.feedForward()
        return self.outputData

    # def ttSplit(self, testSize):
    #     testIdx = np.random.choice(np.arange(len(self.inputData)), int(len(self.inputData) * testSize), replace=False)
    #     self.testData = self.inputData[testIdx]
    #     self.testOut = self.targetData[testIdx]
    #     self.trainData = np.delete(self.inputData, testIdx, axis=0)
    #     self.trainOut = np.delete(self.targetData, testIdx, axis=0)

    def updateInputLayer(self, data):
        self.setInputFeatures(data)
        self.layers[0].UpdateLayer(self.inputs, self.features)
        self.layers[0].h = data

    def sliceInputAndComparison(self, idx):
        self.layers[0].h = self.trainData[idx]
        self.comparisonData = self.trainOut[idx]

    def setInputFeatures(self, data):
        if data.ndim == 1 or data.ndim == 0:
            self.inputs = len(data)
            self.features = 1 
        else:
            self.inputs, self.features = data.shape

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

        for i in range(1, self.nrOfLayers): #looping over hidden layers + output layer
            nIn = self.layers[i-1].neurons
            nOut = self.layers[i].neurons
            self.layers[i].setWmatrix(nIn,nOut)

    def feedForward(self):
        for i in range(1, self.nrOfLayers):
            w = self.layers[i].w
            h = self.layers[i-1].h
            b = self.layers[i].bias

            activator = h @ w.T + b
            self.layers[i].a = activator
            self.layers[i].h = self.activationFunction(activator)
        #   Override last output
        self.layers[-1].h = self.outputFunction(self.layers[-1].a)
        self.outputData = self.layers[-1].h

    def backPropagation(self):
        self.layers[-1].delta = self.activationFunction.derivative(self.layers[-1].a) * self.lossFunction.derivative(self.outputData, self.comparisonData)
        #   Find and set deltas
        for l in range(self.nrOfLayers-2, 0, -1):
            delta = self.layers[l+1].delta
            w = self.layers[l+1].w
            a = self.layers[l].a
            # embed()
            self.layers[l].delta = (delta @ w) * self.activationFunction.derivative(a)
        # embed()
        F = EAC(self.layers) # Generating full matrices W, B, D, H

        
        regularisation = 0
        if self.lmbda > 0.0:
            regularisation = self.lmbda * F.W

        # sgd.simple_initialise()
        gradW = F.regGradW() + regularisation
        gradB = F.regGradB()
        # maxVal = 1e6
        if np.any(gradW > maxVal) or np.any(gradB > maxVal):
            pass
            # print("passed")
        else:
            # print(F.W)
            F.W = self.sgdW.simple_update(gradW, F.W)
            F.B = self.sgdB.simple_update(gradB, F.B)

        # F.W = F.W - self.eta * F.regGradW()
        # F.B = F.B - self.eta * F.regGradB()

        newWeights = F.collapseWeights()
        newBiases = F.collapseBiases()

        for i in range(1, self.nrOfLayers):
            self.layers[i].w = newWeights[i-1]
            self.layers[i].bias = newBiases[i-1]

    def setRandomIndecies(self):
        return np.random.choice(np.arange(self.inputs), size=self.batchSize, replace=False)

    def accuracy(self, prediction, target, tol=1e-3):
        samples = len(target)
        accuracy = 0
        for i in range(samples):
            if abs(prediction[i]-target[i]) < tol:
                accuracy += 1
        accuracy /= samples
        return accuracy

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
    # x = np.linspace(-5,5, 100)
    # y = x*np.cos(x) + 0.5*np.random.randn(len(x))
    # ynorm = (y-np.mean(y))/np.std(y)
    # x = x[:,np.newaxis]
    # ynorm = ynorm[:,np.newaxis]
    # plt.plot(x,ynorm, ".", label="data")
    # # plt.show()
    # dummy = devNeuralNetwork(x, ynorm, hiddenLayers=3, neuronsInEachLayer=4)
    # print(dummy)
    # # dummy.train()
    # # dummy()
    # Niter = 16
    # dummy.train()
    # pred = dummy()
    # plt.plot(x, pred, label="Untrained")
    # LF = LossFunctions()
    # t0 = time()
    # for i in range(1, Niter+1):
    #     N =  (250*i)
    #     dummy.train(N)
    #     pred = dummy()
    #     # embed()
    #     loss = np.mean(LF(pred, ynorm))
    #     # embed()
    #     print(f"Loss for N = {i}: {loss:.2f}  with {N} iterations")
    #     if i % 4 == 0:
    #         plt.plot(x,pred, label=f"N={i}")
    # t1 = time()

    # # xtest = np.linspace(-10,10,1000)

    # # ypredtest = dummy(xtest)
    # # plt.plot(xtest, ypredtest, label="Test point")


    # plt.legend()
    # print(f"Duration: {t1-t0:.2f} s")
    # plt.show()


    # TEST FRANKE
    # Make data.
    space = np.linspace(0,1,20)
    xx, yy = np.meshgrid(space,space)

    def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    zz = FrankeFunction(xx, yy)
    zzr = zz.ravel()
    zzr = (zzr-np.mean(zzr))/np.std(zzr)
    FrankeX = np.zeros((len(zzr),2))
    FrankeX[:,0] = xx.ravel()
    FrankeX[:,1] = yy.ravel()
    FrankeY = zzr[:,np.newaxis]
    FNet = devNeuralNetwork(FrankeX, FrankeY, hiddenLayers=3, activationFunction='tanh', neuronsInEachLayer=40, outputNeurons=1, epochs=1000, batchSize=20, testSize=0.2, lmbda=0.001, eta=0.01)
    print(FNet)
    # FNet.train(10000)
    # FrankePred = FNet()
    t0 = time()
    FNet.train()
    t1 = time()
    print(f"Training time: {t1-t0:.2f} s")
 
    # LF = LossFunctions()
    # print(FNet)
    # for i in range(1, Niter+1):
    #     N =  (250*i)
    #     FNet.train(N)
    #     FrankePred = FNet(FrankeX)
    #     # embed()
    #     loss = np.mean(LF(FrankePred, FNet.targetData))
    #     # embed()
    #     print(f"Loss for N = {i}: {loss:.2f}  with {N} iterations")
    #     # if i % 4 == 0:
    #         # plt.plot(x,pred, label=f"N={i}")
    FrankePred = FNet(FrankeX)
    print(f"Total iterations: {FNet.Niter}")

    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, np.reshape(FrankePred, (20,20)), cmap="coolwarm", alpha=0.7)
    scatter = ax.scatter(xx,yy,zzr, color="green")
    plt.show()

    # print(dummy.weights[0].w.shape)
