import numpy as np
import matplotlib.pyplot as plt
try:
    from src.Layer import Layer
    from src.LossFunctions import LossFunctions
    from src.ActivationFunction import ActivationFunction
    import src.GradientDescent as GradientDescent
    from src.DataHandler import DataHandler
    from src.ExtendAndCollapse import ExtendAndCollapse as EAC
except ModuleNotFoundError:
    from Layer import Layer
    from LossFunctions import LossFunctions
    from ActivationFunction import ActivationFunction
    import GradientDescent as GradientDescent
    from DataHandler import DataHandler
    from ExtendAndCollapse import ExtendAndCollapse as EAC
from IPython import embed
from time import time
from tqdm import trange

np.random.seed(69)



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
                epochs = None,
                batchSize = None,
                eta = 0.01,
                lmbda = 0.0,
                testSize = 0.2):
        self.inputData = inputData
        self.targetData = targetData
        self.comparisonData = self.targetData
        self.hiddenLayers = hiddenLayers
        self.neuronsInEachLayer = neuronsInEachLayer
        self.activationFunction = ActivationFunction(activationFunction)
        self.outputFunction = ActivationFunction(outputFunction)
        self.lossFunction = LossFunctions(lossFunction)

        self.testSize = testSize
        self.ttSplit(self.testSize)
        self.setInputFeatures(self.trainData)
        self.outputNeurons = outputNeurons or self.features
        self.epochs = epochs
        self.batchSize = batchSize or self.inputs
        self.iterations = self.inputs//self.batchSize
        self.eta = eta
        self.lmbda = lmbda
        self.Niter = 0  # Keep track of training iterations

        #   Make list of layer objects
        self.layers = []
        
        self.setInitialisedLayers()
        self.nrOfLayers = len(self.layers)

        #   Initialise the weights
        self.initialiseWeights()
        #   Split into training and test data
        # embed()

    def __str__(self):
        stringToReturn = 'Feed Forward Neural Network\n'
        stringToReturn += f"Number of layers: {self.nrOfLayers}\n"
        stringToReturn      += f'Input layer: {self.layers[0].neurons} neurons\n'
        for i in range(1, len(self.layers)-1):
            stringToReturn  += f'H-layer {i}: {self.layers[i].neurons} neurons\n'
        stringToReturn      += f'Output layer: {self.layers[-1].neurons} neurons\n'
        
        return stringToReturn

    def __call__(self, X=None, Y=None):

        if X is not None:
            self.layers[0].h = X
        self.feedForward()
        return self.outputData

    def ttSplit(self, testSize):
        testIdx = np.random.choice(np.arange(len(self.inputData)), int(len(self.inputData) * testSize), replace=False)
        self.testData = self.inputData[testIdx]
        self.testOut = self.targetData[testIdx]
        self.trainData = np.delete(self.inputData, testIdx, axis=0)
        self.trainOut = np.delete(self.targetData, testIdx, axis=0)

    def updateInputLayer(self, data):
        self.setInputFeatures(data)
        self.layers[0].UpdateLayer(self.inputs, self.features)
        # if data.ndim == 1 or data.ndim:
        #     self.layers[0].h = data
        # else:
        #     self.layers[0].h = data
        self.layers[0].h = data

    def sliceInputAndComparison(self, idx):
        self.layers[0].h = self.inputData[idx]
        self.comparisonData = self.targetData[idx]

    def setInputFeatures(self, data):
        # self.inputs = len(data)
        # self.features = 1 
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
            self.layers[l].delta = (delta @ w) * self.activationFunction.derivative(a)

        F = EAC(self.layers) # Generating full matrices W, B, D, H

        
        regularisation = 0
        if self.lmbda > 0.0:
            regularisation = self.lmbda * F.W

        # sgd.simple_initialise()
        F.W = self.sgdW.simple_update(F.regGradW()+regularisation, F.W)
        F.B = self.sgdB.simple_update(F.regGradB(), F.B)


        # F.W = F.W - self.eta * F.regGradW()
        # F.B = F.B - self.eta * F.regGradB()

        newWeights = F.collapseWeights()
        newBiases = F.collapseBiases()

        for i in range(1, self.nrOfLayers):
            self.layers[i].w = newWeights[i-1]
            self.layers[i].bias = newBiases[i-1]

    def setRandomIndecies(self):
        return np.random.choice(np.arange(self.inputs), size=self.batchSize, replace=False)

    def printTrainingInfo(self, epoch):
        trainLoss = np.mean(self.lossFunction(self.__call__(self.trainData), self.trainOut))
        testLoss = np.mean(self.lossFunction(self.__call__(self.testData), self.testOut))
        stringToPrint = f"Epochs: {epoch}\n"
        stringToPrint += f"Train loss:   {trainLoss:.2f}\n"
        stringToPrint += f"Test loss:    {testLoss:.2f}\n"
        print(stringToPrint)

    def train(self):
        self.sgdW = GradientDescent.SGD.simple_initialise(eta=self.eta)
        self.sgdB = GradientDescent.SGD.simple_initialise(eta=self.eta)
        self.sgdW.set_update_rule("plain")
        self.sgdB.set_update_rule("plain")

        self.updateInputLayer(self.trainData)
        for epoch in trange(self.epochs):
            for i in range(self.iterations):
                self.sliceInputAndComparison(self.setRandomIndecies())
                self.feedForward()
                self.backPropagation()
                self.Niter += 1
            if epoch % 100 == 0:
                self.printTrainingInfo(epoch)


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
    FNet = devNeuralNetwork(FrankeX, FrankeY, hiddenLayers=3, activationFunction='relu*', neuronsInEachLayer=40, outputNeurons=1, epochs=1000, batchSize=20, testSize=0.2, lmbda=0.1, eta=0.01)
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
