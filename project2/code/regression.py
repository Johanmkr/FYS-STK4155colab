import numpy as np
import matplotlib.pyplot as plt
from src.devNeuralNetwork import devNeuralNetwork as NeuralNet
from time import time
from IPython import embed
import pandas as pd

output_path = "../output/data/network_regression/"


class FrankeRegression:
    def __init__(self,
                gridSize=20,
                hiddenLayers = 0,
                neuronsInEachLayer = 4,
                activationFunction = 'sigmoid',
                outputFunction = 'linear',
                outputNeurons = None,
                lossFunction = 'mse',
                optimizer = 'RMSProp',
                epochs = None,
                batchSize = None,
                nrMinibatches = None,
                eta = 0.01,
                lmbda = 0.0,
                testSize = 0.2,
                terminalUpdate = False):


        space = np.linspace(0,1,gridSize)
        self.xx, self.yy = np.meshgrid(space,space)

        self.zz = self.FrankeFunction(self.xx, self.yy)
        self.zzr = self.zz.ravel()
        noise = 0.1 * np.random.randn(len(self.zzr))
        self.zzr += noise

        self.FrankeX = np.zeros((len(self.zzr), 2))
        self.FrankeX[:,0] = self.xx.ravel()
        self.FrankeX[:,1] = self.yy.ravel()
        self.FrankeY = self.zzr[:, np.newaxis]
        self.Net = NeuralNet(self.FrankeX, self.FrankeY, hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputFunction=outputFunction, outputNeurons=outputNeurons, lossFunction=lossFunction,
        optimizer=optimizer, epochs=epochs, batchSize=batchSize, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize,
        terminalUpdate = terminalUpdate)

    def __str__(self):
        self.finalTestLoss()
        self.finalTrainLoss()
        print(self.Net)

        stringToReturn = f"Train loss:  {self.trainLoss:.2f}\n"
        stringToReturn += f"Test loss:  {self.testLoss:.2f}\n"
        return stringToReturn

    def train(self):
        # print(self.Net)
        t0 = time()
        self.Net.train()
        t1 = time()
        print(f"Training time: {t1-t0:.2f} s")

    def finalTestLoss(self):
        self.testLoss = np.mean(self.Net.lossFunction(self.Net(self.Net.testData), self.Net.testOut))
        return self.testLoss

    def finalTrainLoss(self):
        self.trainLoss = np.mean(self.Net.lossFunction(self.Net(self.Net.trainData), self.Net.trainOut))
        return self.trainLoss

    def predict(self, X=None):
        data = X or self.FrankeX
        self.prediction = self.Net(data)
        return self.prediction

    def FrankeFunction(self, x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def plot(self, show=True):
        fig = plt.figure(figsize=(15,15))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.xx, self.yy, np.reshape(self.prediction, (20,20)), cmap="coolwarm", alpha=0.7)
        scatter = ax.scatter(self.xx,self.yy,self.zzr, color="green")
        plt.show()


def EtaLambdaAnalysis(filename):
    #   Fixed parameters
    hiddenLayers = 1
    neuronsInEachLayer=5
    outputNeurons=1
    activationFunction = 'sigmoid'
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'

    #   Parameters to test
    etas = np.logspace(-9,0,10)
    lmbdas = np.logspace(-9,0,10)

    #   Paramteres to find
    mse = np.zeros((len(lmbdas), len(etas)))

    #   Testing loop
    for i, lmbda in enumerate(lmbdas):
        for j, eta in enumerate(etas):
            print(f"Lmbda: {lmbda}\nEta: {eta}")
            Freg = FrankeRegression(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Freg.train()
            print(Freg)
            print("\n\n\n\n")
            mse[i,j] = Freg.finalTestLoss()
    idx = [r"$10^{%.0f}$"%k for k in np.log10(lmbdas)]
    cols = [r"$10^{%.0f}$"%k for k in np.log10(etas)]
    dF = pd.DataFrame(mse, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")

def LayerNeuronsAnalysis(filename):
    #   Fixed parameters
    eta = 1e-2  #FIXME
    lmbda = 1e-3   #FIXME
    outputNeurons=1
    activationFunction = 'sigmoid'
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'

    #   Parameters to test
    layers = np.arange(10)
    neurons = np.arange(1,11)*5

    #   Parameters to find
    mse = np.zeros((len(layers), len(neurons)))

    #   Test loop
    for i, layer in enumerate(layers):
        for j, neuron in enumerate(neurons):
            print(f"Layer: {layer}\nNeuron: {neuron}")
            Freg = FrankeRegression(hiddenLayers=layer, neuronsInEachLayer=neuron, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Freg.train()
            print(Freg)
            print("\n\n\n\n")
            mse[i,j] = Freg.finalTestLoss()
    idx = [r"{%.0f}"%k for k in layers]
    cols = [r"{%.0f}"%k for k in neurons]
    dF = pd.DataFrame(mse, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")

def activationFunctionPerEpochAnalysis(filename):
    #   Fixed parameters
    hiddenLayers = 1    #FIXME
    neuronsInEachLayer=5    #FIXME
    eta = 0.01
    lmbda = 1e-5
    outputNeurons=1
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'

    #   Parameters to test
    activationFunctions = ["sigmoid", "relu", "relu*", "tanh"]

    dF = pd.DataFrame()
    for function in activationFunctions:
        Freg = FrankeRegression(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=function, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, nrMinibatches=nrMinibatches, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
        print(Freg)
        Freg.Net.train(extractInfoPerXEpoch=1)
        loss = np.asarray(Freg.Net.testLossPerEpoch)
        epochslist = np.asarray(Freg.Net.lossEpochs)
        dF[function] = loss
    dF["epochs"] = epochslist
    dF.to_pickle(output_path+filename+".pkl")

def EpochMinibatchAnalysis(filename):
    #   Fixed parameters
    hiddenLayers = 1    #FIXME
    neuronsInEachLayer=5    #FIXME
    eta = 0.01  #FIXME
    lmbda = 1e-5    #FIXME
    outputNeurons=1
    activationFunction = 'tanh'
    epochs=250
    nrMinibatches=5
    testSize=0.2
    optimizer='RMSProp'

    #   Parameters to test
    epoch_array = np.linspace(100, 1000, 10, dtype="int")
    minibatch_array = np.linspace(5, 50, 10, dtype="int")

    #  Parameters to find
    mse = np.zeros((len(epoch_array), len(minibatch_array)))

    #   Test loop
    for i, epoch in enumerate(epoch_array):
        for j, minibatch in enumerate(minibatch_array):
            print(f"Epoch: {epoch}\nMinibatch: {minibatch}")
            Freg = FrankeRegression(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epoch, nrMinibatches=minibatch, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Freg.train()
            print(Freg)
            print("\n\n\n\n")
            mse[i,j] = Freg.finalTestLoss()
    idx = [r"{%.0f}"%k for k in epoch_array]
    cols = [r"{%.0f}"%k for k in minibatch_array]
    dF = pd.DataFrame(mse, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")





        


if __name__=="__main__":
    # Freg = FrankeRegression(hiddenLayers=2, activationFunction="tanh", neuronsInEachLayer=25, outputNeurons=1, epochs=250, nrMinibatches=100, testSize=0.2, lmbda=1e-5, eta=0.001, terminalUpdate=True, optimizer='RMSProp')
    # Freg.train()
    # Freg.predict()
    # print(Freg)
    # Freg.plot()


    # EtaLambdaAnalysis("EtaLmbdaMSE")

    # LayerNeuronsAnalysis("LayerNeuron")

    # activationFunctionPerEpochAnalysis("actFuncPerEpoch")

    EpochMinibatchAnalysis("EpochMinibatch")