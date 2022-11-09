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
                optimizer = 'plain',
                epochs = None,
                batchSize = None,
                eta = 0.01,
                lmbda = 0.0,
                testSize = 0.2,
                terminalUpdate = False):

        self.activationFunction = activationFunction
        self.outputFunction = outputFunction
        self.lossFunction = lossFunction
        self.gridSize = gridSize
        self.optimizer = optimizer
        space = np.linspace(0,1,gridSize)
        self.xx, self.yy = np.meshgrid(space,space)
        # self.xx = (self.xx-np.mean(self.xx))/np.std(self.xx)
        # self.yy = (self.yy-np.mean(self.yy))/np.std(self.yy)
        self.zz = self.FrankeFunction(self.xx, self.yy)
        self.zzr = self.zz.ravel()
        self.zzr = (self.zzr-np.mean(self.zzr))/np.std(self.zzr)

        self.FrankeX = np.zeros((len(self.zzr), 2))
        self.FrankeX[:,0] = self.xx.ravel()
        self.FrankeX[:,1] = self.yy.ravel()
        self.FrankeY = self.zzr[:, np.newaxis]
        self.Net = NeuralNet(self.FrankeX, self.FrankeY, hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputFunction=outputFunction, outputNeurons=outputNeurons, lossFunction=lossFunction,
        optimizer=optimizer, epochs=epochs, batchSize=batchSize, eta=eta, lmbda=lmbda, testSize=testSize,
        terminalUpdate = terminalUpdate)

    def __str__(self):
        self.finalTestLoss()
        self.finalTrainLoss()
        print(self.Net)

        stringToReturn = f"Nr. epochs: {self.Net.epochs}\n"
        stringToReturn += f"Batch size: {self.Net.batchSize}\n"
        stringToReturn += f"Test size: {self.Net.testSize}\n"
        stringToReturn += f"Total iterations: {self.Net.Niter}\n"
        stringToReturn += f"Lambda: {self.Net.lmbda}\n"
        stringToReturn += f"Eta: {self.Net.eta}\n\n"

        stringToReturn += f"Activation function: {self.activationFunction}\n"
        stringToReturn += f"Output function: {self.outputFunction}\n"
        stringToReturn += f"Loss function: {self.lossFunction}\n"
        stringToReturn += f"Optimizer: {self.optimizer}\n\n"

        stringToReturn += f"Train loss:  {self.trainLoss:.2f}\n"
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


def EtaLambdaAnalysis(filename, activationFunction='sigmoid', neuronsInEachLayer=7, hiddenLayers=3, outputNeurons=1, epochs=2500,  batchSize=20, testSize=0.2, optimizer='RMSProp'):
    etas = np.logspace(-9,-1,9)
    lmbdas = np.logspace(-9,-1,9)
    mse = np.zeros((len(lmbdas), len(etas)))
    for i, lmbda in enumerate(lmbdas):
        for j, eta in enumerate(etas):
            print(f"Lmbda: {lmbda}\nEta: {eta}")
            Freg = FrankeRegression(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=activationFunction, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, batchSize=batchSize, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
            Freg.train()
            print(Freg)
            print("\n\n\n\n")
            mse[i,j] = Freg.finalTestLoss()
    idx = [r"$10^{%.0f}$"%k for k in np.log10(lmbdas)]
    cols = [r"$10^{%.0f}$"%k for k in np.log10(etas)]
    dF = pd.DataFrame(mse, index=idx, columns=cols)
    dF.to_pickle(output_path+filename+".pkl")

def activationFunctionPerEpochAnalysis(filename, neuronsInEachLayer=7, hiddenLayers=3, outputNeurons=1, epochs=2500, batchSize=20, eta=0, lmbda=0, testSize=0.2, optimizer="RMSProp"):
    activationFunctions = ["sigmoid", "relu", "relu*", "tanh"]
    for function in activationFunctions:
        Freg = FrankeRegression(hiddenLayers=hiddenLayers, neuronsInEachLayer=neuronsInEachLayer, activationFunction=function, outputNeurons=outputNeurons, optimizer=optimizer, epochs=epochs, batchSize=batchSize, eta=eta, lmbda=lmbda, testSize=testSize, terminalUpdate=False)
        Freg.Net.train(extractInfoPerXEpoch=50)
        loss = np.asarray(Freg.Net.testLossPerEpoch)
        epochs = np.asarray(Freg.Net.lossEpochs)
        


if __name__=="__main__":
    # Freg = FrankeRegression(hiddenLayers=3, activationFunction="sigmoid", neuronsInEachLayer=50, outputNeurons=1, epochs=1000, batchSize=50, testSize=0.2, lmbda=0.0001, eta=0.001, terminalUpdate=True, optimizer='RMSProp')
    # Freg.train()
    # Freg.predict()
    # print(Freg)
    # Freg.plot()
    EtaLambdaAnalysis("EtaLmbdaMSE")