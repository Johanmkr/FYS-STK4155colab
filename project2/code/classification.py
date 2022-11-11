import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.devNeuralNetwork import devNeuralNetwork as NeuralNet
from IPython import embed

from sklearn.model_selection import train_test_split as splitter
from sklearn.datasets import load_breast_cancer
import pandas as pd

np.random.seed(69)
"""Sort out dataset"""



Creg = NeuralNet(inputs, outputs[:,np.newaxis], hiddenLayers=1, neuronsInEachLayer=5, activationFunction='tanh', outputFunction='sigmoid', outputNeurons=1, lossFunction="crossentropy", optimizer="RMSProp", epochs=250, batchSize=10, eta=0.01, lmbda=1e-5, testSize=0.2, terminalUpdate=True, classification=True)

print(Creg)
Creg.train()




output_path = "../output/data/network_regression/"


class CancerData:
    def __init__(self,
                hiddenLayers = 0,
                neuronsInEachLayer = 4,
                activationFunction = 'sigmoid',
                outputFunction = 'sigmoid',
                outputNeurons = None,
                lossFunction = 'crossentropy',
                optimizer = 'RMSProp',
                epochs = None,
                batchSize = None,
                nrMinibatches = None,
                eta = 0.01,
                lmbda = 0.0,
                testSize = 0.2,
                terminalUpdate = False):

        cancer = load_breast_cancer()
        cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        inputs = cancer.data 
        outputs = cancer.target 
        labels = cancer.feature_names[0:30]
        correlation_matrix = cancerpd.corr().round(1)

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

if __name__=="__main__":
    #   Correlation matrix
    # plt.figure(figsize=(15,8))
    # sns.heatmap(data=correlation_matrix, annot=True)
    # plt.show()
    # Creg = 
    # for i, val in enumerate(Creg()):
    #     print(f"{val[0]:.0f}  -  {Creg.testOut[i]}")
    # embed()
    pass
