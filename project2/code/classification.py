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

cancer = load_breast_cancer()
cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
inputs = cancer.data 
outputs = cancer.target 
labels = cancer.feature_names[0:30]
correlation_matrix = cancerpd.corr().round(1)

Creg = NeuralNet(inputs, outputs[:,np.newaxis], hiddenLayers=5, neuronsInEachLayer=7, activationFunction='sigmoid', outputFunction='softmax', outputNeurons=1, lossFunction="crossentropy", optimizer="RMSProp", epochs=100, batchSize=10, eta=0.01, lmbda=1e-5, testSize=0.2, terminalUpdate=True)

Creg.train()

if __name__=="__main__":
    #   Correlation matrix
    # plt.figure(figsize=(15,8))
    # sns.heatmap(data=correlation_matrix, annot=True)
    # plt.show()
    # Creg = 
    embed()
    # pass
