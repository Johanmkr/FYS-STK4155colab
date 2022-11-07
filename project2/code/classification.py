import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.devNeuralNetwork import devNeuralNetwork as NN

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



if __name__=="__main__":
    #   Correlation matrix
    correlation_matrix = cancerpd.corr().round(1)
    plt.figure(figsize=(15,8))
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.show()
