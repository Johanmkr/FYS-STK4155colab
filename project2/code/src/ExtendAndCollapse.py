import numpy as np
from Layer import Layer

class ExtendAndCollapse:
    def __init__(self, layers:list[Layer]) -> None:
        """Sets parameters for and initialises the full weight, bias, delta and h matrices. Makes sure that equal index h correspond to the previous layer compared to weight, bias and delta

        Args:
            layers (list): Contains all the layer objects. 
        """
        self.layers = layers
        self.layersToUse = len(self.layers) - 1
        self.maxNeurons = max([self.layers[i].neurons for i in range(len(self.layers))])
        self.nInputs, self.nFeatures = self.layers[-1].h.shape
        self.weights = []
        self.biases = []
        self.hs = []
        self.deltas = []
        for layer in self.layers[1:]:
            self.weights.append(layer.w)
            self.biases.append(layer.bias)
            self.deltas.append(layer.delta)
        for layer in self.layers[:-1]:
            self.hs.append(layer.h)
        
        #   Make extended matrices
        self.W = np.empty((self.layersToUse, self.maxNeurons, self.maxNeurons))
        self.B = np.empty((self.layersToUse, 1, self.maxNeurons))
        self.H = np.empty((self.layersToUse, self.nInputs, self.maxNeurons))
        self.D = np.empty((self.layersToUse, self.nInputs, self.maxNeurons))

        #   Fill extended matrices with values
        for i in range(self.layersToUse):
            self.W[i,:,:] = self.extendWeight(self.weights[i])
            self.B[i,:,:] = self.extendBias(self.biases[i])
            self.H[i,:,:] = self.extendHDelta(self.hs[i])
            self.D[i,:,:] = self.extendHDelta(self.deltas[i])

    def extendWeight(self, weight):
        """Extends each weight matrix to the largest common dimension in the network. Empty entries are filled with nan-values.

        Args:
            weight (ndarray): Weight matrix 

        Returns:
            ndarray: Extended weight matrix
        """
        returnWeight = np.empty((self.maxNeurons,self.maxNeurons))
        returnWeight.fill(np.nan)
        returnWeight[:weight.shape[0], :weight.shape[1]] = weight
        return returnWeight

    def extendBias(self, bias):
        """Extend each bias vector to the largest common dimension in the network. Empty entries are filled with nan-values. 

        Args:
            bias (ndarray): Bias vector

        Returns:
            ndarray: Extended bias vector
        """
        returnBias = np.empty((1,self.maxNeurons))
        returnBias.fill(np.nan)
        returnBias[:bias.shape[0], :bias.shape[1]] = bias
        return returnBias


    def extendHDelta(self, hdelta):
        """Extends each delta (error) array to the largest common dimension in the network. Empty entries are filled with nan-values. 

        Args:
            hdelta (ndarray): Delta array

        Returns:
            ndarray: Extended delta array
        """
        returnHDelta = np.empty((self.nInputs,self.maxNeurons))
        returnHDelta.fill(np.nan)
        returnHDelta[:hdelta.shape[0], :hdelta.shape[1]] = hdelta 
        return returnHDelta

    def collapseWeights(self):
        """Collapse the full weight matrix W back into individual weight matrices for each layer, removing potential nan-vales in order to obtain original dimensionality.

        Returns:
            list: List of weight objects
        """
        returnWeights = []
        for i in range(self.layersToUse):
            try:
                returnWeights.append(self.W[i][np.isfinite(self.W[i])].reshape(self.weights[i].shape))
            except ValueError:
                pass
                # embed()
        return returnWeights

    def collapseBiases(self):
        """Collapse the full bias matrix B back into individual bias vectors for each layer, removing potential nan-vales in order to obtain original dimensionality.

        Returns:
            list: List of bias objects
        """
        returnBiases = []
        for i in range(self.layersToUse):
            returnBiases.append(self.B[i][np.isfinite(self.B[i])].reshape(self.biases[i].shape))
        return returnBiases

    def regGradW(self, idx=None):
        if idx is not None:
            return np.transpose(self.D[:,idx,:], axes=(0,2,1)) @ self.H[:,idx,:]
        else:
            return np.transpose(self.D, axes=(0,2,1)) @ self.H

    def regGradB(self, idx=None):
        if idx is not None:
            return np.sum(self.D[:,idx,:], axis=1, keepdims=True)
        else:
            return np.sum(self.D, axis=1, keepdims=True)