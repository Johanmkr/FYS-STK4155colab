import numpy as np

class Layer:
    def __init__(self, 
                neurons = None,
                activationFunction = None,
                inputData = None):
        self.neurons = neurons
        if inputData is not None:
            self.h = np.asarray(inputData)  #check dimensionality perhaps
            self.neurons = len(self.h)  #This require self.h to be one-dimensional
        self.h = np.zeros(self.neurons)
        self.bias = np.zeros(self.neurons)   #how do we initialise this
        self.g = activationFunction

    def set_h(self, h):
        self.h = h 
    
    def set_g(self, g):
        self.g = g

    def set_bias(self, bias):
        self.bias = bias

