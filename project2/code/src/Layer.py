import numpy as np

class Layer:
    def __init__(self, 
                neurons = None,
                activationFunction = None,
                inputs = None,
                features = None):
        # if inputData is not None:

        #     self.h = np.asarray(inputData)  #check dimensionality perhaps
        #     self.neurons = len(self.h)  #This require self.h to be one-dimensional
        # else:
        #     self.neurons = neurons
        #     self.h = np.zeros(self.neurons)
        if inputs and features is not None:
            self.neurons = features
            self.h = np.zeros((inputs, self.neurons))
        else:
            self.neurons = neurons
            self.h = None
        # self.a = np.zeros((1,self.neurons))     #   activators W.T @ h + b
        self.a = None
        self.bias = np.ones((1,self.neurons))  #how do we initialise this
        # self.delta = np.zeros((1,self.neurons))
        self.delta = None
        self.g = activationFunction or self.g

    def UpdateLayer(self,
                    inputs = None,
                    features = None):
        self.__init__(inputs=inputs, features=features)

    

