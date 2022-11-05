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
            #   Only hidden layers and output layers need weights and biases
            self.bias = np.ones((1,self.neurons)) * 0.01 #how do we initialise this
            self.w = None
        # self.a = np.zeros((1,self.neurons))     #   activators W.T @ h + b
        self.a = None
        # self.delta = np.zeros((1,self.neurons))
        self.delta = None
        self.g = activationFunction or self.g

    def setWmatrix(self, neuronsIn, neuronsOut):
        self.wsize = (neuronsIn, neuronsOut)
        self.w = np.random.normal(size=self.wsize).T #Transpose since numpy treats matrices wierdly

    def UpdateLayer(self,
                    inputs = None,
                    features = None):
        self.__init__(inputs=inputs, features=features)

    

