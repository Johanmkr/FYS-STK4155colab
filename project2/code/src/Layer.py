import numpy as np

class Layer:
    def __init__(self, 
                neurons:int = None,
                activationFunction:function = None,
                inputs:int = None,
                features:int = None) -> None:

        if inputs and features is not None:
            self.neurons = features
            self.h = np.zeros((inputs, self.neurons))
        else:
            self.neurons = neurons
            self.h = None
            #   Only hidden layers and output layers need weights and biases
            self.bias = np.ones((1,self.neurons)) * 0.01 #how do we initialise this
            self.w = None
        self.a = None 
        self.delta = None
        self.g = activationFunction or self.g

    def setWmatrix(self, neuronsIn:int, neuronsOut:int, initialisation:str="normal") -> None:
        """Initialised the weight matrices based on either the normal, Xavier, and He initialisation scheme. 

        Args:
            neuronsIn (int): Neurons in previous layer.
            neuronsOut (int): Neurons in current layer.
            initialisation (str, optional): Initialisation scheme. Defaults to "normal".
        """
        self.wsize = (neuronsIn, neuronsOut)
        if initialisation == "normal": 
            self.w = np.random.normal(size=self.wsize).T    #Transpose since numpy treats matrices wierdly
        elif initialisation in ["xavier", "Xavier", "glorot", "Glorot"]:
            limit = np.sqrt(2 / float(neuronsIn + neuronsOut))
            self.w = np.random.normal(0.0, limit, size=(neuronsIn, neuronsOut)).T
        elif initialisation in ["He", "he", "kaiming", "Kaimin", "MSRA", "msra"]:
            limit = np.sqrt(2/float(neuronsIn))
            self.w = np.random.normal(0.0, limit, size=(neuronsIn, neuronsOut)).T

    def UpdateLayer(self,
                    inputs:int = None,
                    features:int = None) -> None:
        self.__init__(inputs=inputs, features=features)

    

