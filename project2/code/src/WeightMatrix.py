import numpy as np

class WeightMatrix:
    def __init__(self, 
                neuronsIn = None,
                neuronsOut = None):
        self.neuronsIn = neuronsIn 
        self.neuronsOut = neuronsOut
        self.size = (neuronsIn, neuronsOut)
        self.w = np.random.normal(size=self.size).T #Transpose since numpy treats matrices wierdly

    # def setRandom(self):
    #     self.w = np.random.normal(size=self.size)