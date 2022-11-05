import numpy as np

class ExtendAndCollapse:
    def __init__(self, layers):
        self.layers = layers
        self.layersToUse = len(self.layers) - 1
        self.maxNeurons = max([self.layers])
        self.weights = 


        #This needs a lot more work. 