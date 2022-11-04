import numpy as np

class DataHandler:
    def __init__(self,
                data=None):
        self.data = data
        self.Zscale()

    def setScalingParams(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def Zscale(self):
        self.setScalingParams(np.mean(self.data), np.std(self.data))
        self.dataScaled = (self.data - self.mu)/self.sigma

    def __call__(self):
        return self.dataScaled