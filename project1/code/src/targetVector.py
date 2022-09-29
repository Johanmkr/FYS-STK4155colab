
from src.utils import *




class TargetVector:

    def __init__(self, z):
        self.shape = z.shape
        self.z = z.ravel()
        self.data = z
        self.scaled = False

    def standardScaling(self):
        self.z_org = self.z.copy() # save original z
        self.z -= np.mean(self.z)
        self.z /= np.std(self.z_org)
        self.scaled = True

    def __len__(self):
        return len(self.data.ravel())

    def getVector(self):
        return self.z

    def newObject(self, z, is_scaled=None):
        newobject = TargetVector(z)
        newobject.scaled = is_scaled or self.scaled
        return newobject

    def __getitem__(self,i):
        return self.z[i]

    def getScalingParameters(self):

        if not self.scaled:
            self.mu = np.mean(self.z)
            self.sigma = np.std(self.z)
            return self.mu, self.sigma

        else:
            return None, None