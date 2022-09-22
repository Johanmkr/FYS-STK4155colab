
from src.utils import *




class TargetVector:

    def __init__(self, z):
        self.shape = z.shape
        self.z = z.ravel()
        self.data = self.z.ravel()
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