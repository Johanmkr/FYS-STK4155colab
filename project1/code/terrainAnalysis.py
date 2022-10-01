from src.utils import *
from src.objects import dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('on')


def datapointsTerrain(sx=80, sy=80, tag='2'):
    from imageio.v2 import imread
    terrain1 = imread(f"../data/SRTM_data_Norway_{tag}.tif")
    z = terrain1[::sy, ::sx].astype('float64')
    Ny, Nx = np.shape(z)
    x = np.sort( np.random.rand(Nx))
    y = np.sort( np.random.rand(Ny))
    x, y = np.meshgrid(x, y)
    return x, y, z


x, y, z = datapointsTerrain()
prepper = dataPrepper(x, y, z)
prepper.split()
prepper.scale()
prepper.genereteSeveralOrders()
print(np.shape(z))
#PLOT.visualise_data(x, y, z)


TRAININGS = {} 
PREDICTIONS = {}

POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))

HYPERPARAMS = np.logspace(-6, -1)


def assessModelComplexityBS(B, method, mode):
    Bootstrappings = []
    for d in POLYDEGS:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], B, method=method, mode=mode)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    return Bootstrappings

def assessHyperParamBS(polydeg, B, method, mode):
    Bootstrappings = []
    for lmbda in HYPERPARAMS:
        BS = Bootstrap(TRAININGS[polydeg], PREDICTIONS[polydeg], B, method=method, mode=mode, hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    return Bootstrappings

def assessHyperParamCV(polydeg, k, method, mode):
    Crossvalidations = []
    for lmbda in HYPERPARAMS:
        CV = CrossValidation(TRAININGS[polydeg], PREDICTIONS[polydeg], k, method=method, mode=mode, hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)
    return Crossvalidations

def ptF():
    # bootstrap  OLS

    Bootstrappings = assessModelComplexityBS(50, 'OLS', 'own')
    PLOT.Hastie_MSE(Bootstrappings)



if __name__ == '__main__':
    ptF()