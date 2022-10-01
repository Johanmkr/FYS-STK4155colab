from src.utils import *
from src.objects import dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('off')

PLOT.add_path('terrain')


def datapointsTerrain(sx=40, sy=40, tag='2'):
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
prepper.prep()



#PLOT.visualise_data(x, y, z)


TRAININGS = {} 
PREDICTIONS = {}

POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))

HYPERPARAMS = np.logspace(-6, -1, 10)


def assessModelComplexityBS(B, method, mode):
    Bootstrappings = []
    for d in POLYDEGS:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], B, method=method, mode=mode)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    return Bootstrappings

def assessModelComplexityCV(k, method, mode):
    Crossvalidations = []
    for d in POLYDEGS:
        CV = Bootstrap(TRAININGS[d], PREDICTIONS[d], k, method=method, mode=mode)
        CV()
        Crossvalidations.append(CV)

    return Crossvalidations

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

def ptG():

    # c)
    Bootstrappings = assessModelComplexityBS(100, 'OLS', 'own')
    PLOT.train_test_MSE(Bootstrappings, '_BS', show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)

    d = 8
    idx = 8-1
    PLOT.hist_resampling(Bootstrappings[idx], 'mse', show=True)
    PLOT.hist_resampling(Bootstrappings[idx], 'beta', show=True)


    # d)
    Crossvalidations = assessModelComplexityCV(8, 'OLS', 'own')
    PLOT.train_test_MSE(Crossvalidations, '_CV', show=True)

    # e)
    polydeg = 8
    Bootstrappings = assessHyperParamBS(polydeg, 100, 'Ridge', 'own')
    PLOT.train_test_MSE(Bootstrappings, '_BS', show=True)

    Crossvalidations = assessHyperParamCV(polydeg, 8, 'Ridge', 'own')
    PLOT.train_test_MSE(Crossvalidations, '_CV', show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)

    
    # f)
    '''polydeg = 8
    Bootstrappings = assessHyperParamBS(polydeg, 50, 'Lasso', 'skl')
    PLOT.train_test_MSE(Bootstrappings, '_BS', show=True)

    Crossvalidations = assessHyperParamCV(polydeg, 8, 'Lasso', 'skl')
    PLOT.train_test_MSE(Crossvalidations, '_CV', show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)'''

   



if __name__ == '__main__':
    ptG()

    PLOT.update_info()