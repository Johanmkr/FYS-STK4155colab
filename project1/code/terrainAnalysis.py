from src.utils import *
from src.objects import dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('off')

PLOT.add_path('terrain')


def datapointsTerrain(sx=40, sy=80, tag='2'):
    from imageio.v2 import imread
    terrain1 = imread(f"../data/SRTM_data_Norway_{tag}.tif")
    z = terrain1[::sy, ::sx].astype('float64')
    Ny, Nx = np.shape(z)
    x = np.sort( np.random.rand(Nx))
    y = np.sort( np.random.rand(Ny))
    x, y = np.meshgrid(x, y)
    return x, y, z


x, y, z = datapointsTerrain()
Ny, Nx = np.shape(x)
prepper = dataPrepper(x, y, z)
prepper.prep()



#PLOT.visualise_data(x, y, z)


TRAININGS = {} 
PREDICTIONS = {}

POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))

print(TRAININGS[15].dM)

HYPERPARAMS = np.logspace(-6, -1, 10)

goto_polydeg = 8
goto_B = 100
goto_k = 5

show = False

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
    Bootstrappings = assessModelComplexityBS(goto_B, 'OLS', 'own')
    PLOT.train_test_MSE(Bootstrappings, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)

    d = 8
    idx = 8-1
    PLOT.hist_resampling(Bootstrappings[idx], 'mse', show=show)
    PLOT.hist_resampling(Bootstrappings[idx], 'beta', show=show)


    # d)
    Crossvalidations = assessModelComplexityCV(goto_k, 'OLS', 'own')
    PLOT.train_test_MSE(Crossvalidations, show=show)

    # e)
    polydeg = 8
    Bootstrappings = assessHyperParamBS(polydeg, goto_B, 'Ridge', 'own')
    PLOT.train_test_MSE(Bootstrappings, show=show)

    Crossvalidations = assessHyperParamCV(polydeg, goto_k, 'Ridge', 'own')
    PLOT.train_test_MSE(Crossvalidations, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)

    
    # f)
    '''polydeg = 8
    Bootstrappings = assessHyperParamBS(polydeg, 50, 'Lasso', 'skl')
    PLOT.train_test_MSE(Bootstrappings, show=True)

    Crossvalidations = assessHyperParamCV(polydeg, 8, 'Lasso', 'skl')
    PLOT.train_test_MSE(Crossvalidations, show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)'''

   



if __name__ == '__main__':
    ptG()

    additionalInfo = []
    additionalInfo.append(f'xy-grid: (Nx) x (Ny) = {Nx} x {Ny}')
    additionalInfo.append(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).')
    additionalInfo.append(f'Considered {len(HYPERPARAMS)} λ-values between λ = {HYPERPARAMS[0]:.1e} and λ = {HYPERPARAMS[-1]:.1e} (logarithmically spaced).')

    PLOT.update_info(additionalInfo)