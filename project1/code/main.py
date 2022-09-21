from src.utils import *

from src.designMatrix import DesignMatrix
from src.parameterVector import ParameterVector
from src.Regression import LinearRegression
from src.Resampling import Bootstrap

import plot as PLOT
PLOT.init('off')



testSize = 1/5

Nx, Ny = 20, 20
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


noise = lambda eta: eta*np.random.randn(Ny, Nx)

eta = 0.01
z = FrankeFunction(x, y) + noise(eta)

print(f'\n   Franke function z with noise of stdv. {eta}.\n')

def ptB():

    polydegs = range(1,5+1) 
    Trainings = []
    Predictions = [] 

    main_polydeg = 5

    for j, n in enumerate(polydegs):
        dM = DesignMatrix(n)
        dM.createX(x, y)
            
        reg = LinearRegression(z, dM)
        #reg.scale()
        trainer, predictor = reg.split()
        #trainer.scale()
        #predictor.scale()
        beta = trainer.train() 
        trainer.fit()
        
        trainer.computeExpectationValues()

        predictor.setOptimalbeta(beta)
        predictor.fit()
        predictor.computeExpectationValues()

        Trainings.append(trainer)
        Predictions.append(predictor)

        if n == main_polydeg:
            beta = reg()
            reg.setOptimalbeta(beta)
            reg.fit()
            REG5 = reg

    PLOT.ptB_franke_funcion(x, y, REG5, show=True)

    PLOT.ptB_scores(Trainings, Predictions, pdf_name='scores', show=True)

    PLOT.ptB_beta_params(Trainings, pdf_name='betas', show=True)


def ptC():

    polydegs = range(1,20+1)

    Trainings = []
    Predictions = [] 

    for n in polydegs:
        dM = DesignMatrix(n)
        dM.createX(x, y)
        reg = LinearRegression(z, dM)

        trainer, predictor = reg.split()
        #trainer.scale()
        #predictor.scale()
        beta = trainer.train() 
        trainer.fit()
        trainer.computeExpectationValues()

        predictor.setOptimalbeta(beta)
        predictor.fit()
        predictor.computeExpectationValues()

        Trainings.append(trainer)
        Predictions.append(predictor)


    PLOT.ptC_Hastie(Trainings, Predictions, pdf_name='Hastie', show=True)


    Bootstrappings = []
    for train, test in zip(Trainings, Predictions):
        BS = Bootstrap(train, test)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    PLOT.ptC_tradeoff(Bootstrappings, pdf_name='tradeoff', show=True)





all_pts = ['B', 'C']

def runparts(parts):
    pts = []
    for part in parts:
        pt = part.strip().upper().replace('PT', '')
        if pt == 'ALL':
            pts = all_pts
            break
        else:
            assert pt in all_pts
            pts.append(pt)

    for pt in pts:
        eval(f'pt{pt}()')

try:
    dummy = sys.argv[1]
    parts = sys.argv[1:]
except IndexError:
    parts = input('What parts? ').replace(',', ' ').split()

runparts(parts)