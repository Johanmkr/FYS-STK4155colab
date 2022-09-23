from src.utils import *

from src.designMatrix import DesignMatrix
from src.parameterVector import ParameterVector
from src.Regression import LinearRegression
from src.Resampling import Bootstrap

import plot as PLOT
PLOT.init('off')




Nx, Ny = 20, 20
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)

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
        trainer, predictor = reg.split()
        beta = trainer.train() 
        trainer.computeModel()
        trainer.computeExpectationValues()

        predictor.setOptimalbeta(beta)
        predictor.predict()
        predictor.computeExpectationValues()

        Trainings.append(trainer)
        Predictions.append(predictor)

        if n == main_polydeg:
            # FIX THIS WITH SCALING
            #FIXME
            trainer, predictor = reg.split(scale=True)
            beta = trainer.train() 

            reg.setOptimalbeta(beta)
            reg.computeModel()
            REG5 = reg

    PLOT.ptB_franke_funcion(x, y, REG5, show=True)

    PLOT.ptB_scores(Trainings, Predictions, pdf_name='scores', show=True)

    PLOT.ptB_beta_params(Trainings, pdf_name='betas', show=True)


def ptC():

    polydegs = range(1,13+1)

    Trainings = []
    Predictions = []

    for n in polydegs:
        dM = DesignMatrix(n)
        dM.createX(x, y)
        reg = LinearRegression(z, dM)

        trainer, predictor = reg.split()

        beta = trainer.train() 
        trainer.computeModel()
        trainer.computeExpectationValues()

        predictor.setOptimalbeta(beta)
        predictor.computeModel()
        predictor.computeExpectationValues()

        Trainings.append(trainer)
        Predictions.append(predictor)



    Bootstrappings = []
    for train, test in zip(Trainings, Predictions):
        BS = Bootstrap(train, test)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    # HUSK BOOTSTRAP I HASTIE


    PLOT.ptC_Hastie(Trainings, Predictions, pdf_name='Hastie', show=True)

    PLOT.ptC_tradeoff(Bootstrappings, pdf_name='tradeoff', show=True)



def ptE():
    # ....
    polydegs = range(1,20+1)

    Trainings = []
    Predictions = [] 

    for n in polydegs:
        dM = DesignMatrix(n)
        dM.createX(x, y)
        reg = LinearRegression(z, dM, 'Ridge')

        trainer, predictor = reg.split()
        beta = trainer.train(0.1) 
        trainer.computeModel()
        trainer.computeExpectationValues()

        predictor.setOptimalbeta(beta)
        predictor.computeModel()
        predictor.computeExpectationValues()

        Trainings.append(trainer)
        Predictions.append(predictor)


    Bootstrappings = []
    for train, test in zip(Trainings, Predictions):
        BS = Bootstrap(train, test)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)







all_pts = ['B', 'C', 'E']

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