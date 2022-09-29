from src.utils import *


from src.objects import designMatrix, targetVector, parameterVector
from src.Regression import linearRegression
from src.Resampling import Bootstrap

import plot as PLOT
PLOT.init('on')



def datapoints(eta=.01, N=40):
    x = np.sort( np.random.rand(N))
    y = np.sort( np.random.rand(N))
    x, y = np.meshgrid(x, y)
    noise = lambda eta: eta*np.random.randn(N, N)
    z = FrankeFunction(x, y) + noise(eta)
    return x, y, z

x, y, z = datapoints(eta=0, N=20)







# print(f'\n   Franke function z with noise of stdv. {1}.\n')

def ptB():
    x, y, z = datapoints(eta=1, N=40)
    polydegs = range(1,5+1)
    Trainings = []
    Predictions = [] 

    main_polydeg = 5
    
    for j, n in enumerate(polydegs):
            
        reg = LinearRegression(z, dM)
        trainer, predictor = reg.split(scale=True)
        beta = trainer.train()
        beta.computeVariance(trainer) 
        trainer.computeModel()
        trainer.computeExpectationValues()

        predictor.setOptimalbeta(beta)
        predictor.predict()
        predictor.computeExpectationValues()

        Trainings.append(trainer)
        Predictions.append(predictor)


    # PLOT.ptB_franke_funcion_only(*datapoints(eta=0, N=40), pdf_name="franke", show=True)

    # PLOT.ptB_franke_funcion_only(*datapoints(eta=.1, N=40), pdf_name="franke_noise", show=True)

    # PLOT.ptB_scores(Trainings, Predictions, pdf_name='scores', show=True)

    PLOT.ptB_beta_params(Trainings[::-1], pdf_name='betas', show=True)


def ptC():
    x, y, z = datapoints(eta=0.01, N=20)

    polydegs = range(1,18+1)

    Trainings = []
    Predictions = []

    for n in polydegs:
        dM = DesignMatrix(n)
        dM.createX(x, y)
        reg = LinearRegression(z, dM)

        trainer, predictor = reg.split()

        Trainings.append(trainer)
        Predictions.append(predictor)


    Bootstrappings = []
    for train, test in zip(Trainings, Predictions):
        BS = Bootstrap(train, test)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)


    PLOT.ptC_Hastie(Bootstrappings,  pdf_name='Hastie', show=True)

    PLOT.ptC_tradeoff(Bootstrappings, pdf_name='tradeoff', show=True)























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