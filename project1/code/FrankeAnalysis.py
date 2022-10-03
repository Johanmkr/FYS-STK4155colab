
from src.utils import *
from src.objects import dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('off')

PLOT.add_path('Franke')

def datapointsFranke(eta=.01, N=20):
    x = np.sort( np.random.rand(N))
    y = np.sort( np.random.rand(N))
    x, y = np.meshgrid(x, y)
    noise = eta*np.random.randn(N, N)
    z = FrankeFunction(x, y) + noise
    return x, y, z

eta = 0.1
NN = 20
x, y, z = datapointsFranke(eta, NN)
prepper = dataPrepper(x, y, z)
prepper.prep()


TRAININGS = {}
PREDICTIONS = {}

POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))


HYPERPARAMS = np.logspace(-4, -1, 10)


def ptB():

    polydegs = range(1, 6)
    workouts = {d:deepcopy(TRAININGS[d]) for d in polydegs}
    forecasts = {d:deepcopy(PREDICTIONS[d]) for d in polydegs}

    for d in range(1, 6):
        T, P = workouts[d], forecasts[d]

        reg = linearRegression(T, P, mode='own', method='ols')
        reg.fit()
        T.computeModel()
        P.predict()
        T.computeExpectationValues()
        P.computeExpectationValues()

    PLOT.train_test_MSE_R2(workouts, forecasts, show=True)

    PLOT.beta_params(workouts, show=True) 

def ptC():
    Bootstrappings = []
    for d in POLYDEGS:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], 200)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    PLOT.train_test_MSE(Bootstrappings, '_BS', show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)

    d = 8
    idx = d-1
    PLOT.hist_resampling(Bootstrappings[idx], 'mse', show=True)
    PLOT.hist_resampling(Bootstrappings[idx], 'beta', show=True)


def ptD():
    Crossvalidations = []
    for d in POLYDEGS:
        CV = CrossValidation(TRAININGS[d], PREDICTIONS[d], 8)
        CV()
        Crossvalidations.append(CV)

    PLOT.train_test_MSE(Crossvalidations, '_CV', show=True)

def ptE():
    d = 8 # or 4, 5

    trainer = TRAININGS[d]
    predictor= PREDICTIONS[d]

    Bootstrappings = []

    for lmbda in HYPERPARAMS:
        BS = Bootstrap(trainer, predictor, 200, method='Ridge', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.train_test_MSE(Bootstrappings, '_BS', show=True)


    Crossvalidations = []

    for lmbda in HYPERPARAMS:
        CV = CrossValidation(trainer, predictor, 10, method='Ridge', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.train_test_MSE(Crossvalidations, '_CV', show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)

def ptF():
    d = 8 # or 4, 5
    trainer = TRAININGS[d]
    predictor= PREDICTIONS[d]

    Bootstrappings = []

    for lmbda in HYPERPARAMS:
        BS = Bootstrap(trainer, predictor, 2, method='Lasso', mode='skl', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.train_test_MSE(Bootstrappings, '_BS', show=True)

    Crossvalidations = []

    for lmbda in HYPERPARAMS:
        CV = CrossValidation(trainer, predictor, 5, method='Lasso', mode='skl', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.train_test_MSE(Crossvalidations, '_CV', show=True)
    PLOT.BV_Tradeoff(Bootstrappings, show=True)






if __name__ == '__main__':

    Franke_pts = ['B', 'C', 'D', 'E', 'F']

    def runparts(parts, all_pts=Franke_pts):
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




PLOT.update_info()

with open('README.md', "a") as file:
    file.write('\n')
    file.write('## Additional information:\n\n')
    file.write(f'xy-grid: N x N = {NN} x {NN} \n')
    file.write(f'noise level: η = {eta}\n\n')
    file.write(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).\n')
    file.write(f'Considered {len(HYPERPARAMS)} λ-values between λ = {HYPERPARAMS[0]} and λ = {HYPERPARAMS[-1]} (logarithmically spaced).\n')


