
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
maxPolydeg = 12
POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))


HYPERPARAMS = np.logspace(-6, -2, 20)  # for Ridge at least (polydeg 5)

show = False

goto_polydeg = 5
goto_B = 400 # no bootraps
goto_k = 8  # k folds

def ptEXTRA():
    etas = np.logspace(-4, 0, 5)
    polydegs = range(1, 5+1)

    mark = "η = "
    trainings = {}
    predictions = {}
    for d in polydegs:
        trainings[d] = []
        predictions[d] = []
        
    for i, etai in enumerate(etas):
        x, y, z = datapointsFranke(etai, NN)
        prepper = dataPrepper(x, y, z)
        prepper.prep()
    
        for d in polydegs:

            T = Training(*prepper.getTrain(d))
            P = Prediction(*prepper.getTest(d))
            reg = linearRegression(T, P, mode='own', scheme='ols')
            reg.fit()
            T.computeModel()
            P.predict()
            T.computeExpectationValues()
            P.computeExpectationValues()

            trainings[d].append(T)
            predictions[d].append(P)
    
        mark += f"{etai:.1e}, "

    mark.strip().strip(',')
    PLOT.error_vs_noise(trainings, predictions, etas, show=show, mark=mark)
    

def ptB():

    polydegs = range(1, 6)
    workouts = {d:deepcopy(TRAININGS[d]) for d in polydegs}
    forecasts = {d:deepcopy(PREDICTIONS[d]) for d in polydegs}

    for d in range(1, 6):
        T, P = workouts[d], forecasts[d]

        reg = linearRegression(T, P, mode='own', scheme='ols')
        reg.fit()
        T.computeModel()
        P.predict()
        T.computeExpectationValues()
        P.computeExpectationValues()


    PLOT.train_test_MSE_R2(workouts, forecasts, show=show)

    PLOT.beta_params(workouts, show=show) 


    


def ptC():
    Bootstrappings = []
    for d in POLYDEGS:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], goto_B)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    PLOT.train_test_MSE(Bootstrappings, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)


    idx = goto_polydeg-1
    PLOT.mse_hist_resampling(Bootstrappings[idx], show=show)
    PLOT.beta_hist_resampling(Bootstrappings[idx],  show=show)


def ptD():
    Crossvalidations = []
    for d in POLYDEGS:
        CV = CrossValidation(TRAININGS[d], PREDICTIONS[d], goto_k)
        CV()
        Crossvalidations.append(CV)

    PLOT.train_test_MSE(Crossvalidations,  show=show)

def ptE():

    trainer = TRAININGS[goto_polydeg]
    predictor= PREDICTIONS[goto_polydeg]

    Bootstrappings = []

    for lmbda in HYPERPARAMS:
        BS = Bootstrap(trainer, predictor, goto_B, scheme='Ridge', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    

    
    PLOT.train_test_MSE(Bootstrappings, show=show)
    idx = np.argmin(np.abs(HYPERPARAMS-1e-3))
    PLOT.mse_hist_resampling(Bootstrappings[idx],  show=show)
    PLOT.beta_hist_resampling(Bootstrappings[idx], show=show)
    


    Crossvalidations = []

    for lmbda in HYPERPARAMS:
        CV = CrossValidation(trainer, predictor, goto_k, scheme='Ridge', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.train_test_MSE(Crossvalidations, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)

def ptF():

    trainer = TRAININGS[goto_polydeg]
    predictor= PREDICTIONS[goto_polydeg]

    Bootstrappings = []

    for lmbda in HYPERPARAMS[10:]:
        print('----')
        BS = Bootstrap(trainer, predictor, 10, scheme='Lasso', mode='skl', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.train_test_MSE(Bootstrappings, show=show)

    Crossvalidations = []

    for lmbda in HYPERPARAMS[10:]:
        CV = CrossValidation(trainer, predictor, goto_k, scheme='Lasso', mode='skl', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.train_test_MSE(Crossvalidations, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)




if __name__ == '__main__':

    Franke_pts = ['B', 'C', 'D', 'E', 'F', 'EXTRA']

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




additionalInfo = [f'xy-grid: N x N = {NN} x {NN}']
additionalInfo.append(f'noise level: η = {eta}')
additionalInfo.append(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).')
additionalInfo.append(f'Considered {len(HYPERPARAMS)} λ-values between λ = {HYPERPARAMS[0]:.1e} and λ = {HYPERPARAMS[-1]:.1e} (logarithmically spaced).')




PLOT.update_info(additionalInfo)

