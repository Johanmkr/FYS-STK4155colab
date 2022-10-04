
from src.utils import *
from src.objects import dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('off')

PLOT.add_path('Franke')

def datapointsFranke(eta=.1, N=20):
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
goto_B = 400 # no. of bootraps
goto_k = 8  # no. of folds

def noneAnalysis():
    # Data with noise
    x, y, z = datapointsFranke(0.1, NN)
    prepper1 = dataPrepper(x, y, z)
    prepper1.prep() 
    # Data with no noise
    x, y, z = datapointsFranke(0, NN)
    prepper2 = dataPrepper(x, y, z)
    prepper2.prep()
    
    angles = (28, -20)
    PLOT.visualise_data(*prepper1.dump(), angles=angles, show=show, mark="$η=0.1$")
    PLOT.visualise_data(*prepper2.dump(), angles=angles, tag='_no_noise', show=show, mark="$η=0$")

def olsAnalysis():

    print('\nPerforming analysis using Ordinary least squares\n')
    print('-'*40)
    print('\n')

    def noise_effect():
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

    def simple_analysis():

        polydegs = range(1, 5+1)
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

    def bootstrap_analysis():
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

    def cv_analysis():

        Crossvalidations = []
        for d in POLYDEGS:
            CV = CrossValidation(TRAININGS[d], PREDICTIONS[d], goto_k)
            CV()
            Crossvalidations.append(CV)

        PLOT.train_test_MSE(Crossvalidations,  show=show)
    
    noise_effect()
    simple_analysis()
    print('   > Bootstrap ...\n')
    bootstrap_analysis()
    print('   > k-fold cross-validation ...\n')
    cv_analysis()

def ridgeAnalysis():

    print('\nPerforming analysis using Ridge regression\n')
    print('-'*40)
    print('\n')


    trainer = TRAININGS[goto_polydeg]
    predictor= PREDICTIONS[goto_polydeg]

    def bootstrap_analysis():
        Bootstrappings = []

        for lmbda in HYPERPARAMS:
            BS = Bootstrap(trainer, predictor, goto_B, scheme='Ridge', hyper_param=lmbda)
            BS()
            BS.bias_varianceDecomposition()
            Bootstrappings.append(BS)
        
        PLOT.train_test_MSE(Bootstrappings, show=show)
        PLOT.BV_Tradeoff(Bootstrappings, show=show)

        # assess one choice:
        idx = np.argmin(np.abs(HYPERPARAMS-1e-3))
        PLOT.mse_hist_resampling(Bootstrappings[idx],  show=show)
        PLOT.beta_hist_resampling(Bootstrappings[idx], show=show)

    def cv_analysis():
        Crossvalidations = []

        for lmbda in HYPERPARAMS:
            CV = CrossValidation(trainer, predictor, goto_k, scheme='Ridge', hyper_param=lmbda)
            CV()
            Crossvalidations.append(CV)

        PLOT.train_test_MSE(Crossvalidations, show=show)
    
    print('   > Bootstrap ...\n')
    bootstrap_analysis()
    print('   > k-fold cross-validation ...\n')
    cv_analysis()

def lassoAnalysis():

    print('\nPerforming analysis using Lasso regression\n')
    print('-'*40)
    print('\n')

    trainer = TRAININGS[goto_polydeg]
    predictor= PREDICTIONS[goto_polydeg]

    def bootstrap_analysis():
        Bootstrappings = []

        for lmbda in HYPERPARAMS[10:]:
            print('----')
            BS = Bootstrap(trainer, predictor, 10, scheme='Lasso', mode='skl', hyper_param=lmbda)
            BS()
            BS.bias_varianceDecomposition()
            Bootstrappings.append(BS)
        
        PLOT.train_test_MSE(Bootstrappings, show=show)
        PLOT.BV_Tradeoff(Bootstrappings, show=show)
    
    def cv_analysis():
        Crossvalidations = []

        for lmbda in HYPERPARAMS[10:]:
            CV = CrossValidation(trainer, predictor, goto_k, scheme='Lasso', mode='skl', hyper_param=lmbda)
            CV()
            Crossvalidations.append(CV)

        PLOT.train_test_MSE(Crossvalidations, show=show)
    
    print('   > Bootstrap ...\n')
    bootstrap_analysis()
    print('   > k-fold cross-validation ...\n')
    cv_analysis()
    



if __name__ == '__main__':

    all_parts = ['ols', 'ridge', 'lasso', 'none'] 

    def runparts(parts):
        pts = []
        for part in parts:
            pt = part.strip().lower().replace('analysis', '')
            if pt == 'ALL':
                pts = parts
                break
            else:
                assert pt in all_parts
                pts.append(pt)

        for pt in pts:
            eval(f'{pt}Analysis()')

    try:
        dummy = sys.argv[1]
        parts = sys.argv[1:]
    except IndexError:
        parts = input('OLS, Ridge or Lasso? ').replace(',', ' ').split()

    runparts(parts)


    additionalInfo = [f'xy-grid: N x N = {NN} x {NN}']
    additionalInfo.append(f'noise level: η = {eta}')
    additionalInfo.append(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).')
    additionalInfo.append(f'Considered {len(HYPERPARAMS)} λ-values between λ = {HYPERPARAMS[0]:.1e} and λ = {HYPERPARAMS[-1]:.1e} (logarithmically spaced).')




    PLOT.update_info(additionalInfo)

