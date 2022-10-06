
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
maxPolydeg = 14
POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))

POLYDEGS = POLYDEGS[:-2]


HYPERPARAMS_R = np.logspace(-6, -2, 20)  # for Ridge (polydeg 5)
<<<<<<< HEAD
HYPERPARAMS_L = np.logspace(-6, -2, 20)  # for Lasso
kLIST = range(5,11)
=======
HYPERPARAMS_L = np.logspace(-6, -2, 10)  # for Lasso

show = True

>>>>>>> fff31bef574165ca70a041e81387288230d4cd0f
goto_polydeg = 5 # 
goto_B = 400 # no. of bootraps
goto_k = 8  # no. of folds


show = True

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
    




ref_MSE = {'simple':0.14848043170413214, 'BS':0.16646991921908658, 'CV':0.17656214912552667}

def finalAnalysis():

    d = 5
    scheme = 'OLS'
    lmbda = 0

    trainer, predictor = TRAININGS[d], PREDICTIONS[d]

    reg = linearRegression(trainer, predictor, mode='own', scheme=scheme, shrinkage_parameter=lmbda)
    reg.fit()
    predictor.predict()
    PLOT.compare_data(predictor, predictor, angles=(20, 30), show=show, mark='prediction set')

    BS = Bootstrap(trainer, predictor, goto_B, scheme=scheme, mode='own', hyper_param=lmbda)
    BS()
    MSE_BS = BS.resamplingError()


    CV = CrossValidation(trainer, predictor, goto_k, scheme=scheme, mode='own', hyper_param=lmbda)
    CV()
    MSE_CV = CV.resamplingError()

    MSE_un = predictor.mean_squared_error()
    
    print(MSE_un, MSE_BS, MSE_CV) # This could have been done smarter, but now I am running out of time (and space on my machine)




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
        print('   > Bootstrap ...\n')
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
        print('   > k-fold cross-validation ...\n')

        CVgrid = []
        for i, k in enumerate([5, 6, 7, 8, 9, 10]):
            Crossvalidations = []

            for d in POLYDEGS:
                CV = CrossValidation(TRAININGS[d], PREDICTIONS[d], k)
                CV()
                Crossvalidations.append(CV)
            CVgrid.append(Crossvalidations)

        PLOT.CV_errors(CVgrid, show=show)
    
    noise_effect()
    simple_analysis()
    bootstrap_analysis()
    cv_analysis()

def ridgeAnalysis():

    print('\nPerforming analysis using Ridge regression\n')
    print('-'*40)
    print('\n')


    trainer = TRAININGS[goto_polydeg]
    predictor= PREDICTIONS[goto_polydeg]

    def bootstrap_analysis():
        print('   > Bootstrap ...\n')
        Bootstrappings = []

        for lmbda in HYPERPARAMS_R:
            BS = Bootstrap(trainer, predictor, goto_B, scheme='Ridge', hyper_param=lmbda)
            BS()
            BS.bias_varianceDecomposition()
            Bootstrappings.append(BS)
        
        PLOT.train_test_MSE(Bootstrappings, show=show)
        PLOT.BV_Tradeoff(Bootstrappings, show=show)

        # assess one choice:
        idx = np.argmin(np.abs(HYPERPARAMS_R-1e-4))
        PLOT.mse_hist_resampling(Bootstrappings[idx], show=show)
        PLOT.beta_hist_resampling(Bootstrappings[idx], show=show)

    def cv_analysis():
        print('   > k-fold cross-validation ...\n')
        CVgrid = []
        for k in kLIST[1::2]:
            Crossvalidations = []
            for lmbda in HYPERPARAMS_R:
                CV = CrossValidation(trainer, predictor, k, scheme='Ridge', hyper_param=lmbda)
                CV()
                Crossvalidations.append(CV)
            CVgrid.append(Crossvalidations)

        PLOT.CV_errors(CVgrid, show=show)

    def grid_search():
        print('   > Grid search using k-fold cross-validation ...\n')
        CVgrid = []

        for d in range(1,6):
            trainer = TRAININGS[d]
            predictor = PREDICTIONS[d]
            CV_l  = []
            for lmbda in HYPERPARAMS_R:
                CV = CrossValidation(trainer, predictor, goto_k, scheme='Ridge', hyper_param=lmbda)
                CV()
                CV_l.append(CV)

            CVgrid.append(CV_l)
        
        PLOT.heatmap(CVgrid, show=show)

    bootstrap_analysis()
    #cv_analysis()
    #grid_search()

def lassoAnalysis():

    print('\nPerforming analysis using Lasso regression\n')
    print('-'*40)
    print('\n')

    d_L = 14
    
    trainer = TRAININGS[d_L]
    predictor = PREDICTIONS[d_L]


    def bootstrap_analysis():
        print('   > Bootstrap ...\n')
        Bootstrappings = []

        for lmbda in HYPERPARAMS_L:
            BS = Bootstrap(trainer, predictor, goto_B/40, scheme='Lasso', mode='skl', hyper_param=lmbda)
            BS()
            BS.bias_varianceDecomposition()
            Bootstrappings.append(BS)
        
        PLOT.train_test_MSE(Bootstrappings, show=show)
        PLOT.BV_Tradeoff(Bootstrappings, show=show)
    
    def cv_analysis():
        print('   > k-fold cross-validation ...\n')
        CVgrid = []
        for k in kLIST[1::2]:
            Crossvalidations = []
            for lmbda in HYPERPARAMS_L[1:]:
                CV = CrossValidation(trainer, predictor, k, scheme='Lasso', mode='skl', hyper_param=lmbda)
                CV()
                Crossvalidations.append(CV)
            CVgrid.append(Crossvalidations)

        PLOT.CV_errors(CVgrid, show=show)

    def grid_search():
        print('   > Grid search using k-fold cross-validation ...\n')
        CVgrid = []

        for d in POLYDEGS[5:]:
            print(f'\nd = {d:2.0f} ...')
            trainer = TRAININGS[d]
            predictor = PREDICTIONS[d]
            CV_l  = []
            for lmbda in HYPERPARAMS_L:
                print(f'   > λ = {lmbda:.3e} ...')
                CV = CrossValidation(trainer, predictor, goto_k, scheme='Lasso', mode='skl', hyper_param=lmbda)
                CV()
                CV_l.append(CV)
            CVgrid.append(CV_l)
        
        def fmt(x):
            if np.abs(x) < 1e-10:
                s = ''
            else:
                s = r'$\text{MSE}^\text{OLS}$'
        PLOT.heatmap(CVgrid, ref_error=ref_MSE['CV'],  show=show)

    cv_analysis()
    bootstrap_analysis()
    
    #grid_search()
    



if __name__ == '__main__':

    all_parts = ['ols', 'ridge', 'lasso', 'none', 'final'] 

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
    additionalInfo.append(f'Ridge: Considered {len(HYPERPARAMS_R)} λ-values between λ = {HYPERPARAMS_R[0]:.1e} and λ = {HYPERPARAMS_R[-1]:.1e} (logarithmically spaced).')
    additionalInfo.append(f'Lasso: Considered {len(HYPERPARAMS_L)} λ-values between λ = {HYPERPARAMS_L[0]:.1e} and λ = {HYPERPARAMS_L[-1]:.1e} (logarithmically spaced).')


    PLOT.update_info(additionalInfo)

