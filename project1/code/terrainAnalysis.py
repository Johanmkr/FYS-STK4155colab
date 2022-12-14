from src.utils import *
from src.objects import dataPrepper, groupedVector
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation

from imageio.v2 import imread

import plot as PLOT
PLOT.init() # arg. 'on' initiates testMode
PLOT.add_path('terrain')


def datapointsTerrain(location='GrandCanyon', sx=slice(2500, 3400, 30), sy=slice(2500, 3400, 30)):
    terrain = imread(f"../data/SRTM_data_{location}.tif")
    z = terrain[sx, sy].astype('float64')
    Ny, Nx = np.shape(z)
    x = np.sort( np.random.rand(Nx))
    y = np.sort( np.random.rand(Ny))
    x, y = np.meshgrid(x, y)
    return x, y, z


x, y, z = datapointsTerrain()
Ny, Nx = np.shape(x)
prepper = dataPrepper(x, y, z)
prepper.prep(True)


TRAININGS = {}
PREDICTIONS = {}

POLYDEGS = range(1, maxPolydeg+1)

# nice params
goto_polydeg = 6 
goto_B = 400
goto_k = 10
kLIST = [6, 8, 10]


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))



### OLS:
POLYDEGS_O = POLYDEGS
d_O = 6
lmbda_O = 0

### Ridge:
POLYDEGS_R = POLYDEGS[3:]
HYPERPARAMS_R = np.logspace(-5, -2, 12)
d_R = 18
lmbda_R = 1.23e-4


### Lasso:
POLYDEGS_L = POLYDEGS[3:15]
HYPERPARAMS_L = np.logspace(-5, -2, 10)
d_L = 11
lmbda_L = 1e-5


# show plots or only save
SHOW = True


def assessModelComplexityBS(B, method, mode, polydegs):
    Bootstrappings = []
    for d in polydegs:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], B, scheme=method, mode=mode)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    return Bootstrappings

def assessModelComplexityCV(k, method, mode, polydegs):
    Crossvalidations = []
    for d in polydegs:
        CV = Bootstrap(TRAININGS[d], PREDICTIONS[d], k, scheme=method, mode=mode)
        CV()
        Crossvalidations.append(CV)

    return Crossvalidations

def assessHyperParamBS(polydeg, B, method, mode, hyperparams):
    Bootstrappings = []
    for lmbda in hyperparams:
        BS = Bootstrap(TRAININGS[polydeg], PREDICTIONS[polydeg], B, scheme=method, mode=mode, hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    return Bootstrappings

def assessHyperParamCV(polydeg, k, method, mode, hyperparams):
    Crossvalidations = []
    for lmbda in hyperparams:
        CV = CrossValidation(TRAININGS[polydeg], PREDICTIONS[polydeg], k, scheme=method, mode=mode, hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)
    return Crossvalidations


def grid_searchCV(k, method, mode, polydegs, hyperparams):
    CVgrid = []
    CVgrid = []
    for d in polydegs:
        print('d = ', d, ' ...')
        CVs_lmbda = assessHyperParamCV(d, goto_k, method, mode, hyperparams=hyperparams)
        CVgrid.append(CVs_lmbda)
        
    return CVgrid


def noneAnalysis():
    PLOT.visualise_data(*prepper.dump(), show=SHOW, cmap='terrain')

    def simple_analysis():
        workouts = {d:deepcopy(TRAININGS[d]) for d in [d_O, d_R]}
        forecasts = {d:deepcopy(PREDICTIONS[d]) for d in [d_O, d_R]}
        
        reg = linearRegression(workouts[d_O], forecasts[d_O], mode='own', scheme='OLS')
        reg.fit()
        reg = linearRegression(workouts[d_R], forecasts[d_R], mode='own', scheme='Ridge', shrinkage_parameter=lmbda_R)
        reg.fit()
        PLOT.beta_params(workouts, grouped=True, show=SHOW, tag='_ols', mark="comparison of $??$'s between Ridge and OLS") 
    
    simple_analysis()


def finalAnalysis():

    # OLS MODEL
    d = d_O
    lmbda = lmbda_O
    scheme = 'ols'
    mode = 'own'

    # Ridge MODEL
    d = d_R
    lmbda = lmbda_R
    scheme = 'Ridge'
    mode = 'skl'

    trainer, predictor = TRAININGS[d], PREDICTIONS[d]

    reg = linearRegression(trainer, predictor, mode=mode, scheme=scheme, shrinkage_parameter=lmbda)
    reg.fit()
    predictor.predict()
    PLOT.compare_data(predictor, predictor, angles=(6, 49), cmap='terrain', show=SHOW, mark='prediction set')

    MSE_un = predictor.mean_squared_error()

    BS = Bootstrap(trainer, predictor, 600, scheme=scheme, mode=mode, hyper_param=lmbda)
    BS()
    MSE_BS = BS.resamplingError()

    CV = CrossValidation(trainer, predictor, goto_k, scheme=scheme, mode=mode, hyper_param=lmbda)
    CV()
    MSE_CV = CV.resamplingError()


    MSE_str = f'\n {scheme} scheme with d = {d} and ?? = {lmbda}\n'
    MSE_str += '-'*40
    MSE_str += f'\n        unresampled MSE = {MSE_un:.4f}'
    MSE_str += f'\n          bootstrap MSE = {MSE_BS:.4f}'
    MSE_str += f'\n   cross-validation MSE = {MSE_CV:.4f}\n'
    MSE_str += '-'*40
    print(MSE_str)


def olsAnalysis():

    print('\nPerforming analysis using Ordinary least squares\n')
    print('-'*40)
    print('\n')


    def simple_analysis():
        polydegs = range(1, 8)
        workouts = {d:deepcopy(TRAININGS[d]) for d in polydegs}
        forecasts = {d:deepcopy(PREDICTIONS[d]) for d in polydegs}

        for d in polydegs:
            T, P = workouts[d], forecasts[d]

            reg = linearRegression(T, P, mode='own', scheme='OLS')
            reg.fit()
            T.computeModel()
            P.predict()
            T.computeExpectationValues()
            P.computeExpectationValues()
        
        PLOT.train_test_MSE_R2(workouts, forecasts, show=SHOW)
        PLOT.beta_params(workouts, grouped=True, show=SHOW, mark="$??$'s grouped by order $d$") 

    
    simple_analysis()
    
    # Bootstrap 
    print('   > Bootstrap ...\n')
    Bootstrappings = assessModelComplexityBS(goto_B, 'OLS', 'own', POLYDEGS_O)
    PLOT.train_test_MSE(Bootstrappings, show=SHOW)
    PLOT.BV_Tradeoff(Bootstrappings, show=SHOW)
    idx = d_O-1
    PLOT.beta_hist_resampling(Bootstrappings[idx], grouped=False, show=SHOW)
    PLOT.mse_hist_resampling(Bootstrappings[idx], show=SHOW)

    
    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    CVgrid = []
    for k in kLIST:
        Crossvalidations = assessModelComplexityCV(k, 'OLS', 'own', POLYDEGS_O)
        CVgrid.append(Crossvalidations)
    PLOT.CV_errors(CVgrid, show=SHOW)


def ridgeAnalysis():

    print('\nPerforming analysis using Ridge regression\n')
    print('-'*40)
    print('\n')

    mode = 'skl' # 'skl' is a bit faster ...

    def simple_analysis():
        workout = {d:deepcopy(TRAININGS[d]) for d in [d_R]}
        forecast = {d:deepcopy(PREDICTIONS[d]) for d in [d_R]}

        reg = linearRegression(workout[d_R], forecast[d_R], mode=mode, scheme='Ridge', shrinkage_parameter=lmbda_R)
        reg.fit()
        PLOT.beta_params(workout, grouped=True, show=SHOW, mark="$??$'s grouped by order $d$") 

    simple_analysis()
    
    # Bootstrap 
    print('   > Bootstrap ...\n')
    Bootstrappings = assessHyperParamBS(d_R, goto_B, 'Ridge', mode, hyperparams=HYPERPARAMS_R)
    PLOT.train_test_MSE(Bootstrappings, show=SHOW)
    PLOT.BV_Tradeoff(Bootstrappings, show=SHOW)
    idx = np.argmin(np.abs(HYPERPARAMS_R-lmbda_R))
    PLOT.beta_hist_resampling(Bootstrappings[idx], grouped=False, show=SHOW)
    PLOT.mse_hist_resampling(Bootstrappings[idx], show=SHOW)
   
    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    CVgrid = []
    for k in kLIST[1:3]:
        Crossvalidations = assessHyperParamCV(d_R, k, 'Ridge', mode, hyperparams=HYPERPARAMS_R)
        CVgrid.append(Crossvalidations)
    PLOT.CV_errors(CVgrid, show=SHOW)

    # Cross validation (grid search)
    print('   > k-fold cross-validation with grid search ...\n')
    CVgrid = grid_searchCV(goto_k, 'Ridge', mode, POLYDEGS_R, HYPERPARAMS_R)
    PLOT.heatmap(CVgrid, show=SHOW)

   
def lassoAnalysis():

    print('\nPerforming analysis using Lasso regression\n')
    print('-'*40)
    print('\n')

    # Bootstrap
    print('   > Bootstrap ...\n')
    Bootstrappings = assessHyperParamBS(goto_polydeg, 50, 'Lasso', 'skl', hyperparams=HYPERPARAMS_L)
    PLOT.train_test_MSE(Bootstrappings, show=SHOW)
    
    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessHyperParamCV(goto_polydeg, goto_k, 'Lasso', 'skl', hyperparams=HYPERPARAMS_L)
    PLOT.train_test_MSE(Crossvalidations, show=SHOW)
    PLOT.BV_Tradeoff(Bootstrappings, show=SHOW)

    # Cross validation (grid search)
    print('   > k-fold cross-validation with grid search ...\n')
    CVgrid = grid_searchCV(goto_k, 'Lasso', 'skl', POLYDEGS_L, HYPERPARAMS_L)
    PLOT.heatmap(CVgrid,show=SHOW)






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


    additionalInfo = []
    additionalInfo.append(f'xy-grid: N x N = {Nx} x {Ny}')
    additionalInfo.append(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).')
    additionalInfo.append(f'Ridge: Considered {len(HYPERPARAMS_R)} ??-values between ?? = {HYPERPARAMS_R[0]:.1e} and ?? = {HYPERPARAMS_R[-1]:.1e} (logarithmically spaced).')
    additionalInfo.append(f'Lasso: Considered {len(HYPERPARAMS_L)} ??-values between ?? = {HYPERPARAMS_L[0]:.1e} and ?? = {HYPERPARAMS_L[-1]:.1e} (logarithmically spaced).')

    PLOT.update_info(additionalInfo)