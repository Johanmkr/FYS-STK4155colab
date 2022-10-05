
from src.utils import *
from src.objects import dataPrepper, groupedVector
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation

from imageio.v2 import imread

import plot as PLOT
PLOT.init('off')

PLOT.add_path('terrain')



def datapointsTerrain(location='GrandCanyon', sx=slice(2500, 3400, 30), sy=slice(2500, 3400, 30)):
    
    terrain1 = imread(f"../data/SRTM_data_{location}.tif")
    z = terrain1[sx, sy].astype('float64')
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


for d in POLYDEGS:
    TRAININGS[d] = Training(*prepper.getTrain(d))
    PREDICTIONS[d] = Prediction(*prepper.getTest(d))


HYPERPARAMS = np.logspace(-5, -2, 12)

goto_polydeg = 8
goto_B = 200
goto_k = 10


show = True


def assessModelComplexityBS(B, method, mode, polydegs=POLYDEGS):
    Bootstrappings = []
    for d in polydegs:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], B, scheme=method, mode=mode)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    return Bootstrappings

def assessModelComplexityCV(k, method, mode, polydegs=POLYDEGS):
    Crossvalidations = []
    for d in polydegs:
        CV = Bootstrap(TRAININGS[d], PREDICTIONS[d], k, scheme=method, mode=mode)
        CV()
        Crossvalidations.append(CV)

    return Crossvalidations

def assessHyperParamBS(polydeg, B, method, mode, hyperparams=HYPERPARAMS):
    Bootstrappings = []
    for lmbda in hyperparams:
        BS = Bootstrap(TRAININGS[polydeg], PREDICTIONS[polydeg], B, scheme=method, mode=mode, hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    return Bootstrappings

def assessHyperParamCV(polydeg, k, method, mode, hyperparams=HYPERPARAMS):
    Crossvalidations = []
    for lmbda in hyperparams:
        CV = CrossValidation(TRAININGS[polydeg], PREDICTIONS[polydeg], k, scheme=method, mode=mode, hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)
    return Crossvalidations


def grid_searchCV(k, method, mode, polydegs=POLYDEGS, hyperparams=HYPERPARAMS):
    CVgrid = []
    CVgrid = []
    for d in polydegs:
        print('d = ', d, ' ...')
        CVs_lmbda = assessHyperParamCV(d, goto_k, method, mode, hyperparams=hyperparams)
        CVgrid.append(CVs_lmbda)
        
    return CVgrid




def noneAnalysis():
    PLOT.visualise_data(*prepper.dump(), show=show, cmap='terrain')
    #PLOT.visualise_data(*prepper.getTrain(goto_polydeg), show=show)


def finalAnalysis():

    d = 18
    lmbda = 1.23e-4

    #d = 15
    scheme = 'Ridge'
    #lmbda = 1.08e-8

    trainer, predictor = TRAININGS[d], PREDICTIONS[d]

    reg = linearRegression(trainer, predictor, mode='own', scheme=scheme, shrinkage_parameter=lmbda)
    reg.fit()
    predictor.predict()
    PLOT.compare_data(predictor, predictor, angles=(6, 49), cmap='terrain', show=show, mark='prediction set')
    #trainer.computeModel()
    #PLOT.compare_data(trainer, trainer, angles=(6, 49), cmap='terrain', tag='train', show=show, mark='training set')




def olsAnalysis():

    print('\nPerforming analysis using Ordinary least squares\n')
    print('-'*40)
    print('\n')


    def simple_analysis():
        polydegs = range(1, 9)
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
        
        PLOT.train_test_MSE_R2(workouts, forecasts, show=show)
        PLOT.beta_params(workouts, grouped=True, show=show, mark="$β$'s grouped by order $d$") 
        
        # Select one polydeg to visualise for
        T, P = workouts[goto_polydeg], forecasts[goto_polydeg]
        PLOT.compare_data(P, P, cmap='terrain', show=show, mark="prediction set")

    #
    # simple_analysis()

    # Bootstrap 
    print('   > Bootstrap ...\n')
    Bootstrappings = assessModelComplexityBS(goto_B, 'OLS', 'own')
    PLOT.train_test_MSE(Bootstrappings, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)
    idx = goto_polydeg-1
    PLOT.beta_hist_resampling(Bootstrappings[idx], grouped=False, show=show)

    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessModelComplexityCV(goto_k, 'OLS', 'own')
    PLOT.train_test_MSE(Crossvalidations, show=show)



def ridgeAnalysis():

    print('\nPerforming analysis using Ridge regression\n')
    print('-'*40)
    print('\n')
    d_R = 18

    # Bootstrap 
    print('   > Bootstrap ...\n')
    
    Bootstrappings = assessHyperParamBS(d_R, goto_B, 'Ridge', 'skl', hyperparams=np.logspace(-5, -2, 20))
    PLOT.train_test_MSE(Bootstrappings, show=show)
    # Trade-off : d = 13, 14, 15, 16
    PLOT.BV_Tradeoff(Bootstrappings, show=show)

    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessHyperParamCV(d_R, goto_k, 'Ridge', 'skl', hyperparams=np.logspace(-5, -2, 20))
    PLOT.train_test_MSE(Crossvalidations, show=show)

    # Cross validation (grid search)
    print('   > k-fold cross-validation with grid search ...\n')
    t0 = time()
    CVgrid = grid_searchCV(goto_k, 'Ridge', 'skl', POLYDEGS[3:], HYPERPARAMS)
    t1 = time()
    print(t1-t0)
    print('plotting')
    levels = [0, 0.18] #chnage
    def fmt(x):
        if np.abs(x) < 1e-12:
            s = ''
        else:
            s = r'$\text{MSE}^\text{OLS}$'
        return s
    PLOT.heatmap(CVgrid,  show=show, mark=f"{len(HYPERPARAMS)} $λ$'s from {HYPERPARAMS[0]:.2e} to {HYPERPARAMS[0]:.2e}")
    '''# plot for polydegs 1-5
    hyperparams = np.logspace(-6, 0, 60)
    t0 = time()
    CVgrid = grid_searchCV(goto_k, 'Ridge', 'skl', range(1,6), hyperparams)
    t1 = time()
    print(t1-t0)
    print('plotting')
    PLOT.heatmap(CVgrid, tag='_loworder', show=show, mark=f"{len(hyperparams)} $λ$'s from {hyperparams[0]:.2e} to {hyperparams[0]:.2e}")'''
    


   
def lassoAnalysis():

    print('\nPerforming analysis using Lasso regression\n')
    print('-'*40)
    print('\n')

    # Bootstrap
    '''print('   > Bootstrap ...\n')
    Bootstrappings = assessHyperParamBS(goto_polydeg, 50, 'Lasso', 'skl')
    PLOT.train_test_MSE(Bootstrappings, show=show)
    
    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessHyperParamCV(goto_polydeg, 8, 'Lasso', 'skl')
    PLOT.train_test_MSE(Crossvalidations, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)'''
    HYPERPARAMS_L = np.logspace(-5, -2, 10)
    POLYDEGS_L = POLYDEGS[3:-2]
    k_L = 5
    # Cross validation (grid search)
    print('   > k-fold cross-validation with grid search ...\n')
    t0 = time()
    CVgrid = grid_searchCV(k_L, 'Lasso', 'skl', POLYDEGS_L, HYPERPARAMS_L)
    t1 = time()
    print(t1-t0)
    print('plotting')
    PLOT.heatmap(CVgrid, show=show, mark=f"{len(HYPERPARAMS_L)} $λ$'s from {HYPERPARAMS_L[0]:.2e} to {HYPERPARAMS_L[0]:.2e}")












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
    additionalInfo.append(f'xy-grid: (Nx) x (Ny) = {Nx} x {Ny}')
    additionalInfo.append(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).')
    additionalInfo.append(f'Considered {len(HYPERPARAMS)} λ-values between λ = {HYPERPARAMS[0]:.1e} and λ = {HYPERPARAMS[-1]:.1e} (logarithmically spaced).')

    PLOT.update_info(additionalInfo)