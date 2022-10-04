
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


HYPERPARAMS = np.logspace(-6, -1, 10)

goto_polydeg = 8
goto_B = 200
goto_k = 5


show = False


def assessModelComplexityBS(B, method, mode):
    Bootstrappings = []
    for d in POLYDEGS:
        BS = Bootstrap(TRAININGS[d], PREDICTIONS[d], B, scheme=method, mode=mode)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    return Bootstrappings

def assessModelComplexityCV(k, method, mode):
    Crossvalidations = []
    for d in POLYDEGS:
        CV = Bootstrap(TRAININGS[d], PREDICTIONS[d], k, scheme=method, mode=mode)
        CV()
        Crossvalidations.append(CV)

    return Crossvalidations

def assessHyperParamBS(polydeg, B, method, mode):
    Bootstrappings = []
    for lmbda in HYPERPARAMS:
        BS = Bootstrap(TRAININGS[polydeg], PREDICTIONS[polydeg], B, scheme=method, mode=mode, hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    return Bootstrappings

def assessHyperParamCV(polydeg, k, method, mode):
    Crossvalidations = []
    for lmbda in HYPERPARAMS:
        CV = CrossValidation(TRAININGS[polydeg], PREDICTIONS[polydeg], k, scheme=method, mode=mode, hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)
    return Crossvalidations



def noneAnalysis():
    PLOT.visualise_data(*prepper.dump(), show=show)
    #PLOT.visualise_data(*prepper.getTrain(goto_polydeg), show=show)



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
        PLOT.compare_data(P, P, show=show, mark="prediction set")

    simple_analysis()

    # Bootstrap 
    print('   > Bootstrap ...\n')
    Bootstrappings = assessModelComplexityBS(goto_B, 'OLS', 'own')
    PLOT.train_test_MSE(Bootstrappings, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)
    idx = goto_polydeg-1
    PLOT.beta_hist_resampling(Bootstrappings[idx], grouped=True, show=show)

    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessModelComplexityCV(goto_k, 'OLS', 'own')
    PLOT.train_test_MSE(Crossvalidations, show=show)


def ridgeAnalysis():

    print('\nPerforming analysis using Ridge regression\n')
    print('-'*40)
    print('\n')


    # Bootstrap 
    print('   > Bootstrap ...\n')
    Bootstrappings = assessHyperParamBS(goto_polydeg, goto_B, 'Ridge', 'own')
    PLOT.train_test_MSE(Bootstrappings, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)

    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessHyperParamCV(goto_polydeg, goto_k, 'Ridge', 'own')
    PLOT.train_test_MSE(Crossvalidations, show=show)
    


   
def lassoAnalysis():

    print('\nPerforming analysis using Lasso regression\n')
    print('-'*40)
    print('\n')

    # Bootstrap
    print('   > Bootstrap ...\n')
    Bootstrappings = assessHyperParamBS(goto_polydeg, 50, 'Lasso', 'skl')
    PLOT.train_test_MSE(Bootstrappings, show=show)
    
    # Cross-validation
    print('   > k-fold cross-validation ...\n')
    Crossvalidations = assessHyperParamCV(goto_polydeg, 8, 'Lasso', 'skl')
    PLOT.train_test_MSE(Crossvalidations, show=show)
    PLOT.BV_Tradeoff(Bootstrappings, show=show)










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


    additionalInfo = []
    additionalInfo.append(f'xy-grid: (Nx) x (Ny) = {Nx} x {Ny}')
    additionalInfo.append(f'Considered {len(POLYDEGS)} polynomial degrees between d = {POLYDEGS[0]} and d = {POLYDEGS[-1]} (linarly spaced).')
    additionalInfo.append(f'Considered {len(HYPERPARAMS)} λ-values between λ = {HYPERPARAMS[0]:.1e} and λ = {HYPERPARAMS[-1]:.1e} (logarithmically spaced).')

    PLOT.update_info(additionalInfo)