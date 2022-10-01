
from src.utils import *
from src.objects import dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('on')

def datapointsFranke(eta=.01, N=20):
    x = np.sort( np.random.rand(N))
    y = np.sort( np.random.rand(N))
    x, y = np.meshgrid(x, y)
    noise = eta*np.random.randn(N, N)
    z = FrankeFunction(x, y) + noise
    return x, y, z


x, y, z = datapointsFranke(0.1)
prepper = dataPrepper(x, y, z)
prepper.split()
prepper.scale()
prepper.genereteSeveralOrders()



def ptB():
    Tr = [] # trainings
    Pr = [] # predictions

    for d in range(1, 6):
        trainer = Training(*prepper.getTrain(d))
        predictor = Prediction(*prepper.getTest(d))


        reg = linearRegression(trainer, predictor, mode='own', method='ols')
        reg.fit()
        trainer.computeModel()
        predictor.predict()
        trainer.computeExpectationValues()
        predictor.computeExpectationValues()

        Tr.append(trainer)
        Pr.append(predictor)

        
    PLOT.ptB_scores(Tr, Pr, pdf_name='scores',show=True)
    PLOT.ptB_beta_params(Tr, pdf_name='betas',show=True) #change names?


TRAININGS = [] # DICT?
PREDICTIONS = []

POLYDEGS = range(1, maxPolydeg+1)


for d in POLYDEGS:
    trainer = Training(*prepper.getTrain(d))
    predictor = Prediction(*prepper.getTest(d))

    TRAININGS.append(trainer)
    PREDICTIONS.append(predictor)


def ptC():
    Bootstrappings = []
    for train, test in zip(TRAININGS, PREDICTIONS):
        BS = Bootstrap(train, test, 200)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    PLOT.ptC_Hastie(Bootstrappings, pdf_name='Hastie_OLS',  show=True)
    PLOT.ptC_tradeoff(Bootstrappings, pdf_name='tradeoff_OLS', show=True)

    d = 5
    #PLOT.ptC_bootstrap_hist_beta(Bootstrappings[d-1], show=True)
    #PLOT.ptC_bootstrap_hist_mse(Bootstrappings[d-1], show=True)

def ptD():
    Crossvalidations = []
    for train, test in zip(TRAININGS, PREDICTIONS):
        CV = CrossValidation(train, test, 8)
        CV()
        Crossvalidations.append(CV)

    PLOT.ptD_cross_validation(Crossvalidations, pdf_name='CV_OLS', show=True)

def ptE():
    d = 8 # or 4, 5
    
    lmbdas = np.logspace(-4,-1,20)

    trainer = Training(*prepper.getTrain(d))
    predictor = Prediction(*prepper.getTest(d))

    Bootstrappings = []

    for lmbda in lmbdas:
        BS = Bootstrap(trainer, predictor, 200, method='Ridge', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.ptE_Hastie_Ridge(Bootstrappings, pdf_name='Hastie_Ridge', show=True)


    Crossvalidations = []

    for lmbda in lmbdas:
        CV = CrossValidation(trainer, predictor, 10, method='Ridge', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.ptE_CV_Ridge(Crossvalidations,  pdf_name='CV_Ridge', show=True)
    PLOT.ptE_tradeoff_Ridge(Bootstrappings,  pdf_name='tradeoff_Ridge', show=True)

def ptF():
    d = 8 # or 4, 5
    lmbdas = np.logspace(-4,-1,10)

    trainer = Training(*prepper.getTrain(d))
    predictor = Prediction(*prepper.getTest(d))

    Bootstrappings = []

    for lmbda in lmbdas:
        BS = Bootstrap(trainer, predictor, 2, method='Lasso', mode='skl', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.ptE_Hastie_Lasso(Bootstrappings,  pdf_name='Hastie_Lasso',  show=True)

    Crossvalidations = []

    for lmbda in lmbdas:
        CV = CrossValidation(trainer, predictor, 5, method='Lasso', mode='skl', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.ptE_CV_Lasso(Crossvalidations, pdf_name='CV_Lasso', show=True)
    PLOT.ptE_tradeoff_Lasso(Bootstrappings, pdf_name='tradeoff_Lasso', show=True)





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