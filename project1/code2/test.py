
from src.utils import *
from src.objects import designMatrix, targetVector, parameterVector, dataPrepper
from src.Regression import Training, Prediction, linearRegression
from src.Resampling import Bootstrap, CrossValidation


import plot as PLOT
PLOT.init('on')

def datapoints(eta=.01, N=20):
    x = np.sort( np.random.rand(N))
    y = np.sort( np.random.rand(N))
    x, y = np.meshgrid(x, y)
    noise = eta*np.random.randn(N, N)
    z = FrankeFunction(x, y) + noise
    return x, y, z


x, y, z = datapoints(0.1)

prepper = dataPrepper(x, y, z)
prepper.split()
prepper.scale()
prepper.genereteSeveralOrders()
ztest = prepper.getTest()[0]
ztrain = prepper.getTrain()[0]












def OLS_stuff():

    Tr = []# trainings
    Pr = []# predictions

    for d in range(1, 6):
        trainer = Training(*prepper.getTrain(d))
        predictor = Prediction(*prepper.getTest(d))


        reg = linearRegression(trainer, predictor, mode='own', method='ols')
        reg.fit()
        ztilde = trainer.computeModel()
        zpred = predictor.predict()
        trainer.computeExpectationValues()
        predictor.computeExpectationValues()

        Tr.append(trainer)
        Pr.append(predictor)

        
    PLOT.ptB_scores(Tr, Pr, show=True)
    PLOT.ptB_beta_params(Tr, show=True)



    Tr = []# trainings
    Pr = []# predictions

    polydegs = range(1, maxPolydeg+1)


    for d in polydegs:
        trainer = Training(*prepper.getTrain(d))
        predictor = Prediction(*prepper.getTest(d))

        Tr.append(trainer)
        Pr.append(predictor)


    Bootstrappings = []
    for train, test in zip(Tr, Pr):
        BS = Bootstrap(train, test, 200, method='OLS')
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)

    PLOT.ptC_Hastie(Bootstrappings,  show=True)
    PLOT.ptC_tradeoff(Bootstrappings, show=True)

    PLOT.ptC_bootstrap_hist_beta(Bootstrappings[5], show=True)
    PLOT.ptC_bootstrap_hist_mse(Bootstrappings[5], show=True)

    Crossvalidations = []

    for train, test in zip(Tr, Pr):
        CV = CrossValidation(train, test, 8)
        CV()
        Crossvalidations.append(CV)

    PLOT.ptD_cross_validation(Crossvalidations, show=True)

    

def Ridge_stuff():

    current_polydeg = 8 # or 4, 5
    lmbdas = np.logspace(-4,-1,20)

    trainer = Training(*prepper.getTrain(current_polydeg))
    predictor = Prediction(*prepper.getTest(current_polydeg))

    Bootstrappings = []

    for lmbda in lmbdas:
        BS = Bootstrap(trainer, predictor, 200, method='Ridge', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.ptE_Hastie_Ridge(Bootstrappings, show=True)


    Crossvalidations = []

    for lmbda in lmbdas:
        CV = CrossValidation(trainer, predictor, 10, method='Ridge', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.ptE_CV_Ridge(Crossvalidations, show=True)

    PLOT.ptE_tradeoff_Ridge(Bootstrappings, show=True)



def Lasso_stuff():

    current_polydeg = 8 # or 4, 5
    lmbdas = np.logspace(-4,-1,10)

    trainer = Training(*prepper.getTrain(current_polydeg))
    predictor = Prediction(*prepper.getTest(current_polydeg))

    Bootstrappings = []

    for lmbda in lmbdas:
        BS = Bootstrap(trainer, predictor, 2, method='Lasso', mode='skl', hyper_param=lmbda)
        BS()
        BS.bias_varianceDecomposition()
        Bootstrappings.append(BS)
    
    PLOT.ptE_Hastie_Ridge(Bootstrappings, show=True)


    Crossvalidations = []

    for lmbda in lmbdas:
        CV = CrossValidation(trainer, predictor, 5, method='Lasso', mode='skl', hyper_param=lmbda)
        CV()
        Crossvalidations.append(CV)

    PLOT.ptE_CV_Ridge(Crossvalidations, show=True)

    PLOT.ptE_tradeoff_Lasso(Bootstrappings, show=True)



#Lasso_stuff()
Ridge_stuff()


#OLS_stuff()
