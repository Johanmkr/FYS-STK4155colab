
from src.utils import *
from src.objects import designMatrix, targetVector, parameterVector
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


class dataPrepper:

    def __init__(self, x, y, z, test_size=0.2) -> None:
        
        X = featureMatrix_2Dpolynomial(x, y)
        self.dM_org = designMatrix(X)
        self.tV_org = targetVector(z)

        self.original = {'design matrix':self.dM_org, 'target vector':self.tV_org}
        
        self.npoints = len(self.tV_org) # number of points
        self.idx = np.random.permutation(self.npoints) # shuffle indices
        self.test_size = test_size
        self.polydeg = self.dM_org.polydeg
        
        self.several = False

     
    def split(self, test_size=None):
        test_size = test_size or self.test_size
        s =  int(self.npoints*test_size)

        z = self.tV_org[self.idx]
        X = self.dM_org[self.idx]

        self.tV_train = targetVector(z[s:])
        self.dM_train = designMatrix(X[s:])
        self.tV_test = targetVector(z[:s])
        self.dM_test = designMatrix(X[:s])

    def scale(self):
        self.mu_z, self.sigma_z = self.tV_train.getScalingParams()
        self.mu_X, self.sigma_X = self.dM_train.getScalingParams()

        self.tV_test.setScalingParams(self.mu_z, self.sigma_z)
        self.dM_test.setScalingParams(self.mu_X, self.sigma_X)

        self.tV_train.scale()
        self.dM_train.scale()
        self.tV_test.scale()
        self.dM_test.scale()

    def __call__(self):
        return self.tV_train, self.dM_train, self.tV_test, self.dM_test

    def getTrain(self, polydeg=None):
        if self.several and polydeg is not None:
            return self.tV_train, self.dM_train[polydeg]
        else:
            return self.tV_train, self.dM_train

    def getTest(self, polydeg=None):
        if self.several and polydeg is not None:
            return self.tV_test, self.dM_test[polydeg]
        else:
            return self.tV_test, self.dM_test

    def reduceOrderPolynomial(self, polydeg):
        if polydeg < self.polydeg:
            nfeatures = polydeg2features(polydeg)

            Xtrain = self.dM_train.getMatrix().copy()
            Xtrain = Xtrain[:,:nfeatures]
            dM_train = designMatrix(Xtrain)

            Xtest = self.dM_test.getMatrix().copy()
            Xtest = Xtest[:,:nfeatures]
            dM_test = designMatrix(Xtest)
        elif polydeg == self.polydeg:
            dM_train = self.dM_train
            dM_test = self.dM_test

        else:
            raise ValueError("Polynomial degree cannot be higher than the original one.")

        
        return dM_train, dM_test

    def genereteSeveralOrders(self, polydegs=None, save=True):
        polydegs = polydegs or range(1, self.polydeg+1)
        dMs_train = {}
        dMs_test = {}
        for d in polydegs:
            dM_train, dM_test = self.reduceOrderPolynomial(d)
            dMs_train[d] = dM_train
            dMs_test[d] = dM_test
        if save:
            dMs_train[self.polydeg] = self.dM_train
            dMs_test[self.polydeg] = self.dM_test
            self.dM_train = dMs_train
            self.dM_test = dMs_test
            self.several = True
        return dMs_train, dMs_test



'''
class Pipeline:
    def __init__(self, x, y, z) -> None:
        prepper = dataPrepper(x, y, z)
        prepper.split()
        prepper.scale()
        prepper.genereteSeveralOrders()
'''

prepper = dataPrepper(x, y, z)
prepper.split()
prepper.scale()
prepper.genereteSeveralOrders()
ztest = prepper.getTest()[0]
ztrain = prepper.getTrain()[0]

Tr = []# trainings
Pr = []# predictions

for d in range(1, 7):
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

    
'''PLOT.ptB_scores(Tr, Pr, show=True)
PLOT.ptB_beta_params(Tr, show=True)
'''


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
    BS = Bootstrap(train, test, 600, method='OLS')
    BS()
    BS.bias_varianceDecomposition()
    Bootstrappings.append(BS)

PLOT.ptC_Hastie(Bootstrappings, show=True)
PLOT.ptC_tradeoff(Bootstrappings, show=True)

Crossvalidations = []

for train, test in zip(Tr, Pr):
    CV = CrossValidation(train, test, 8)
    CV()
    Crossvalidations.append(CV)

PLOT.ptD_cross_validation(Crossvalidations, show=True)



    