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




figures = []
info = {'METHOD':[], 'RESAMPLING':[]}

figures.append('Fig1.pdf')
info['METHOD'] = 'OLS'
info['RESAMPLING'] = '-'


infopd = pd.DataFrame(info, figures)


from tabulate import tabulate
infopd.to_pickle('test.pkl')
###
infopd = pd.read_pickle('test.pkl')

row = pd.DataFrame({'METHOD':'Ridge', 'RESAMPLING':'Bootstrap'}, ['Fig2.pdf'])
#infopd.to_markdown('test.md')
infopd = pd.concat([infopd, row])
infodict = infopd.transpose().to_dict()
print(infodict.keys())
infopd = pd.DataFrame.from_dict(infodict, orient='index')
infopd.to_pickle('test.pkl')
infopd = pd.read_pickle('test.pkl')
print(infopd)
infopd.to_markdown('test.md')
