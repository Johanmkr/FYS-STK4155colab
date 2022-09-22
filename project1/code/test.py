from tarfile import TarError
from src.utils import *

from src.designMatrix import DesignMatrix
from src.Regression import LinearRegression
from src.Resampling import Bootstrap
from src.parameterVector import ParameterVector
from src.targetVector import TargetVector

from sklearn.preprocessing import StandardScaler



Nx, Ny = 20, 20
x = np.random.rand(Nx)
y = np.random.rand(Ny)
x, y = np.meshgrid(x, y)


noise = lambda eta: eta*np.random.randn(Ny, Nx)
eta = 0.01
z = FrankeFunction(x, y) + noise(eta)


dM = DesignMatrix(5)

dM.createX(x, y)

tV = TargetVector(z)

reg = LinearRegression(tV, dM, mode='own')
trainer, predictor = reg.split(scale=True)


beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()
print(trainer.MSE)
print(np.mean(trainer.data))
print(trainer.model- trainer.data)


predictor.setOptimalbeta(beta)
ztilde = predictor.predict()
predictor.computeExpectationValues()
print(beta)
print(predictor.MSE)
print(np.mean(predictor.data))
print(predictor.model-predictor.data)
print(np.mean(predictor.model))



sys.exit()
reg = LinearRegression(z, dM, mode='skl')
trainer, predictor = reg.split()

# train:
beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()

# predict:ko
predictor.setOptimalbeta(beta)
ztildeown = predictor.computeModel()
predictor.computeExpectationValues()
print(beta)
print(predictor.MSE)

### with scaling
reg.scale()

# train:
beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()

# # predict:
predictor.setOptimalbeta(beta)
ztildeskl = predictor.computeModel()
predictor.computeExpectationValues()
print(beta)
print(predictor.MSE)


for zs, zo in zip(ztildeown, ztildeskl):
    print(zs, zo)









    


