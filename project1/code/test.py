from src.utils import *

from src.designMatrix import DesignMatrix
from src.Regression import LinearRegression
from src.Resampling import Bootstrap
from src.parameterVector import ParameterVector




Nx, Ny = 20, 20
x = np.random.rand(Nx)
y = np.random.rand(Ny)
x, y = np.meshgrid(x, y)


noise = lambda eta: eta*np.random.randn(Ny, Nx)
eta = 0.01
z = FrankeFunction(x, y) + noise(eta)


dM = DesignMatrix(5)

dM.createX(x, y)
reg = LinearRegression(z, dM, mode='own')
trainer, predictor = reg.split()

# train:
beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()

# predict:
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

# predict:
predictor.setOptimalbeta(beta)
ztildeskl = predictor.computeModel()
predictor.computeExpectationValues()
print(beta)
print(predictor.MSE)


for zs, zo in zip(ztildeown, ztildeskl):
    print(zs, zo)








