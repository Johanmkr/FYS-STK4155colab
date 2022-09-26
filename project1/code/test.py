from tarfile import TarError
from src.utils import *

from src.designMatrix import DesignMatrix
from src.Regression import LinearRegression
from src.Resampling import Bootstrap
from src.parameterVector import ParameterVector
from src.targetVector import TargetVector

from sklearn.preprocessing import StandardScaler




def datapoints(eta=.01, N=20):
    x = np.sort( np.random.rand(N))
    y = np.sort( np.random.rand(N))
    x, y = np.meshgrid(x, y)
    noise = lambda eta: eta*np.random.randn(N, N)
    z = FrankeFunction(x, y) + noise(eta)
    return x, y, z


x, y, z = datapoints()


def datapoints(eta=.01, N=20, data='franke'):
    x = np.sort( np.random.rand(N))
    y = np.sort( np.random.rand(N))
    x, y = np.meshgrid(x, y)
    if data == 'franke':
        noise = lambda eta: eta*np.random.randn(N, N)
        z = FrankeFunction(x, y) + noise(eta)
    elif data == 'terrain' or data == 'terrain1':
        from imageio.v2 import imread
        terrain1 = imread("../data/SRTM_data_Norway_1.tif")
        z = terrain1[:N, :N].astype('float64')
    elif data == 'terrain2':
        from imageio.v2 import imread
        terrain2 = imread("../data/SRTM_data_Norway_1.tif")
        z = terrain2[:N, :N].astype('float64')
    return x, y, z



x, y, z = datapoints(N=1000, data='terrain')
'''x = np.linspace(0,1, np.shape(z)[0])
y = np.linspace(0,1, np.shape(z)[1])
x, y = np.meshgrid(x, y)
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def surface_plot(ax, x, y, z):
    surf =  ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.05, 1.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return ax, surf
fig, ax = plt.subplots(ncols=1, subplot_kw={'projection':'3d'})
ax, surf = surface_plot(ax, x, y, z)
# ax.set_title(r"Franke function for $x,y\in[0,1]$, $N=40$, $\eta=%.1f$"%eta)
# Customize the z axis.
ax.set_zlim(np.min(z), np.max(z))
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()

dM = DesignMatrix(10)

dM.createX(x, y)

tV = TargetVector(z)
reg = LinearRegression(tV, dM, mode='own')


trainer, predictor = reg.split(scale=False)
beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()
print(trainer.MSE)
print(beta)
'''print(np.mean(trainer.data))
print(trainer.model- trainer.data)'''
print('\n\n\n')

predictor.setOptimalbeta(beta)
ztilde = predictor.predict()
predictor.computeExpectationValues()
print(predictor.MSE)

'''print(beta)
print(predictor.MSE)
print(np.mean(predictor.data))
print(predictor.model-predictor.data)
print(np.mean(predictor.model))'''
print('\n\n\n')
print('\n\n\n')
trainer, predictor = reg.split(scale=True)

beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()
print(trainer.MSE)
print(beta)

print('\n\n\n')

predictor.setOptimalbeta(beta)
ztilde = predictor.predict()
predictor.computeExpectationValues()
print(predictor.MSE)





