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


dM = DesignMatrix(5)

dM.createX(x, y)
print(dM)
tV = TargetVector(z)
lmbda = 1
reg = LinearRegression(tV, dM, mode='own', method='OLS')
trainer, predictor = reg.split(scale=True)
beta = trainer.train(lmbda)
amp_beta = beta.group()

print(amp_beta)
reg.changeMode()
beta = trainer.train(lmbda)
print(beta)
predictor.setOptimalbeta(beta)
zpred = predictor.predict()
predictor.computeExpectationValues()
print(predictor.MSE)




#z = np.reshape(z, np.shape(x))
#ztilde = np.reshape(z, np.shape(x))

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def surface_plot(ax, x, y, z, tri=True):
    if tri:
        surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    else:
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.05, 1.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return ax, surf

fig, axes = plt.subplots(ncols=2, subplot_kw={'projection':'3d'})
ax1, ax2 = axes

surface_plot(ax1, x, y, z, False)
ax1.scatter(x, y, z, color='r')




x, y, z = predictor.griddata()
x, y, ztilde = predictor.gridmodel()

#surface_plot(ax2, x, y, ztilde)
ax2.scatter(x, y, ztilde, color='r')


for ax in axes:
    #ax.set_zlim(np.min(z), np.max(z))
    ax.set_zlim(-0.05, 1.05)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
plt.tight_layout()
plt.show()


sys.exit()
trainer.computeModel()
trainer.computeExpectationValues()
print(trainer.MSE)
print(beta)
predictor.setOptimalbeta(beta)
ztilde_pred = predictor.predict()
predictor.computeExpectationValues()
print(predictor.MSE)



sys.exit()

def datapointsTerrain(sx=10, sy=10, tag='2'):
    from imageio.v2 import imread
    terrain1 = imread(f"../data/SRTM_data_Norway_{tag}.tif")
    z = terrain1[::sy, ::sx].astype('float64')
    Ny, Nx = np.shape(z)
    x = np.sort( np.random.rand(Nx))
    y = np.sort( np.random.rand(Ny))
    x, y = np.meshgrid(x, y)
    return x, y, z



x, y, z = datapointsTerrain('2')

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def surface_plot(ax, x, y, z):
    surf =  ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.05, 1.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return ax, surf
fig, axes = plt.subplots(ncols=2, subplot_kw={'projection':'3d'})
ax1, ax2 = axes
ax1, surf1 = surface_plot(ax1, x, y, z)



dM = DesignMatrix(5)

dM.createX(x, y)

tV = TargetVector(z)
reg = LinearRegression(tV, dM, mode='own')

print('\n\n\n')
trainer, predictor = reg.split(scale=False)

beta = trainer.train()
trainer.computeModel()
trainer.computeExpectationValues()
print(trainer.MSE)
print(beta)
bbeta = np.zeros(len(beta.beta)+1)
bbeta[0] = np.mean(z)
bbeta[1:] = beta.beta
ztildee = reg.dM.X @ beta.beta
ztilde = np.reshape(ztildee, np.shape(z))
print('\n\n\n')

predictor.setOptimalbeta(beta)
ztilde_pred = predictor.predict()
predictor.computeExpectationValues()
print(predictor.MSE)


ax2, surf2 = surface_plot(ax2, x, y, ztilde)

for ax in axes:
    ax.set_zlim(np.min(z), np.max(z))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
plt.tight_layout()
plt.show()