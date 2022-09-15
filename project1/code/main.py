import numpy as np
from plot import *
from utils import *




### Franke function as test function:
def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


###  Create grid
Nx, Ny = 20, 20
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)

noise = lambda a: a*np.random.randn(Ny, Nx)
z = FrankeFunction(x, y) + noise(0.1)

data = Data2D(x, y, z)

data.create_designMatrix()

models = []
for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    model = data.prepare_model(n)
    model.optimize()
    model()
    ##plot_3d_model(data, model)
    models.append(model)

#plt.show()


'''plot_beta_params(models)
plt.show()'''

plot_bias_variance_tradeoff(models)
plt.show()
