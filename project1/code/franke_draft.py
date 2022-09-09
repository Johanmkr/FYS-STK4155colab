
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = [14, 9]


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)

def plot_Frankefunction():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


#   Own code to fit the Franke-function using OLS and polynomial in x and y up to fifth order. 
#   Set up design matrix X

def fit_params(n=5):
    xx = np.ravel(x)
    yy = np.ravel(y)

    N = len(xx)
    l = int((n+1)*(n+2)/2)
    X = np.ones((N,l))

    for i in range(1, n+1):
        q = int((i)+(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (xx**(i-k))*(yy**k)


    H = X.T @ X
    beta = np.linalg.pinv(H) @ X.T @ z.ravel()

    ztilde = X @ beta 
    ztilde = np.reshape(ztilde, (20,20))
    print(f"Number of elements in beta: {l}")
    return beta, ztilde


#   useful functions:
#   mean squared error
def MSE(data, model):
    n = np.size(data)
    return np.sum((data-model)**2)/n

#   r2 score
def R2(data, model):
    return 1 - np.sum((data - model) ** 2) / np.sum((data - np.mean(data)) ** 2)


N = 5
mses = np.zeros(N-1)
r2s = np.zeros(N-1)
ns = np.zeros(N-1)
beta_list = [] 
i = 0
for n in range(2,N+1):
    print(f"n={n}")
    beta, ztilde = fit_params(n)
    beta_list.append(beta)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, ztilde, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.suptitle(f"n={n}", fontsize=20)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    mse = MSE(z.ravel(), ztilde.ravel())
    r2 = R2(z.ravel(), ztilde.ravel())
    mses[i] = mse 
    r2s[i] = r2 
    ns[i] = n
    i+=1


plt.show()

max_dim = len(max(beta_list, key=len))

beta_n = np.zeros((len(beta_list), max_dim))
for i, beta in enumerate(beta_list):
    beta_n[i, :len(beta)] = beta
    beta_n[i, len(beta):] = np.nan


df = pd.DataFrame(data=beta_n, index=[i for i in range(2, N+1)], columns=[r'$\beta_{%i}$'%j for j in range(max_dim)])


fig, ax = plt.subplots(figsize=(14,7))


sns.heatmap(df, annot=True, ax=ax, cmap='viridis', square=True)
ax.set_ylabel(r'$n$')
ax.set_title(r'$\beta$ for different polynomial degrees $n$')
plt.show()



fig, ax = plt.subplots()
plt.plot(ns, mses/mses.max(), label="MSE")
plt.plot(ns, r2s/r2s.max(), label="R2")
plt.legend()
# plt.tight_layout()
plt.show()