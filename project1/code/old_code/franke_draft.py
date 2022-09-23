
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import seaborn as sns
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['figure.figsize'] = [12, 7]


scale_mode = 'on'

# Make data.
Npoints = 20
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x = np.linspace(0, 1, Npoints)
y = np.linspace(0, 1, Npoints)
print(np.shape(x))
x, y = np.meshgrid(x,y)
print(np.shape(x))


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)


# Subtracting mean value
if scale_mode.lower().strip() == 'on':
    #z -= np.mean(z)
    z /= np.max(z)




def plot_Frankefunction():
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.05, 1.06)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show()



#   Own code to fit the Franke-function using OLS and polynomial in x and y up to fifth order.
#   Set up design matrix X

polydeg_max = 20
def fit_params(nn=5):
    xx = np.ravel(x)
    yy = np.ravel(y)

    N = len(xx)
    n = polydeg_max
    l = int((n+1)*(n+2)/2)
    X = np.ones((N,l))

    j = 1
    cols = [r'$x^0 y^0$']
    for i in range(1, n+1):
        #print('i', i)
        q = int((i)+(i+1)/2)
        #print('q', q)
        for k in range(i+1):
            #print('q+k', q+k)
            X[:,j] = (xx**(i-k))*(yy**k)
            #print(f'j = {j}: x^{i-k} y^{k}')
            cols.append(r'$x^{%i} y^{%i}$'%((i-k), k))
            j+=1

    ll = int((nn+1)*(nn+2)/2)
    X = X[:, :ll]
    cols = cols[:ll]

    Xpd = pd.DataFrame(X, columns=cols)
    #display(Xpd)

    # scikit:
    poly = PolynomialFeatures(nn)
    X_sci = poly.fit_transform(np.c_[xx, yy])
    # what is np.c_[xx, yy] ????
    
    assert np.all(np.abs(X-X_sci))<10e-12

    H = X.T @ X
    beta = np.linalg.pinv(H) @ X.T @ z.ravel()

    # Split into train and test
    X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2)
    #print(X_train)

    # NEW beta
    H = X_train.T @ X_train
    beta = np.linalg.pinv(H) @ X_train.T @ z_train.ravel()
    ztilde = X_train @ beta
    #ztilde = np.reshape(ztilde, (20,20))

    zpredict = X_test @ beta
    #zpredict = np.reshape(zpredict, (20,20))
    print(f"Number of elements in beta: {ll}")
    return beta, X, ztilde, zpredict, z_train, z_test, X_train


#   useful functions:
#   mean squared error
def MSE(data, model):
    n = np.size(data)
    return np.sum((data-model)**2)/n

#   r2 score
def R2(data, model):
    return 1 - np.sum((data - model) ** 2) / np.sum((data - np.mean(data)) ** 2)


N = polydeg_max
train_mses = np.zeros(N)
train_r2s = np.zeros(N)
test_mses = np.zeros(N)
test_r2s = np.zeros(N)
ns = np.zeros(N)
#biass = np.zeros(N)
beta_list = []
i = 0

for n in range(1,N+1):
    print(f"n={n}")
    beta, X, ztilde, zpredict, z_train, z_test, X_train = fit_params(n)
    beta_list.append(beta)
    z_all = X @ beta
    z_all = np.reshape(z_all, (Npoints,Npoints))

    ##bias = z -np.mean(z_all)

    if n in [4, 5, N]:
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        #ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(x, y, z_all, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Plot points
        #ax.scatter(X_train[:,0], X_train[:,1], z_train)

        # Customize the z axis.
        ax.set_zlim(-0.05, 1.05)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.suptitle(f"n={n}", fontsize=20)

        # Add a color bar which maps values to colors.
        #ztilde = np.reshape(ztilde, (20,20))
        fig.colorbar(surf, shrink=0.5, aspect=5)
    #mse = MSE(z.ravel(), ztilde.ravel())
    #r2 = R2(z.ravel(), ztilde.ravel())
    train_mses[i] = MSE(z_train.ravel(), ztilde.ravel())
    train_r2s[i] = R2(z_train.ravel(), ztilde.ravel())
    test_mses[i] =  MSE(z_test.ravel(), zpredict.ravel())
    test_r2s[i] = R2(z_test.ravel(), zpredict.ravel())
    ns[i] = n
    i+=1

plot_Frankefunction()

plt.show()

max_dim = len(max(beta_list, key=len))

beta_n = np.zeros((len(beta_list), max_dim))
for i, beta in enumerate(beta_list):
    beta_n[i, :len(beta)] = beta
    beta_n[i, len(beta):] = np.nan


df = pd.DataFrame(data=beta_n, index=[i for i in range(1, N+1)], columns=[r'$\beta_{%i}$'%j for j in range(max_dim)])


'''fig, ax = plt.subplots(figsize=(14,7))

sns.heatmap(df, annot=True, ax=ax, cmap='viridis', square=True)
ax.set_ylabel(r'$n$')
ax.set_title(r'$\beta$ for different polynomial degrees $n$')
plt.show()'''
#print(df)


fig, ax = plt.subplots()
ax.plot(ns, train_mses, label="Train MSE")
ax.plot(ns, test_mses, label="Test MSE")
ax.plot(ns, train_r2s,ls='--', label="Train R2")
ax.plot(ns, test_r2s, ls='--', label="Test R2")
ax.set_yscale('log')
ax.legend()
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(ns, train_mses, label="Train MSE")
ax1.plot(ns, test_mses, label="Test MSE")
ax2.plot(ns, train_r2s, label="Train R2")
ax2.plot(ns, test_r2s, label="Test R2")
ax1.set_yscale('log')
ax1.legend()
ax2.legend()
fig.tight_layout()
plt.show()





""" WITH SKLEARN"""


