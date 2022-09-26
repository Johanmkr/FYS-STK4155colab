import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

# other rc parameters
plt.rc('figure', figsize=(12,7))
SMALL_SIZE = 24
MEDIUM_SIZE = 24
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# folder paths
here = os.path.abspath(".")
data_path = here + "/../output/data/"
plot_path = here + "/../output/figures/"
latex_path = here + "/../latex/"


def init(test_mode='off'):
    testmode = str(test_mode).strip().lower()
    global testMode
    if testmode in ['off', 'false']:
        testMode = False
    elif testmode in ['on', 'true']:
        testMode = True



def save_push(fig, pdf_name, save=True, push=True, show=False, tight=True):
    """
    This function handles wether you want to show,
    save and/or push the file to git.
    Args:
        fig (matplotlib.figure): Figure you want to handle
        pdfname (string): Name of output pdf file
        args (argparse)
    """
    if tight:
        fig.tight_layout()
    file = plot_path + pdf_name.replace('.pdf', '').strip() + ".pdf"
    if testMode:
        if show:
            plt.show()
        else:
            plt.close()
    else:
        if save and pdf_name != 'none':
            print(f'Saving plot: {file}')
            fig.savefig(file, bbox_inches="tight")
        if push and pdf_name != 'none':
            os.system(f"git add {file}")
            os.system("git commit -m 'upload plot'")
            os.system("git push")
        if show:
            plt.show()
        else:
            plt.clf()

        plt.close()
        


def set_axes_2d(ax, xlabel='none', ylabel='none', title='none', legend=True, xlim='none', ylim='none'):
    if xlabel != 'none':
        ax.set_xlabel(xlabel)
    if ylabel != 'none':
        ax.set_ylabel(ylabel)
    if title != 'none':
        ax.set_title(title)
    if legend:
        ax.legend()

    if xlim != 'none':
        ax.set_xlim(xlim)
    if ylim != 'none':
        ax.set_ylim(ylim)
    





def surface_plot(ax, x, y, z):
    surf =  ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.05, 1.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return ax, surf

def plot_franke(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.suptitle(f"n={n}", fontsize=20)
    fig.suptitle(f"Franke", fontsize=20)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    plt.show()


def OLDplot_3d_model(x, y, z, ztilde, degree, pdf_name='none', show=False):
    fig, (ax0, ax) = plt.subplots(ncols=2, subplot_kw={'projection':'3d'})
    ax0, surf0 = surface_plot(ax0, x, y, z)
    ax, surf = surface_plot(ax, x, y, ztilde)
    ax0.set_title(r"Original data", fontsize=20)
    ax.set_title(r"Polynomial degree $n=%i$"%(degree), fontsize=20)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()


    save_push(fig, pdf_name, show=show)


def OLDplot_beta_params(models, pdf_name='none', show=False):
    fig, ax = plt.subplots()
    for model in models:
        ax.errorbar(range(model.features), model.beta, None, fmt='o', ms='10',  ls='', label=r'$n=%i$'%model.polydeg)
    B = model.features # max
    ax.set_xticks(range(B))
    ax.set_xticklabels([r'$\beta_{%i}$'%i for i in range(B)])
    ax.legend()
    ax.set_title(r'$\beta_i$ for various polynomial degrees')
    ax.set_xlim(0, B)
    fig.tight_layout()

    save_push(fig, pdf_name, show=show)
    


def OLDplot_bias_variance_tradeoff(models, pdf_name='none', show=False):
    fig, ax = plt.subplots()
    MSE = np.zeros(len(models))
    var = np.zeros(len(models))
    bias = np.zeros(len(models))
    polydegs = np.zeros(len(models))
    for i, model in enumerate(models):
        MSE[i] = model.test.MSE
        var[i] = model.test.var
        bias[i] = model.test.bias2
        polydegs[i] = model.polydeg

    P = model.polydeg 
    ax.errorbar(polydegs, MSE, None, fmt='o', ms='10', ls='--', label='test MSE')
    ax.errorbar(polydegs, var, None, fmt='o', ms='10', ls='--', label='variance')
    ax.errorbar(polydegs, bias, None, fmt='o', ms='10', ls='--', label=r'bias$^2$')
    ax.legend()
    ax.set_title(r'Bias-variance trade-off')
    ax.set_xlabel('polynomial degree')
    ax.set_ylabel('error')
    ax.set_xlim(1-0.05, P+0.05)
    fig.tight_layout()


    save_push(fig, pdf_name, show=show)

    



### define styles:
## believe I will change this...

dTRAIN = {'ls':'--', 'c':'b'}
dTEST = {'ls':'-',  'c':'r'}

dMSE = {'c':'firebrick'}
dR2 = {'c':'olive'}
dBIAS = {'c':'dodgerblue'}
dVAR = {'c':'orange'}




'''
# Pt. b)
'''


def ptB_franke_funcion(x, y, regression, pdf_name='none', show=False):
    fig, (ax0, ax1) = plt.subplots(ncols=2, subplot_kw={'projection':'3d'})
    z = np.reshape(regression.data, np.shape(x))
    ztilde = np.reshape(regression.model, np.shape(x))
    ax0, surf0 = surface_plot(ax0, x, y, z)
    ax1, surf1 = surface_plot(ax1, x, y, ztilde)
    ax0.set_title(r"Original data")
    ax1.set_title(r"Polynomial degree $n=%i$"%(regression.polydeg))
    # Customize the z axis.
    for ax in [ax0, ax1]:
        ax.set_zlim(np.min(z), np.max(z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.suptitle(f"n={n}", fontsize=20)
    fig.suptitle(f"Franke")

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)


    save_push(fig, pdf_name, show=show)
    

def ptB_franke_funcion_only(x, y, z, pdf_name='none', show=False):
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

    save_push(fig, pdf_name, show=show)

def ptB_scores(trainings, predictions, pdf_name='none', show=False):
    """
    Part b) Plot MSE and R2 as functions of polynomial degree.
    """    

    fig, axes = plt.subplots(nrows=2, sharex=True)
    ax1, ax2 = axes.flat

    N = len(trainings)
    n = np.zeros(N)
    MSE = np.zeros((2,N))
    R2 = np.zeros((2,N))

    i = 0
    for train, test in zip(trainings, predictions):
        n[i] = train.polydeg

        MSE[0,i] = train.MSE
        MSE[1,i] = test.MSE
        R2[0,i] = train.R2
        R2[1,i] = test.R2

        i += 1
    
    ax1.plot(n, MSE[0], ls=dTRAIN['ls'],c=dMSE['c'], label='train MSE')
    ax1.plot(n, MSE[1], ls=dTEST['ls'], c=dMSE['c'], label='test MSE')
    ax2.plot(n, R2[0],  ls=dTRAIN['ls'],c=dR2['c'], label='train $R^2$')
    ax2.plot(n, R2[1],  ls=dTEST['ls'], c=dR2['c'], label='test $R^2$')

    ax2.set_xticks(list(n))
    ax2.set_xticklabels([f'{ni:.0f}' for ni in n])

    set_axes_2d(ax1, ylabel='score', title='Mean squared error')
    set_axes_2d(ax2, xlabel='polynomial degree', ylabel='score', title='$R^2$-score', xlim=(n[0], n[-1]))

    pdfname = 'ptB_' + pdf_name.strip().replace('ptB_', '') 
    save_push(fig, pdfname, show=show)


def ptB_beta_params(trainings, pdf_name='none', show=False):
    """
    Part b) Plot beta params.
    """
    fig, ax = plt.subplots()


    for train in trainings:
        beta = train.pV
        # from IPython import embed; embed()
        ax.errorbar(range(train.features), beta[:], beta.stdv, fmt='o', ms='10',  ls='--', label=r'$d=%i$'%train.polydeg)

    B = trainings[0].features # max when reversed order
    # B = train.features #max when correct order
    ax.set_xticks(range(B))
    #ax.set_xticklabels([r'$\beta_{%i}$'%i for i in range(B)])
    # ax.set_xticklabels(train.pV.idx_tex)
    ax.set_xticklabels(trainings[0].pV.idx_tex)


    set_axes_2d(ax, title=r'$\beta$ for various polynomial degrees $d$', xlim=(0,B-1))

    pdfname = 'ptB_' + pdf_name.strip().replace('ptB_', '') 
    save_push(fig, pdfname, show=show)




"""  
Pt. c)
"""


def ptC_Hastie(bootstrappings, pdf_name='none', show=False):
    fig, ax = plt.subplots()

    N = len(bootstrappings)
    n = np.zeros(N)
    B = bootstrappings[0].B
    trainMSE = np.zeros((N,B))
    testMSE = np.zeros((N,B))

    i = 0
    for bs in bootstrappings:
        n[i] = bs.polydeg
        trainMSE[i], testMSE[i] = bs.mean_squared_error()
        i+=1
    
    ax.plot(n, trainMSE, lw=0.3, c=dTRAIN['c'], alpha=0.3)
    ax.plot(n, testMSE,  lw=0.3, c=dTEST['c'],  alpha=0.3)

    ax.errorbar(n, np.mean(trainMSE, axis=1), np.std(trainMSE, axis=1), c=dTRAIN['c'], capsize=2, fmt='.', ls='-', lw=1.1, alpha=0.9, label='train MSE')
    ax.errorbar(n, np.mean(testMSE, axis=1),  np.std(testMSE, axis=1),  c=dTEST['c'],  capsize=2, fmt='.', ls='-', lw=1.1, alpha=0.9, label='test MSE')

    ax.set_xticks(list(n))
    ax.set_xticklabels([f'{ni:.0f}' for ni in n])

    ymax = np.min([0.5, np.max(testMSE), np.max(trainMSE)])

    set_axes_2d(ax, xlabel='polynomial degree', ylabel='score', title='Mean squared error', xlim=(n[0], n[-1]), ylim=(0,ymax))
    pdfname = 'ptC_' + pdf_name.strip().replace('ptC_', '') 
    save_push(fig, pdfname, show=show)


def ptC_tradeoff(bootstrappings, pdf_name='none', show=False):
    fig, ax = plt.subplots()

    N = len(bootstrappings)
    n = np.zeros(N)
    error = np.zeros(N)
    bias2 = np.zeros(N)
    var = np.zeros(N)

    i = 0
    for bs in bootstrappings:
        n[i] = bs.polydeg
        error[i] = bs.error
        bias2[i] = bs.bias2
        var[i] = bs.var

        i += 1
    
    ax.plot(n, error, ls=dTEST['ls'], c=dMSE['c'],  label='error')
    ax.plot(n, bias2, ls=dTEST['ls'], c=dBIAS['c'], label='bias$^2$')
    ax.plot(n, var,   ls=dTEST['ls'], c=dVAR['c'],  label='variance')

    ax.set_xticks(list(n))
    ax.set_xticklabels([f'{ni:.0f}' for ni in n])

    set_axes_2d(ax, xlabel='polynomial degree', ylabel='score', title='Bias-variance tradeoff', xlim=(n[0], n[-1]))#, ylim=(0,0.3))
    pdfname = 'ptC_' + pdf_name.strip().replace('ptC_', '') 
    save_push(fig, pdfname, show=show)


    # by no means finished yet
def ptC_bootstrap_hist(bootstrappings, pdf_name='none', show=False):
    fig, ax = plt.subplots()

    N = len(bootstrappings)
    n = np.zeros(N)
    MSE = np.zeros(N)
    bias2 = np.zeros(N)
    var = np.zeros(N)

    i = 0
    for bs in bootstrappings:
        n[i] = bs.polydeg
        if n[i] % 2 == 0:
            print(n[i])

            betas = bs.betas
            beta_means = np.mean(betas.betas, axis=0)
            n_counts, binsboot = np.histogram(beta_means, 50)
            y = norm.pdf(binsboot, np.mean(beta_means), np.std(beta_means))
            ax.plot(binsboot, y, label=f"n: {int(n[i])}")
            # from IPython import embed; embed()
        i += 1
    ax.legend()
    plt.show()