import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

# other rc parameters
plt.rc('figure', figsize=(12,7))
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
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
            fig.savefig(file)
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

dTRAIN = {'ls':'--'}
dTEST = {'ls':'-'}

dMSE = {'c':'firebrick'}
dR2 = {'c':'olive'}
dBIAS = {'c':'dodgerblue'}
dVAR = {'c':'orange'}



'''
# Pt. b)
'''


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
        beta = train.beta
        ax.errorbar(range(train.features), beta[:], beta.stdv, fmt='o', ms='10',  ls='', label=r'$n=%i$'%train.polydeg)

    B = train.features # max
    ax.set_xticks(range(B))
    ax.set_xticklabels([r'$\beta_{%i}$'%i for i in range(B)])


    set_axes_2d(ax, title=r'$\beta$ for various polynomial degrees $n$', xlim=(0,B-1))

    pdfname = 'ptB_' + pdf_name.strip().replace('ptB_', '') 
    save_push(fig, pdfname, show=show)




"""  
Pt. c)
"""


def ptC_Hastie(trainings, predictions, pdf_name='none', show=False):
    fig, ax = plt.subplots()

    N = len(trainings)
    n = np.zeros(N)
    MSE = np.zeros((2,N))

    i = 0
    for train, test in zip(trainings, predictions):
        n[i] = train.polydeg

        MSE[0,i] = train.MSE
        MSE[1,i] = test.MSE

        i += 1
    
    ax.plot(n, MSE[0], ls=dTRAIN['ls'],c=dMSE['c'], label='train MSE')
    ax.plot(n, MSE[1], ls=dTEST['ls'], c=dMSE['c'], label='test MSE')

    ax.set_xticks(list(n))
    ax.set_xticklabels([f'{ni:.0f}' for ni in n])

    set_axes_2d(ax, xlabel='polynomial degree', ylabel='score', title='Mean squared error', xlim=(n[0], n[-1]))
    pdfname = 'ptC_' + pdf_name.strip().replace('ptC_', '') 
    save_push(fig, pdfname, show=show)


def ptC_tradeoff(bootstrappings, pdf_name='none', show=False):
    fig, ax = plt.subplots()

    N = len(bootstrappings)
    n = np.zeros(N)
    MSE = np.zeros(N)
    bias2 = np.zeros(N)
    var = np.zeros(N)

    i = 0
    for bs in bootstrappings:
        n[i] = bs.polydeg

        MSE[i] = bs.MSE
        bias2[i] = bs.bias2
        var[i] = bs.var

        i += 1
    
    ax.plot(n, MSE,   ls=dTEST['ls'], c=dMSE['c'],  label='error')
    ax.plot(n, bias2, ls=dTEST['ls'], c=dBIAS['c'], label='bias$^2$')
    ax.plot(n, var,   ls=dTEST['ls'], c=dVAR['c'],  label='variance')

    ax.set_xticks(list(n))
    ax.set_xticklabels([f'{ni:.0f}' for ni in n])

    set_axes_2d(ax, xlabel='polynomial degree', ylabel='score', title='Bias-variance tradeoff', xlim=(n[0], n[-1]))#, ylim=(0,0.3))
    pdfname = 'ptC_' + pdf_name.strip().replace('ptC_', '') 
    save_push(fig, pdfname, show=show)