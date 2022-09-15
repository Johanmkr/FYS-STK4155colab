import numpy as np
import seaborn as sns
import os
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
data_path = here + "/../../output/data/"
plot_path = here + "/../../output/figures/"
latex_path = here + "/../../latex/"

def save_push(fig, pdf_name, save=True, push=False, show=False):
    """
    This function handles wether you want to show,
    save and/or push the file to git.
    Args:
        fig (matplotlib.figure): Figure you want to handle
        pdfname (string): Name of output pdf file
        args (argparse)
    """
    file = plot_path + pdf_name.replace('.pdf', '').strip() + ".pdf"
    if save:
        print(f'Saving plot: {file}')
        fig.savefig(file)
    if push:
        os.system(f"git add {file}")
        os.system("git commit -m 'upload plot'")
        os.system("git push")
    if show:
        plt.show()
    if not show:
        plt.close()
    else:
        plt.clf()


def surface_plot(ax, x, y, z):
    surf =  ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.05, 1.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    return ax, surf


def plot_3d_model(data, model, pdf_name='none', show=False):
    ztilde = model()
    fig, (ax0, ax) = plt.subplots(ncols=2, subplot_kw={'projection':'3d'})
    ax0, surf0 = surface_plot(ax0, data.x, data.y, data.z)
    ax, surf = surface_plot(ax, data.x, data.y, model.model)
    ax0.set_title(r"Original data", fontsize=20)
    ax.set_title(r"Polynomial degree $n=%i$"%(model.polydeg), fontsize=20)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()

    if pdf_name != 'none':
        filename = pdf_name.replace('pdf', '') + '.pdf'
        save_push(fig, filename)
    if show:
        plt.show()


def plot_beta_params(models, pdf_name='none', show=False):
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

    if pdf_name != 'none':
        filename = pdf_name.replace('pdf', '') + '.pdf'
        save_push(fig, filename)
    if show:
        plt.show()
    


def plot_bias_variance_tradeoff(models, pdf_name='none', show=False):
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

    if pdf_name != 'none':
        filename = pdf_name.replace('pdf', '') + '.pdf'
        save_push(fig, filename)
    if show:
        plt.show()
    