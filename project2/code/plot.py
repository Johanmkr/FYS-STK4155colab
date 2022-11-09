
from src.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import pandas as pd
from IPython import embed

# The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

# other rc parameters
plt.rc('figure', figsize=(12,7))
SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 30
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# folder paths
here = os.path.abspath(".")
data_path = here + "/../output/data/"
plot_path = here + "/../output/figures/"
latex_path = here + "/../latex/"


def save_push(fig, pdf_name, save=True, push=True, show=False, tight=True):
    if tight:
        fig.tight_layout()
    file = plot_path + pdf_name.replace('.pdf', '').strip() + ".pdf"
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
    if xlim != 'none':
        ax.set_xlim(xlim)
    if ylim != 'none':
        ax.set_ylim(ylim)

    if xlabel != 'none':
        if xlabel == 'eta':
            ax.set_xlabel(r'learning rate $\eta$')
            ax.set_xscale('log')
        else:
            ax.set_xlabel(xlabel)
    if ylabel != 'none':
        if ylabel == 'eta':
            ax.set_ylabel(r'learning rate $\eta$')
            ax.set_yscale('log')
        else:
            ax.set_ylabel(ylabel)

    if title != 'none':
        ax.set_title(title)
    if legend:
        ax.legend()

def get_last_colour():
    return plt.gca().lines[-1].get_color()



# save useful information
import src.infoFile_ala_Nanna as Nanna
Nanna.init()

Nanna.sayhello()
Nanna.define_categories({'method':'method', 'opt':'optimiser', 'n_obs':r'$n_\mathrm{obs}$', 'no_epochs':'#epochs', 'eta':r'$\eta$', 'gamma':r'$\gamma$', 'rho':r'$\varrho_1$, $\varrho_2$'})


def set_pdf_info(pdfname, **params):
    Nanna.set_file_info(pdfname.replace('.pdf', '') + '.pdf', **params)




def simple_regression_polynomial(X:ndarray, y:ndarray, filenames:list[str], labels:list[str], epochs=(500, 1000), pdfname="untitled", savepush=True, show=True):
    x = X[:,0]
    #X = np.sort(X, axis=0)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(x, y, 'o', ms=7, alpha=.7, color='k', label="data")
    for k, file in enumerate(filenames):
        f = data_path + "simple_regression/" + file.replace("simple_regression/", "",)
        theta_opt = []
        with open(f, 'r') as infile:
            theta_str = infile.readline()
            theta_str = theta_str.split('=')[1].strip().strip('[').strip(']')
            theta_list = theta_str.split()
            for theta_j in theta_list:
                theta_opt.append(float(theta_j))
        theta_opt = np.array(theta_opt)
        eta, MSE1, MSE2 = np.loadtxt(f, delimiter=",")
        eta_opt = eta[np.argmin(MSE2)]
        #ax.plot(X[:,0], X@theta_opt, 'o', lw=1.5, ms=4, label=labels[k] + r" ($\eta = %.2f \cdot 10^{-3}$)" %(eta_opt*1e3))
        ax.plot(X[:,0], X@theta_opt, lw=1.2, alpha=.8, label=labels[k] + r" ($\eta = %.2f \cdot 10^{-3}$)" %(eta_opt*1e3))

    set_axes_2d(ax, xlabel=r"$x$", ylabel=r"$y$")
    if save_push:
        save_push(fig, pdf_name=pdfname, tight=False, show=show)
    if not save_push:
        plt.show()


    

def simple_regression_errors(filenames:list[str], labels:list[str], epochs=(500, 1000), pdfname="untitled", save_push=True, show=True):
    fig, ax = plt.subplots(layout="constrained")
    for k, file in enumerate(filenames):
        eta, MSE1, MSE2 = np.loadtxt(data_path + "simple_regression/" + file.replace("simple_regression/", ""), delimiter=",")
        ax.plot(eta, MSE1, "o--", lw=2, ms=8, alpha=.8)
        ax.plot(eta, MSE2, "o-",  lw=2, ms=8, alpha=.8, c=get_last_colour(), label=labels[k])
    
    ax.plot(eta[0], -10, 'o--', lw=2, ms=8, color='grey', label=f"after {epochs[0]} epochs", alpha=0.7)
    ax.plot(eta[0], -10, 'o-',  lw=2, ms=8, color='grey', label=f"after {epochs[1]} epochs", alpha=0.7)

    set_axes_2d(ax, xlabel="eta", ylabel="MSE", ylim=(0,1.0))

    save_push(fig, pdf_name=pdfname, tight=False, show=show)
    

def heatmap_plot(filename, pdfname='untitled', save_push=False, show=True):
    fig, ax = plt.subplots(layout='constrained')
    score = pd.read_pickle(data_path+"network_regression/"+filename)
    sns.heatmap(score, annot=True, ax=ax, cmap='viridis', vmax=1)
    # ax.set_xticklabels([f"{x:.2e}" for x in x])
    # set_axes_2d(ax=ax, xlabel="eta", ylabel="eta")
    # ax.set_yticklabels([f"{y:.2e}" for y in y])
    ax.invert_yaxis()
    ax.set_title("Test score")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\lambda$")

    if save_push:
        save_push(fig, pdf_name=pdfname, tight=False, show=True)
    if not save_push:
        plt.show()

def epoch_plot(filename, pdfname="untitled", save_push=False):
    fig, ax = plt.subplots(layout="constrained")
    score = pd.read_pickle(data_path+"network_regression/"+filename)
    for func in score.columns:
        if func == "epochs":
            pass
        else:
            x = np.asarray(score["epochs"])
            y = np.asarray(score[func])
            ax.plot(x,y, label=func)
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("MSE")
    set_axes_2d(ax, xlabel="Epoch", ylabel="MSE")

    if save_push:
        save_push(fig, pdf_name=pdfname, tight=False, show=True)
    if not save_push:
        plt.show()


def update():
    Nanna.update()