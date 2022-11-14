
from src.utils import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
import pandas as pd


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


CMAP_ACCURACY = "gnuplot2" # "spring"
CMAP_MSE = CMAP_ACCURACY + "_r"




def save_push(fig, pdf_name, save=True, push=True, show=False, tight=False):
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

def get_last_colour():
    return plt.gca().lines[-1].get_color()



# save useful information
import src.infoFile_ala_Nanna as Nanna
Nanna.init()

Nanna.sayhello()
Nanna.define_categories({
    'method':'method',                      #   steepest descent scheme
    'opt':'optimiser',                      #   update rule name
    'n_obs':r'$n_\mathrm{obs}$',            #   #observations
    'no_epochs':'#epochs',                  #   #epochs
    'no_minibatches':'$m$',                 #   #minibatches
    'lmbda':r'$\lambda$',                   #   penalty parameter
    'eta':r'$\eta$',                        #   global learning rate
    'gamma':r'$\gamma$',                    #   momentum scale
    'rho':r'$\varrho_1$, $\varrho_2$',      #   parameters in adaptive optimisers
    'theta0':r'$\boldsymbol{\theta}_0$',    #   start value
    'L':r'$L-1$',                           #   #hidden layers
    'N':r'$N_l$',                           #   #neurons per layer
    #'note':'note'                           #   comment 
    })  


def set_pdf_info(pdfname:str, **params):
    Nanna.set_file_info(pdfname.replace('.pdf', '') + '.pdf', **params)


def copy_info(info:dict, filename:str):
    params = {} 
    for param, title in Nanna.CATEGORIES.items():
        try:
            params[param] = info[filename][title]
        except KeyError:
            continue
    return params



def simple_regression_polynomial(filenames:list[str], labels:list[str], epochs=(500, 1000), pdfname="untitled", savepush=True, show=True):
    X = np.loadtxt(data_path + "simple_regression/design_matrix.txt", delimiter=",")
    y = np.loadtxt(data_path + "simple_regression/target_data.txt", delimiter=",")
    info1pd = pd.read_pickle(data_path + "simple_regression/info.pkl")
    info1 = info1pd.transpose().to_dict()

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
    
    params = {} 
    for param in ['method', 'n_obs', 'no_epochs', 'no_minibatches', 'lmbda', 'theta0']:
        title = Nanna.CATEGORIES[param]
        try:
            params[param] = info1[filenames[0]][title]
        except KeyError:
            continue
    params['eta'] = '...'
    params['opt'] = '...'

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()

    if savepush:
        save_push(fig, pdf_name=pdfname, tight=False, show=show)
        set_pdf_info(pdfname, **params)
    else:
        if show:
            plt.show()
    

def simple_regression_errors(filenames:list[str], labels:list[str], epochs=(25, 50), pdfname="untitled", savepush=True, show=True):
    info1pd = pd.read_pickle(data_path + "simple_regression/info.pkl")
    info1 = info1pd.transpose().to_dict()
    fig, ax = plt.subplots(layout="constrained")
    for k, file in enumerate(filenames):
        eta, MSE1, MSE2 = np.loadtxt(data_path + "simple_regression/" + file.replace("simple_regression/", ""), delimiter=",")
        ax.plot(eta, MSE1, "o--", lw=2, ms=8, alpha=.8)
        ax.plot(eta, MSE2, "o-",  lw=2, ms=8, alpha=.8, c=get_last_colour(), label=labels[k])
    
    params = {} 
    for param in ['method', 'n_obs', 'no_epochs', 'no_minibatches', 'lmbda', 'theta0']:
        try:
            params[param] = info1[filenames[0]][Nanna.CATEGORIES[param]]
        except KeyError:
            continue
    params['eta'] = '...'
    params['opt'] = '...'


    ax.plot(eta[0], -10, 'o--', lw=2, ms=8, color='grey', label=f"after {epochs[0]} epochs", alpha=0.7)
    ax.plot(eta[0], -10, 'o-',  lw=2, ms=8, color='grey', label=f"after {epochs[1]} epochs", alpha=0.7)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, 1.0)
    ax.set_xscale("log")
    ax.legend()

    if savepush:
        save_push(fig, pdf_name=pdfname, tight=False, show=show)
        set_pdf_info(pdfname, **params)
    else:
        if show:
            plt.show()
    

def MSEheatmap_plot(filename, pdfname='untitled', savepush=False, show=True, xlabel=r"$\eta$", ylabel=r"$\lambda$"):
    info2pd = pd.read_pickle(data_path + "network_regression/info.pkl")
    info2 = info2pd.transpose().to_dict()
    fig, ax = plt.subplots(layout='constrained', figsize=(13,11))
    score = pd.read_pickle(data_path+"network_regression/"+filename)
    sns.heatmap(score, annot=True, ax=ax, cmap=CMAP_MSE, vmin=0, vmax=1, cbar_kws={'label': "Test MSE", "extend": "max"})
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savepush:
        save_push(fig, pdf_name=pdfname, tight=False, show=True)
        params = copy_info(info2, filename)
        set_pdf_info(pdfname, **params)
    else:
        if show:
            plt.show()

def epoch_plot(filename, pdfname="untitled", savepush=False, show=True):
    info2pd = pd.read_pickle(data_path + "network_regression/info.pkl")
    info2 = info2pd.transpose().to_dict()
    fig, ax = plt.subplots(layout="constrained")
    score = pd.read_pickle(data_path+"network_regression/"+filename)
    for func in score.columns:
        if func == "epochs":
            pass
        else:
            x = np.asarray(score["epochs"])
            y = np.asarray(score[func])
            ax.plot(x,y, label=func)
    # ax.set_ylim(0,1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MSE")
    plt.legend()

    if savepush:
        save_push(fig, pdf_name=pdfname, tight=False, show=True)
        params = copy_info(info2, filename)
        set_pdf_info(pdfname, **params)
    else:
        if show:
            plt.show()

def CancerHeatmap_plot(filename, pdfname='untitled', savepush=False, show=True, xlabel=r"$\eta$", ylabel=r"$\lambda$"):
    info3pd = pd.read_pickle(data_path + "network_classification/info.pkl")
    info3 = info3pd.transpose().to_dict()
    fig, ax = plt.subplots(layout='constrained', figsize=(13,11))
    score = pd.read_pickle(data_path+"network_classification/"+filename)
    sns.heatmap(score, annot=True, ax=ax, cmap=CMAP_ACCURACY, vmin=0, vmax=1, cbar_kws={'label': "Test accuracy", "extend": "max"})
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savepush:
        save_push(fig, pdf_name=pdfname, tight=False, show=True)
        params = copy_info(info3, filename)
        set_pdf_info(pdfname, **params)
    else:
        if show:
            plt.show()

def Cancerepoch_plot(filename, pdfname="untitled", savepush=False, show=True):
    info3pd = pd.read_pickle(data_path + "network_classification/info.pkl")
    info3 = info3pd.transpose().to_dict()
    fig, ax = plt.subplots(layout="constrained")
    score = pd.read_pickle(data_path+"network_classification/"+filename)
    for func in score.columns:
        if func == "epochs":
            pass
        else:
            x = np.asarray(score["epochs"])
            y = np.asarray(score[func])
            ax.plot(x,y, label=func)
    # ax.set_ylim(0,1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    plt.legend()

    if savepush:
        save_push(fig, pdf_name=pdfname, tight=False, show=True)
        params = copy_info(info3, filename)
        set_pdf_info(pdfname, **params)
    if not savepush:
        if show:
            plt.show()


def update():
    Nanna.update()