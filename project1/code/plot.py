from distutils.log import info
from ensurepip import bootstrap
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd


# The style we want
plt.style.use('seaborn')
#plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

# other rc parameters
plt.rc('figure', figsize=(12,7))
SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
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

def read_info():
    global INFO
    infopd = pd.read_pickle(plot_path + 'info.pkl')
    INFO = infopd.transpose().to_dict()

def add_path(folder, read=True):
    global plot_path
    if folder.lower() in ['franke', 'frankefig', 'frankefigs']:
        folder = 'Franke/'
    elif folder.lower() in ['terrain', 'terran', 'terraindata']:
        folder = 'terrain/'
    else:
        raise Warning("Folder name not accepted. Figures are saved to main figure folder.")
        folder = ''
    plot_path += folder
    global show_path
    show_path = '/output/figures/' + folder

    if read:
        read_info()

'''b = pd.DataFrame({'method':'ols', 'mode':'own', 'tull':0}, ['dummy'])
b.to_pickle('info.pkl')
'''
def save_push(fig, pdf_name, save=True, push=True, show=False, tight=True):
    """
    This function handles wether you want to show, save and/or push the file to git.
    Args:
        fig (matplotlib.figure): Figure you want to handle
        pdfname (string): Name of output pdf file
        args (argparse)
    """
    if tight:
        fig.tight_layout()
    file = plot_path + pdf_name.replace('.pdf', '').strip() + ".pdf"
    if testMode:
        plt.show()
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



def tune_hyper_parameter(resamplings):
    complexity = []
    hyper_parameter = []
    for i, rs in enumerate(resamplings):
        complexity.append(rs.polydeg)
        hyper_parameter.append(rs.lmbda)
    
    if complexity[1] != complexity[0]:
        tuning = False
    elif hyper_parameter[1] != hyper_parameter[0]:
        tuning = True
    else:
        raise TypeError("Make sure that either the model complexity or the hyper parameter stays constant in the resampling.")

    return tuning
    
def complexity_measure(resamplings):
    return not tune_hyper_parameter()

def complexity_or_tuning(complexity, hyper_parameter):

    complexity_measure, tune_hyper_parameter = False, False
    if complexity[1] != complexity[0]:
        complexity_measure = True
        variable = complexity
        xlabel = r'polynomial degree $d$'
        xscale = 'linear'
    elif hyper_parameter[1] != hyper_parameter[0]:
        tune_hyper_parameter = True
        variable = hyper_parameter
        xlabel = r'hyper parameter $\lambda$'
        xscale = 'log'
    else:
        raise TypeError("Make sure that either the model complexity or the hyper parameter stays constant in the resampling.")
    
    return variable, xlabel, xscale



def MSE_train_test(ax, resamplings, errorbar=False):
    N = len(resamplings) # no of models
    complexity = np.zeros(N)
    hyper_parameter = np.zeros(N)
    Niter = resamplings[1].Niter # b, k
    trainMSE = np.zeros((N,Niter))
    testMSE = np.zeros((N,Niter))

    for i, rs in enumerate(resamplings):
        complexity[i] = rs.polydeg
        hyper_parameter[i] = rs.lmbda
        trainMSE[i], testMSE[i] = rs.mean_squared_error()

    variable, xlabel, xscale = complexity_or_tuning(complexity, hyper_parameter)

    ax.plot(variable, trainMSE, lw=0.3, c=dTRAIN['c'], alpha=0.2)
    ax.plot(variable, testMSE,  lw=0.3, c=dTEST['c'],  alpha=0.2)

    if errorbar:
        errTrain, errTest = np.std(trainMSE, axis=1), np.std(testMSE, axis=1)
    else:
        errTrain, errTest = None, None

    muTrain, muTest = np.mean(trainMSE, axis=1), np.mean(testMSE, axis=1)

    markers, caps, bars = ax.errorbar(variable, muTrain, errTrain, c=dTRAIN['c'], lw=1.1, alpha=0.9, label='train MSE')
    markers, caps, bars = ax.errorbar(variable, muTest,  errTest,  c=dTEST['c'],  lw=1.1, alpha=0.9, label='test MSE')   
   
    ymax = np.max([muTrain[0], muTest[0]])*1.25
    set_axes_2d(ax, xlabel=xlabel, ylabel='score',  ylim=(0,ymax))
    ax.set_xscale(xscale)



def bias_variance_tradeoff(ax, bootstrappings):
    N = len(bootstrappings)
    complexity = np.zeros(N) 
    hyper_parameter = np.zeros(N)
    error = np.zeros(N)
    bias2 = np.zeros(N)
    var = np.zeros(N)

    for i, bs in enumerate(bootstrappings):
        complexity[i] = bs.polydeg
        hyper_parameter[i] = bs.lmbda
        error[i] = bs.error
        bias2[i] = bs.bias2
        var[i] = bs.var

    variable, xlabel, xscale = complexity_or_tuning(complexity, hyper_parameter)
    
    ax.plot(variable, error, ls=dTEST['ls'], c=dMSE['c'],  label='error')
    ax.plot(variable, bias2, ls=dTEST['ls'], c=dBIAS['c'], label='bias$^2$')
    ax.plot(variable, var,   ls=dTEST['ls'], c=dVAR['c'],  label='variance')

    ymax = np.max([error[0], bias2[0], var[0]])*1.1
    set_axes_2d(ax, xlabel=xlabel, ylabel='score', ylim=(0,ymax))
    ax.set_xscale(xscale)




def make_histogram(ax, bootstrap, mse=True):

    polydeg = bootstrap.polydeg
    B = bootstrap.B
    bins = int(B/4)
    if mse:
        MSEs = bootstrap.mean_squared_error()
        ax.hist(MSEs[0], bins, alpha=.7, label='train')
        ax.hist(MSEs[1], bins, alpha=.7, label='test')
        legend = True
        xlabel = 'MSE'
    else:
        _, beta = bootstrap.getOptimalParameters()
        ax.hist(beta, bins, alpha=.7)
        legend = False
        xlabel = r'$\bar{\beta}$'

    set_axes_2d(ax, xlabel=xlabel, ylabel='frequency', title=r'polynomial degree %i (%i bootstraps)'%(polydeg, B), legend=legend)
   


### define styles:
## believe I will change this...

dTRAIN = {'ls':'--', 'c':'b'}
dTEST = {'ls':'-',  'c':'r'}

dMSE = {'c':'firebrick'}
dR2 = {'c':'olive'}
dBIAS = {'c':'dodgerblue'}
dVAR = {'c':'orange'}




def visualise_data(x, y, z):
    fig, ax = plt.subplots(ncols=1, subplot_kw={'projection':'3d'})
    ax, surf = surface_plot(ax, x, y, z)
    ax.set_zlim(np.min(z), np.max(z))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()

    save_push(fig, "none", save=False, show=True)





def set_info(pdfname, method=None, mode=None, polydeg=None, lmbda=None, resampling_type=None, resampling_iter='', mark=None):
    if pdfname in INFO.keys():
        pass

    INFO[pdfname] = {'method':method, 'mode':mode, 'polynomial degree':polydeg, 'lambda':lmbda}
    if resampling_type == None:
        INFO[pdfname]['resampling (iter)'] = None
    else:
        INFO[pdfname]['resampling (iter)'] = f'{resampling_type} ({resampling_iter})'
    INFO[pdfname]['mark'] = mark









def train_test_MSE(resamplings, tag='', show=False, mark=None):
    fig, ax = plt.subplots()
    MSE_train_test(ax, resamplings)
    res = resamplings[0]
    method = res.method
    pdfname = f'MSE_{method}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    try:
        Niter = res.B
        type = 'BS'
    except AttributeError:
        Niter = res.k
        type = 'CV'
    if tune_hyper_parameter(resamplings):
        polydeg = res.polydeg
        lmbda = '...'
    else:
        polydeg = '...'
        lmbda = res.lmbda

    set_info(pdfname, method, resamplings[0].mode, polydeg=polydeg, lmbda=lmbda, resampling_type=type, resampling_iter=Niter, mark=mark)


def BV_Tradeoff(bootstrappings, tag='', show=False, mark=None):
    fig, ax = plt.subplots()

    bias_variance_tradeoff(ax, bootstrappings)
    method = bootstrappings[0].method
    pdfname = f'tradeoff_{method}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, method, bootstrappings[0].mode, resampling_type='BS', resampling_iter=bootstrappings[0].B, mark=mark)



def beta_params(trainings, tag='', show=False, mark=None):
    fig, ax = plt.subplots()

    for d in trainings.keys():
        beta = trainings[d].pV
        markers, caps, bars = ax.errorbar(range(1,beta.nfeatures+1), beta[:], beta.stdv, fmt='o', ms='10',  ls='--', lw=0.8,label=r'$d=%i$'%d, capsize=5, capthick=2, elinewidth=1.2) 
        [bar.set_alpha(0.7) for bar in bars]
        [cap.set_alpha(0.9) for cap in caps]

    t = trainings[max(trainings.keys())]
    b = t.pV
    Bmax = b.nfeatures
    ax.set_xticks(range(1,Bmax+1))
    ax.set_xticklabels(b.idx_tex)
   
    padx = 1/3; pady = np.max(np.abs(b[:]))*0.1
    set_axes_2d(ax, xlim=(0-padx,Bmax-1+padx), ylim=(np.min(b[:])-pady, np.max(b[:])+pady))
    pdfname = f'beta_{t.method}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, t.method, t.mode, lmbda=t.lmbda, mark=mark)


def hist_resampling(bootstrap, which='mse', tag='', show=False, mark=None):
    fig, ax = plt.subplots()
    if which.lower() == 'mse':
        make_histogram(ax, bootstrap)
        which = 'MSE'
    else:
        make_histogram(ax, bootstrap, False)
        which = 'beta'

    pdfname = f'{which}_hist_{bootstrap.method}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, bootstrap.method, bootstrap.mode, polydeg=bootstrap.polydeg, resampling_type='BS', resampling_iter=bootstrap.B, lmbda=bootstrap.lmbda, mark=mark)


def train_test_MSE_R2(trainings, predictions, tag='', show=False, mark=None):

    fig, axes = plt.subplots(nrows=2, sharex=True)
    ax1, ax2 = axes.flat

    N = len(trainings)
    polydegs = np.zeros(N)
    MSE = np.zeros((2,N))
    R2 = np.zeros((2,N))

    i = 0
    for d in trainings.keys():
        train, test = trainings[d], predictions[d]
        polydegs[i] = d
        MSE[:,i] = train.MSE, test.MSE
        R2[:,i] = train.R2, test.R2
        i+=1
    
    ax1.plot(polydegs, MSE[0], lw=2, ls=dTRAIN['ls'],c=dMSE['c'], label='train MSE')
    ax1.plot(polydegs, MSE[1], lw=2, ls=dTEST['ls'], c=dMSE['c'], label='test MSE')
    ax2.plot(polydegs, R2[0],  lw=2, ls=dTRAIN['ls'],c=dR2['c'],  label='train $R^2$')
    ax2.plot(polydegs, R2[1],  lw=2, ls=dTEST['ls'], c=dR2['c'],  label='test $R^2$')

    ax2.set_xticks(list(polydegs))
    ax2.set_xticklabels([f'{d:.0f}' for d in polydegs])

    set_axes_2d(ax1, ylabel='score')
    pad = 1/4
    set_axes_2d(ax2, xlabel='polynomial degree', ylabel='score', xlim=(polydegs[0]-pad, polydegs[-1]+pad))
    method = trainings[d].method
    pdfname = f'MSE_R2_scores_{method}{tag}.pdf'
    save_push(fig, pdfname, show=show)

    set_info(pdfname, method, trainings[d].mode, lmbda=trainings[d].lmbda, mark=mark)








def update_info():
    if testMode:
        pass
    else:

        infopd = pd.DataFrame.from_dict(INFO, orient='index')
        # sort ?
        
        
        infopd.to_pickle(plot_path + 'info.pkl')
        infopd.to_markdown(plot_path + 'README.md')

        with open(plot_path + 'README.md', 'a') as file:
            file.write('\n\n\n')
            file.write(f'# Information about plots in `{show_path}`')
            file.write('\n\n')