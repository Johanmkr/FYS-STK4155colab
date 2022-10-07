
from audioop import cross
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
import matplotlib.ticker as ticker
import pandas as pd
from collections import OrderedDict

# The style we want
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('font', family='DejaVu Sans')

# other rc parameters
plt.rc('figure', figsize=(12,7))
SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 30
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
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
        if xlabel == 'Complexity':
            ax.set_xlabel(r'polynomial degree $d$')
            ax.xaxis.get_major_locator().set_params(integer=True)
        elif xlabel == 'HyperParameter':
            ax.set_xlabel(r'hyper parameter $\lambda$')
            ax.set_xscale('log')
        else:
            ax.set_xlabel(xlabel)
    if ylabel != 'none':
        if ylabel == 'Complexity':
            ax.set_ylabel(r'polynomial degree $d$')
            ax.yaxis.get_major_locator().set_params(integer=True)
        if ylabel == 'HyperParameter':
            ax.set_ylabel(r'hyper parameter $\lambda$')
            ax.set_yscale('log')
        else:
            ax.set_ylabel(ylabel)
    if title != 'none':
        ax.set_title(title)
    if legend:
        ax.legend()

    if xlim != 'none':
        ax.set_xlim(xlim)
    if ylim != 'none':
        ax.set_ylim(ylim)
    




boxStyle = {'facecolor':'lavender', 'alpha':0.7, 'boxstyle':'round'}

def surface_plot(ax, x, y, z, angles, cmap=cm.coolwarm):
    if len(np.shape(x)) == 2:
        surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
    else: 
        surf = ax.plot_trisurf(x, y, z, cmap=cmap, linewidth=0, antialiased=False)
    ax.view_init(*angles)
    ax.set_zlim(np.min(z), np.max(z))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    return ax, surf


def tune_hyper_parameter(resamplings):
    complexity = []
    hyper_parameter = []
    for i, rs in enumerate(resamplings):
        complexity.append(rs.polydeg)
        hyper_parameter.append(rs.lmbda)
        if i >= 2:
            break
    
    if complexity[1] != complexity[0]:
        tuning = False
    elif hyper_parameter[1] != hyper_parameter[0]:
        tuning = True
    else:
        raise TypeError("Make sure that either the model complexity or the hyper parameter stays constant in the resampling.")

    return tuning
    
def complexity_measure(resamplings):
    return not tune_hyper_parameter(resamplings)


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
    
    if tune_hyper_parameter(resamplings):
        variable = hyper_parameter
        xlabel = 'HyperParameter'
        elem = -1
    else:
        variable = complexity
        xlabel = 'Complexity'
        elem = 0

    if Niter < 20:
        alpha = 0.4
        lw = 0.6
    else:
        alpha = 1/Niter**(1/3)
        lw = 0.2
    
    
    ax.plot(variable, trainMSE, ls='--', lw=lw, c='dodgerblue', alpha=alpha) 
    ax.plot(variable, testMSE,  ls='-',  lw=lw, c='salmon',     alpha=alpha)

    muTrain, muTest = np.mean(trainMSE, axis=1), np.mean(testMSE, axis=1)
    if errorbar:
        errTrain, errTest = np.std(trainMSE, axis=1), np.std(testMSE, axis=1)
        markers, caps, bars = ax.errorbar(variable, muTrain, errTrain, c='royalblue', lw=1.8, ls='--', label='train MSE')
        markers, caps, bars = ax.errorbar(variable, muTest,  errTest,  c='orangered', lw=1.8, ls='-',  label='test MSE')  
    else:
        ax.plot(variable, muTrain, 'o--', c='royalblue', lw=1.8, label='train MSE')
        ax.plot(variable, muTest,  'o-',  c='orangered', lw=1.8, label='test MSE') 


    ymax = np.max([muTrain[elem], muTest[elem]])*1.25
    ymin = np.max([0, np.min([muTrain, muTest])*0.90])
    set_axes_2d(ax, xlabel=xlabel, ylabel='score', xlim=(variable[0], variable[-1]),  ylim=(ymin,ymax))


def MSE_test(ax, crossvalidations, errorbar=False):

    colours = {5:'cornflowerblue', 6:'mediumseagreen', 7:'coral', 8:'palevioletred', 9:'darkkhaki', 10:'mediumpurple'}
    ymin = 10
    ymax = 0
    for i in range(len(crossvalidations)):
        cv_list = crossvalidations[i]
        N = len(cv_list)
        if i == 0:
            tuning = tune_hyper_parameter(cv_list)
            variable = np.zeros(N)
        res_error = np.zeros(N)
        for j in range(N):
            cv = cv_list[j]
            k = len(cv) # number of folds
            if i == 0:
                if tuning:
                    variable[j] = cv.lmbda
                else:
                    variable[j] = cv.polydeg
            res_error[j] = cv.resamplingError()
        
        ymin = np.min([ymin, np.min(res_error)*0.9])
        if tuning:
            ymax = np.max([ymax, res_error[-1]*1.10])
        else:
            ymax = np.max([ymax, res_error[0]*1.25])
        ax.plot(variable, res_error, 'o-', lw=2.2, c=colours[k], alpha=0.75, label=r'$k=%i$'%k)


    if tuning:
        xlabel = 'HyperParameter'
    else:
        xlabel = 'Complexity'

    set_axes_2d(ax, xlabel=xlabel, ylabel='CV accuracy', xlim=(variable[0], variable[-1]),  ylim=(ymin,ymax))




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

    tuning = tune_hyper_parameter(bootstrappings)
    if tuning:
        variable = hyper_parameter
        xlabel = 'HyperParameter'
    else:
        variable = complexity
        xlabel = 'Complexity'

    ax.plot(variable, error, ls='-', lw=2.5, c='r', label='error')
    ax.plot(variable, bias2, ls='-', lw=2.5, c='b', label='bias$^2$')
    ax.plot(variable, var,   ls='-', lw=2.5, c='g', label='variance')

    if tuning:
        ymax = np.max([error[-1], bias2[-1], var[-1]])*1.1
    else:
        ymax = np.max([error[0], bias2[0], var[0]])*1.1

    set_axes_2d(ax, xlabel=xlabel, ylabel='score', ylim=(0,ymax))




def make_histogram(ax, bootstrap, which):

    B = bootstrap.B
    bins = int(B/3)
    if which.lower() == 'mse':
        MSEs = bootstrap.mean_squared_error()
        ax.hist(MSEs[0], bins, alpha=.7, label='train')
        ax.hist(MSEs[1], bins, alpha=.7, label='test')
        xlabel = 'MSE'
    else:
        beta, beta_grouped, beta_mean = bootstrap.getOptimalParameters()
        
        if which.lower() == 'beta_mean':
            ax.hist(beta_mean, bins, alpha=.7)
            xlabel = r'$\bar{\beta}$'
        elif which.lower() == 'beta_grouped':
            b0 = beta_grouped[0]
            bins = 10
            for i in range(len(b0)):
                ax.hist(beta_grouped[:][i], bins, alpha=.4, label=b0.idx_tex[i])
            xlabel = r'$\beta^{(d_i)}$'
        else:
            raise ValueError('Provide valid histogram measure.')
    
    ax.text(0.8,0.9, str(bootstrap), transform=ax.transAxes, va='top', fontsize=SMALL_SIZE, bbox=boxStyle)
    set_axes_2d(ax, xlabel=xlabel, ylabel='frequency', legend=False)
    if which.lower() != 'beta_mean':
        ax.legend(loc='center right')
   

def data_model_comparison(ax, z_exp, X_exp, z_comp, X_comp, angles, cmap):
    ax, surf = surface_plot(ax, X_comp[:,0], X_comp[:,1], z_comp[:], angles, cmap) #computed
    ax.scatter(X_exp[:,0], X_exp[:,1], z_exp[:], marker='^', c='navy') #expected
    return ax







def set_info(pdfname, scheme=None, mode=None, polydeg=None, lmbda=None, resampling_type=None, resampling_iter='', mark=None):
    if scheme == 'ols':
        scheme = 'OLS'
    elif scheme == 'ridge':
        scheme = 'Ridge'
    elif scheme == 'lasso':
        scheme = 'Lasso'

    INFO[pdfname] = {'scheme':scheme, 'mode':mode, '$d$':polydeg, '$\lambda$':lmbda}
    if resampling_type == None:
        INFO[pdfname]['resampling (iter)'] = None
    else:
        INFO[pdfname]['resampling (iter)'] = f'{resampling_type} ({resampling_iter})'
    INFO[pdfname]['mark'] = mark






def visualise_data(z_data, X_data, angles=(40, 30), cmap='default', tag='', show=False, mark=None):
    x = X_data[:,0]; y = X_data[:,1]; z = z_data[:]
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'}, figsize=(9,7))
    if cmap == 'default':
        ax, surf = surface_plot(ax, x, y, z, angles, cm.coolwarm)
    elif cmap == 'terrain':
        ax, surf = surface_plot(ax, x, y, z, angles, cm.terrain)
    pdfname = f'data3D{tag}.pdf'
    fig.subplots_adjust(left=0.02, bottom=0.04, right=0.96, top=0.96)
    save_push(fig, pdfname, show=show, tight=False)
    if mark == None:
        mark = 'visualise x, y, z data'
    set_info(pdfname, mark=mark)





def compare_data(expected, computed, angles=(40, 30), cmap='default', tag='', show=False, mark=None):
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'}, figsize=(9,7))
    if cmap == 'default':
        cmapp = cm.coolwarm
    else:
        cmapp = cm.terrain
    ax = data_model_comparison(ax, expected.tV, expected.dM, computed.mV, computed.dM, angles=angles, cmap=cmapp)
    fig.text(0.8, 0.9, str(computed), ha='right', va='top', fontsize=SMALL_SIZE, bbox=boxStyle)
    scheme = computed.scheme.lower()
    pdfname = f'comparison3D_{scheme}{tag}.pdf'
    fig.subplots_adjust(left=0.02, bottom=0.04, right=0.96, top=0.96)
    save_push(fig, pdfname, show=show, tight=False)
    set_info(pdfname, scheme, mode=computed.mode, polydeg=computed.polydeg, lmbda=computed.lmbda, mark=mark)




def train_test_MSE(resamplings,  tag='', show=False, mark=None):
    fig, ax = plt.subplots(layout="constrained")
    MSE_train_test(ax, resamplings)
    res = resamplings[0]
    scheme = res.scheme.lower()
    type = res.ID()
    Niter = len(res)
    if tune_hyper_parameter(resamplings):
        polydeg = res.polydeg
        lmbda = '...'
        ax.text(0.8, 0.9, r'$d=%i$'%polydeg, transform=ax.transAxes, ha='right', va='top', fontsize=SMALL_SIZE, bbox=boxStyle)
    else:
        polydeg = '...'
        lmbda = res.lmbda
    pdfname = f'MSE_{scheme}_{type}{tag}.pdf'
    save_push(fig, pdfname, show=show, tight=False)
    set_info(pdfname, scheme, resamplings[0].mode, polydeg=polydeg, lmbda=lmbda, resampling_type=type, resampling_iter=Niter, mark=mark)


def CV_errors(crossvalidations_list, tag='', show=False, mark=None):
    fig, ax = plt.subplots(layout="constrained")
    if not isinstance(crossvalidations_list[0], list):
        crossvalidations_list = [crossvalidations_list]
    MSE_test(ax, crossvalidations_list)
    cv_list = crossvalidations_list[0]
    cv = cv_list[0]
    scheme = cv.scheme.lower()
    if tune_hyper_parameter(cv_list):
        polydeg = cv.polydeg
        lmbda = '...'
        ax.text(0.8, 0.9, r'$d=%i$'%polydeg, transform=ax.transAxes, ha='right', va='top', fontsize=SMALL_SIZE, bbox=boxStyle)
    else:
        polydeg = '...'
        lmbda = cv.lmbda
    pdfname = f'MSE_{scheme}_CV{tag}.pdf'
    save_push(fig, pdfname, show=show, tight=False)
    set_info(pdfname, scheme, cv.mode, polydeg=polydeg, lmbda=lmbda, resampling_type='CV', resampling_iter='...', mark=mark)




def BV_Tradeoff(bootstrappings, tag='', show=False, mark=None):
    fig, ax = plt.subplots(layout="constrained")
    bias_variance_tradeoff(ax, bootstrappings)
    bs = bootstrappings[0]
    scheme = bs.scheme.lower()
    if tune_hyper_parameter(bootstrappings):
        polydeg = bs.polydeg
        lmbda = '...'
        ax.text(0.8, 0.9, r'$d=%i$'%polydeg, transform=ax.transAxes, ha='right', va='top', fontsize=SMALL_SIZE, bbox=boxStyle)
    else:
        polydeg = '...'
        lmbda = bs.lmbda
    pdfname = f'tradeoff_{scheme}{tag}.pdf'
    save_push(fig, pdfname, show=show, tight=False)
    set_info(pdfname, scheme, bs.mode, polydeg=polydeg, lmbda=lmbda, resampling_type='BS', resampling_iter=bs.B, mark=mark)



def beta_params(trainings, grouped=False, tag='', show=False, mark=None):
    fig, ax = plt.subplots()

    for d in trainings.keys():
        if grouped:
            beta = trainings[d].pV.group()
        else:
            beta = trainings[d].pV
        markers, caps, bars = ax.errorbar(range(1,beta.nfeatures+1), beta[:], beta.stdv, fmt='o', ms='10',  ls='--', lw=0.8, label=r'$d=%i$'%d, capsize=5, capthick=2, elinewidth=1.2) 
        [bar.set_alpha(0.7) for bar in bars]
        [cap.set_alpha(0.9) for cap in caps]

    t = trainings[max(trainings.keys())]
    if grouped:
        b = t.pV.group()
    else:
        b = t.pV
    Bmax = b.nfeatures
    ax.set_xticks(range(1, Bmax+1))
    ax.set_xticklabels(b.idx_tex)
   
    padx = 1/3; pady = np.max(np.abs(b[:]))*0.1
    set_axes_2d(ax, xlim=(1-padx, Bmax+padx), ylim=(np.min(b[:])-pady, np.max(b[:])+pady))
    scheme = t.scheme.lower()
    pdfname = f'beta_{scheme}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, scheme, t.mode, lmbda=t.lmbda, mark=mark)


def mse_hist_resampling(bootstrap, tag='', show=False, mark=None):
    assert isinstance(tag, str)
    fig, ax = plt.subplots()
    make_histogram(ax, bootstrap, 'mse')
    scheme = bootstrap.scheme.lower()
    pdfname = f'MSE_hist_{scheme}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, scheme, bootstrap.mode, polydeg=bootstrap.polydeg, resampling_type='BS', resampling_iter=bootstrap.B, lmbda=bootstrap.lmbda, mark=mark)


def beta_hist_resampling(bootstrap, grouped=False, tag='', show=False, mark=None):
    assert isinstance(tag, str)
    fig, ax = plt.subplots()
    scheme = bootstrap.scheme.lower()
    if grouped:
        make_histogram(ax, bootstrap, 'beta_grouped')
        pdfname = f'beta_polydeg_hist_{scheme}{tag}.pdf'
    else:
        make_histogram(ax, bootstrap, 'beta_mean')
        pdfname = f'beta_hist_{scheme}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, scheme, bootstrap.mode, polydeg=bootstrap.polydeg, resampling_type='BS', resampling_iter=bootstrap.B, lmbda=bootstrap.lmbda, mark=mark)


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
    
    ax2.plot(polydegs, MSE[0], lw=2.5, ls='--', c='deepskyblue', label='train MSE')
    ax2.plot(polydegs, MSE[1], lw=2.5, ls='-',  c='forestgreen', label='test MSE')
    ax1.plot(polydegs, R2[0],  lw=2.5, ls='--', c='teal',        label='train $R^2$')
    ax1.plot(polydegs, R2[1],  lw=2.5, ls='-',  c='darkgreen',   label='test $R^2$')


    set_axes_2d(ax1, ylabel='score')
    pad = 1/5
    set_axes_2d(ax2, xlabel='Complexity', ylabel='score', xlim=(polydegs[0]-pad, polydegs[-1]+pad))
    scheme = trainings[d].scheme.lower()
    pdfname = f'MSE_R2_scores_{scheme}{tag}.pdf'
    save_push(fig, pdfname, show=show)
    set_info(pdfname, scheme, trainings[d].mode, lmbda=trainings[d].lmbda, mark=mark)


def error_vs_noise(trainings, predictions, eta_vals, tag='', show=False, mark=None):
    fig, ax = plt.subplots()

    M = len(eta_vals)
    N = len(trainings)
    polydegs = np.zeros(N)
    trainMSE = np.zeros((N,M))
    testMSE = np.zeros((N,M))

    
    i = 0
    for d in trainings.keys():
        polydegs[i] = d
        for j in range(M):
            train, test = trainings[d][j], predictions[d][j]
            trainMSE[i, j] = train.MSE
            testMSE[i, j] = test.MSE
        i+=1
    
    for j in range(M):
        ax.plot(polydegs, trainMSE[:,j], 'o--', lw=2)
        ax.plot(polydegs, testMSE[:,j],  'o-',  lw=2, c=plt.gca().lines[-1].get_color(), label=r'$\eta=%6.4f$'%eta_vals[j])

    ax.plot(-2,0, 'o--', lw=1.5, c='k', alpha=0.5, label='train MSE')
    ax.plot(-2,0, 'o-',  lw=1.5, c='k', alpha=0.5, label='test MSE')

    pad = 1/4
    ax.set_yscale('log')
    set_axes_2d(ax, xlabel='Complexity', ylabel='score', xlim=(polydegs[0]-pad, polydegs[-1]+pad))
    scheme = trainings[d][0].scheme.lower()
    pdfname = f'error_vs_noise_{scheme}{tag}.pdf'
    save_push(fig, pdfname, show=show)

    set_info(pdfname, scheme, trainings[d][0].mode, lmbda=trainings[d][0].lmbda, mark=mark)


def heatmap(resampling_grid, ref_error=0, tag='', show=False, mark=None):
    fig, ax = plt.subplots(layout="constrained")
    N_polydegs = len(resampling_grid)
    N_params = len(resampling_grid[0])

    pred_error = np.zeros((N_params, N_polydegs))
    polydegs = np.zeros_like(pred_error)
    hparams = np.zeros_like(pred_error)


    for i in range(N_polydegs):
        for j in range(N_params):
            rs = resampling_grid[i][j]
            pred_error[j, i] = rs.resamplingError()
            polydegs[j, i] = rs.polydeg
            hparams[j, i] = rs.lmbda


    # cmaps: Accent, magma, plasma,c ividis
    L = 30
    levels0 = np.linspace(pred_error.min(), pred_error.max(), L)
    CS = ax.contourf(polydegs, hparams, pred_error, cmap=cm.magma, levels=levels0)
    #
    j0, i0 = np.unravel_index(np.argmin(pred_error), np.shape(pred_error))
    d0, l0, err0 = polydegs[j0,i0], hparams[j0,i0], pred_error[j0,i0]
    # contours
    if ref_error>err0:
        #fmt1 = {levels[0]:'', levels[1]:'MSE'}
        #ax.clabel(cntr, fmt=fmt1, colors='springgreen', fontsize=SMALL_SIZE)
        ii = np.argmin(np.abs(levels0-ref_error))
        cntr = ax.contour(CS, levels=[levels0[0], levels0[ii]], colors='springgreen')
        

    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(r'prediction error')
    ax.axhline(l0, ls='--', lw=0.8, color='r', alpha=0.7)
    ax.axvline(d0, ls='--', lw=0.8, color='r', alpha=0.7)
    ax.text(polydegs[j0,int(N_polydegs/5)], l0, r'$\lambda = %.2e$'%l0,   fontsize=SMALL_SIZE, color='r', bbox=boxStyle)
    ax.text(d0, hparams[int(3*N_params/5),i0], r'$d = %i$'%d0,fontsize=SMALL_SIZE, color='r', rotation=-90, bbox=boxStyle)

    set_axes_2d(ax, xlabel='Complexity', ylabel='HyperParameter', legend=False, ylim=(np.min(hparams), np.max(hparams)), xlim=(np.min(polydegs)-0.02, np.max(polydegs)+0.02))
    

    scheme = rs.scheme.lower()
    type = rs.ID()
    Niter = rs.Niter
    pdfname = f'MSE_heatmap_{scheme}_{type}{tag}.pdf'
    save_push(fig, pdfname, show=show, tight=False)

    set_info(pdfname, scheme, rs.mode, resampling_type=type, resampling_iter=Niter, mark=mark)

        














def remove_pdf(info, pdfname):
    infoT = info.transpose()
    infoT.pop(pdfname)
    info = infoT.transpose()
    return info



def update_info(additional_information='none'):
    if testMode:
        pass
    else:
        INFOd = OrderedDict(sorted(INFO.items(), key=lambda i:i[0].lower()))
        infopd = pd.DataFrame.from_dict(INFOd, orient='index')
        infopd.to_pickle(plot_path + 'info.pkl')
        infopd.to_markdown(plot_path + 'README.md')

        with open(plot_path + 'README.md', 'a') as file:
            file.write('\n\n\n')
            file.write(f'# Information about plots in `{show_path}`')
            file.write('\n\n')

            if additional_information != 'none':
                file.write('\n## Additional information:\n\n')
                for line in additional_information:
                    file.write(f'* {line}\n')

        print(f'\nSuccessfully written information to \n    {show_path}README.md.\n')