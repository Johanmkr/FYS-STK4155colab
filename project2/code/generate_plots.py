from src.utils import *
import plot as PLOT


affirmative = ['yes', 'ye', 'y', 'yey', 'yeah', 'yay', 'yess']

SHOW = True
SAVEPUSH = False

'''
--------------------------
SIMPLE REGRESSION ANALYSIS
--------------------------
'''


files0 = ["plain_GD.txt", "momentum_GD.txt", "adagrad_GD.txt", "rmsprop_GD.txt", "adam_GD.txt"]
labels0 =  ["plain GD", "momentum GD", "AdaGrad", "RMSProp", "Adam"]

files = ["plain_SGD.txt", "momentum_SGD.txt", "adagrad_SGD.txt", "rmsprop_SGD.txt", "adam_SGD.txt"]
labels =  ["plain SGD", "momentum SGD", "AdaGrad", "RMSProp", "Adam"]

if input('Show plots for: SIMPLE REGRESSION ANALYSIS? [y/n][yes/no]').lower().strip() in affirmative:
    PLOT.simple_regression_errors(files0, labels0, pdfname="errors_gradient_descent_nonstochastic", savepush=SAVEPUSH, show=SHOW)

    PLOT.simple_regression_errors(files, labels, pdfname="errors_gradient_descent", savepush=SAVEPUSH, show=SHOW)

    PLOT.simple_regression_polynomial(files, labels, pdfname="polynomial_gradient_descent", savepush=SAVEPUSH, show=SHOW)

    PLOT.simple_regression_errors(["ridge_"+file for file in files], labels, pdfname="ridge_errors_gradient_descent",savepush=SAVEPUSH, show=SHOW)


'''
-----------------------------
FRANKE NN REGRESSION ANALYSIS
-----------------------------
'''
if input('Show plots for: NEURAL NETWORK REGRESSION ANALYSIS? [y/n][yes/no]').lower().strip() in affirmative:
    PLOT.MSEheatmap_plot("EtaLmbdaMSE.pkl", pdfname="eta_lambda_analysis", savepush=SAVEPUSH, show=SHOW)

    PLOT.MSEheatmap_plot("LayerNeuron.pkl", pdfname="layer_neuron_analysis", savepush=SAVEPUSH, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)

    PLOT.epoch_plot("actFuncPerEpoch1000.pkl", pdfname="actFuncPer1000Epoch", savepush=SAVEPUSH, show=SHOW)

    PLOT.epoch_plot("actFuncPerEpoch250.pkl", pdfname="actFuncPerEpoch", savepush=SAVEPUSH, show=SHOW)

    PLOT.MSEheatmap_plot("EpochMinibatch.pkl", pdfname="epoch_minibatch_analysis", savepush=SAVEPUSH, xlabel=r"$m$", ylabel=r"$\#$epochs", show=SHOW)

    PLOT.FrankePlot(pdfname='franke', savepush=SAVEPUSH, show=SHOW)


'''
---------------------------------
CANCER NN CLASSIFICATION ANALYSIS
---------------------------------
'''
if input('Show plots for: NEURAL NETWORK CLASSIFICATION ANALYSIS? [y/n][yes/no]').lower().strip() in affirmative:
    PLOT.CancerHeatmap_plot("EtaLmbdaMSECancer.pkl", pdfname="eta_lambda_analysisCancer", savepush=SAVEPUSH, show=SHOW)

    PLOT.CancerHeatmap_plot("LayerNeuronCancer.pkl", pdfname="layer_neuron_analysisCancer", savepush=SAVEPUSH, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)

    PLOT.Cancerepoch_plot("actFuncPerEpochCancer1000.pkl", pdfname="actFuncPerEpoch1000Cancer", savepush=SAVEPUSH, show=SHOW)

    PLOT.Cancerepoch_plot("actFuncPerEpochCancer250.pkl", pdfname="actFuncPerEpochCancer", savepush=SAVEPUSH, show=SHOW)

    PLOT.CancerHeatmap_plot("EpochMinibatchCancer.pkl", pdfname="epoch_minibatch_analysisCancer", savepush=SAVEPUSH, xlabel=r"$m$", ylabel=r"$\#$epochs", show=SHOW)


'''
-----------------------------------
CANCER LOGISTIC REGRESSION ANALYSIS
-----------------------------------
'''
if input('Show plots for: LOGISTIC REGRESSION ANALYSIS? [y/n][yes/no]').lower().strip() in affirmative:
    PLOT.CancerHeatmap_plot("logistic.pkl", pdfname="logistic", savepush=SAVEPUSH, show=SHOW)




if SAVEPUSH:
    '''
    Update info.pkl and README.md:
    '''
    PLOT.update()
