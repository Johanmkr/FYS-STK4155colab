from src.utils import *
import plot as PLOT



SHOW = True

'''
--------------------------
SIMPLE REGRESSION ANALYSIS
--------------------------
'''


files0 = ["plain_GD.txt", "momentum_GD.txt", "adagrad_GD.txt", "rmsprop_GD.txt", "adam_GD.txt"]
labels0 =  ["plain GD", "momentum GD", "AdaGrad", "RMSProp", "Adam"]

files = ["plain_SGD.txt", "momentum_SGD.txt", "adagrad_SGD.txt", "rmsprop_SGD.txt", "adam_SGD.txt"]
labels =  ["plain SGD", "momentum SGD", "AdaGrad", "RMSProp", "Adam"]




''' OLS '''
# PLOT.simple_regression_errors(files0, labels0, pdfname="errors_gradient_descent_nonstochastic", savepush=True, show=SHOW)

#PLOT.simple_regression_errors(files, labels, pdfname="errors_gradient_descent", savepush=True, show=SHOW)

# PLOT.simple_regression_polynomial(files, labels, pdfname="polynomial_gradient_descent", savepush=True, show=SHOW)

''' Ridge '''
# PLOT.simple_regression_errors(["ridge_"+file for file in files], labels, pdfname="ridge_errors_gradient_descent",savepush=True, show=SHOW)


'''
-----------------------------
FRANKE NN REGRESSION ANALYSIS
-----------------------------
'''



# PLOT.MSEheatmap_plot("EtaLmbdaMSE.pkl", pdfname="eta_lambda_analysis", savepush=True, show=SHOW)

# PLOT.MSEheatmap_plot("LayerNeuron.pkl", pdfname="layer_neuron_analysis", savepush=True, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)

# PLOT.epoch_plot("actFuncPerEpoch1000.pkl", pdfname="actFuncPer1000Epoch", savepush=True, show=SHOW)

# PLOT.epoch_plot("actFuncPerEpoch250.pkl", pdfname="actFuncPerEpoch", savepush=True, show=SHOW)


# PLOT.MSEheatmap_plot("EpochMinibatch.pkl", pdfname="epoch_minibatch_analysis", savepush=True, xlabel=r"$m$", ylabel=r"no. of epochs", show=SHOW)


PLOT.FrankePlot('none', pdfname='franke', savepush=True)

'''
---------------------------------
CANCER NN CLASSIFICATION ANALYSIS
---------------------------------
'''
# PLOT.CancerHeatmap_plot("EtaLmbdaMSECancer.pkl", pdfname="eta_lambda_analysisCancer", savepush=True, show=SHOW)

# PLOT.CancerHeatmap_plot("LayerNeuronCancer.pkl", pdfname="layer_neuron_analysisCancer", savepush=True, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)

# PLOT.Cancerepoch_plot("actFuncPerEpochCancer1000.pkl", pdfname="actFuncPerEpoch1000Cancer", savepush=True, show=SHOW)

# PLOT.Cancerepoch_plot("actFuncPerEpochCancer250.pkl", pdfname="actFuncPerEpochCancer", savepush=True, show=SHOW)

# PLOT.CancerHeatmap_plot("EpochMinibatchCancer.pkl", pdfname="epoch_minibatch_analysisCancer", savepush=True, xlabel=r"$m$", ylabel=r"no. of epochs", show=SHOW)

'''
-----------------------------------
CANCER LOGISTIC REGRESSION ANALYSIS
-----------------------------------
'''

# PLOT.CancerHeatmap_plot("logistic.pkl", pdfname="logistic", savepush=True, show=SHOW)




# Update README.md:
PLOT.update()
