from src.utils import *
import plot as PLOT



SHOW = True

# '''
# --------------------------
# SIMPLE REGRESSION ANALYSIS
# --------------------------
# '''


# files = ["plain_SGD.txt", "momentum_SGD.txt", "adagrad_SGD.txt", "rmsprop_SGD.txt", "adam_SGD.txt"]
# labels =  ["plain SGD", "momentum SGD", "AdaGrad", "RMSProp", "Adam"]

# ''' OLS '''
# PLOT.simple_regression_errors(files, labels, pdfname="errors_gradient_descent", savepush=True, show=SHOW)

# PLOT.simple_regression_polynomial(files, labels, pdfname="polynomial_gradient_descent", savepush=True, show=SHOW)

# ''' Ridge '''
# PLOT.simple_regression_errors(["ridge_"+file for file in files], labels, pdfname="ridge_errors_gradient_descent",savepush=True, show=SHOW)


# '''
# -----------------------------
# FRANKE NN REGRESSION ANALYSIS
# -----------------------------
# '''


# no_of_observations = int(20*20)
# no_epochs = 250 #???
# no_of_minibatches = no_of_observations//10
# optimiser = 'RMSProp'
# no_hidden_layers = 1
# no_neurons = 5

# PLOT.MSEheatmap_plot("EtaLmbdaMSE.pkl", pdfname="eta_lambda_analysis", savepush=True, show=SHOW)

# PLOT.epoch_plot("actFuncPerEpoch.pkl", pdfname="actFuncPerEpoch")

# PLOT.MSEheatmap_plot("LayerNeuron.pkl", pdfname="layer_neuron_analysis", savepush=True, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)
# # #xlabel="#neurons", ylabel=r"#hidden layers", show=SHOW)

# PLOT.MSEheatmap_plot("EpochMinibatch.pkl", pdfname="epoch_minibatch_analysis", savepush=True, xlabel=r"m", ylabel=r"Epochs", show=SHOW)








'''
Cancer data
'''
PLOT.CancerHeatmap_plot("EtaLmbdaMSECancer.pkl", pdfname="eta_lambda_analysisCancer", savepush=True, show=SHOW)

PLOT.Cancerepoch_plot("actFuncPerEpochCancer.pkl", pdfname="actFuncPerEpochCancer")

PLOT.CancerHeatmap_plot("LayerNeuronCancer.pkl", pdfname="layer_neuron_analysisCancer", savepush=True, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)
# #xlabel="#neurons", ylabel=r"#hidden layers", show=SHOW)

PLOT.CancerHeatmap_plot("EpochMinibatchCancer.pkl", pdfname="epoch_minibatch_analysisCancer", savepush=True, xlabel=r"m", ylabel=r"Epochs", show=SHOW)

# Update README.md:
PLOT.update()
