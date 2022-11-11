from src.utils import *
import plot as PLOT


# SIMPLE REGRESSION


no_of_observations = 400
no_epochs1, no_epochs2 = 25, 50
no_of_minibatches = no_of_observations//10
np.random.seed(269)
theta0 = np.random.randn(3)
lmbda_ridge = 0.1

SHOW = False

'''
--------------------------
SIMPLE REGRESSION ANALYSIS
--------------------------
'''

files = ["plain_SGD.txt", "momentum_SGD.txt", "adagrad_SGD.txt", "rmsprop_SGD.txt", "adam_SGD.txt"]
labels =  ["plain SGD", "momentum SGD", "AdaGrad", "RMSProp", "Adam"]

''' OLS '''
# PLOT.simple_regression_errors(files, labels, pdfname="errors_gradient_descent", savepush=True, show=SHOW)
PLOT.set_pdf_info("errors_gradient_descent", method='SGD', opt='...', eta='...', theta0=theta0, no_epochs=(no_epochs1, no_epochs2), no_minibatches=no_of_minibatches, n_obs=no_of_observations, lmbda=0) 

# PLOT.simple_regression_polynomial(files, labels, pdfname="polynomial_gradient_descent", savepush=True, show=SHOW)
# PLOT.set_pdf_info("polynomial_gradient_descent",  method='SGD', opt='...', eta='...', theta0=theta0, no_epochs=(no_epochs1, no_epochs2), no_minibatches=no_of_minibatches, n_obs=no_of_observations, lmbda=0) 

''' Ridge '''
# PLOT.simple_regression_errors(["ridge_"+file for file in files], labels, pdfname="ridge_errors_gradient_descent",savepush=True, show=SHOW)
# PLOT.set_pdf_info("ridge_errors_gradient_descent",  method='SGD', opt='...', eta='...', theta0=theta0, no_epochs=(no_epochs1, no_epochs2), no_minibatches=no_of_minibatches, n_obs=no_of_observations, lmbda=lmbda_ridge) 


'''
-----------------------------
FRANKE NN REGRESSION ANALYSIS
-----------------------------
'''


no_of_observations = int(20*20)
no_epochs = 250 #???
no_of_minibatches = no_of_observations//10
optimiser = 'RMSProp'
no_hidden_layers = 1
no_neurons = 5

# PLOT.MSEheatmap_plot("EtaLmbdaMSE.pkl", pdfname="eta_lambda_analysis", savepush=True, show=SHOW)
# PLOT.set_pdf_info("eta_lambda_analysis", method='SGD', opt=optimiser, eta='', no_epochs=no_epochs, no_minibatches=no_of_minibatches, n_obs=no_of_observations, lmbda='...', L=no_hidden_layers, N=no_neurons) 

# PLOT.epoch_plot("actFuncPerEpoch.pkl", pdfname="actFuncPerEpoch")

# PLOT.MSEheatmap_plot("LayerNeuron.pkl", pdfname="layer_neuron_analysis", savepush=True, xlabel="$N_l$", ylabel=r"$L-1$", show=SHOW)
# #xlabel="#neurons", ylabel=r"#hidden layers", show=SHOW)
# PLOT.set_pdf_info("layer_neuron_analysis", method='SGD', opt=optimiser, eta='?', no_epochs=no_epochs, no_minibatches=no_of_minibatches, n_obs=no_of_observations, lmbda='?', L=r'$0,1,\dots,9$', N=r'$5, 10, \dots, 50$') 

PLOT.MSEheatmap_plot("EpochMinibatch.pkl", pdfname="epoch_minibatch_analysis", savepush=True, xlabel=r"m", ylabel=r"Epochs", show=SHOW)


# Update README.md:
PLOT.update()
