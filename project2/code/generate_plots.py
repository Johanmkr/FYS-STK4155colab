from src.utils import *
import plot as PLOT





# np.random.seed(169)

# SIMPLE REGRESSION


no_of_observations = 400
no_epochs1, no_epochs2 = 25, 50
no_of_minibatches = no_of_observations//10
np.random.seed(269)
theta0 = np.random.randn(3)


files = ["plain_SGD.txt", "momentum_SGD.txt", "adagrad_SGD.txt", "rmsprop_SGD.txt", "adam_SGD.txt"]
labels =  ["plain SGD", "momentum SGD", "AdaGrad", "RMSProp", "Adam"]

PLOT.simple_regression_errors(files, labels, pdfname="errors_gradient_descent", savepush=True, show=True)
PLOT.set_pdf_info("errors_gradient_descent", method='SGD', eta='...', theta0=theta0, epochs=(no_epochs1, no_epochs2), no_minibatches=no_of_minibatches) 

PLOT.simple_regression_polynomial(files, labels, pdfname="polynomial_gradient_descent", savepush=True, show=True)
# PLOT.set_pdf_info("polynomial_gradient_descent", method='SGD', eta='...', theta0=theta0) # more here?

PLOT.simple_regression_errors(["ridge_"+file for file in files], labels, pdfname="ridge_errors_gradient_descent",savepush=True, show=True)
# 


# PLOT.heatmap_plot("EtaLmbdaMSE.pkl")


PLOT.heatmap_plot("EtaLmbdaMSE.pkl")

# PLOT.epoch_plot("actFuncPerEpoch.pkl", pdfname="actFuncPerEpoch")

PLOT.heatmap_plot("LayerNeuron.pkl", pdfname="LayerNeuron", savepush=False, xlabel="Neurons", ylabel="Hidden layers")
