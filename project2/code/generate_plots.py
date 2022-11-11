from src.utils import *
import plot as PLOT







# SIMPLE REGRESSION
# TODO: put necessary info in utils? 
n_etas = 10
learningRates = np.logspace(-5, -1, n_etas)     #   the η's we consider

n_obs = 1000
x = np.linspace(-1,1, n_obs)
X = np.zeros((len(x),3))
X[:,0] = x
X[:,1] = x**2
X[:,2] = x**3
noise_scale = 0.1
theta_actual = np.array([2,1.7,-0.4])
y = X@theta_actual+ np.random.randn(n_obs)*noise_scale

X, y, X_train, y_train, X_test, y_test = Z_score_normalise_split(X, y)


noEpochs1, noEpochs2 = 500, 1000
noMinibatches = 50
theta0 = np.array([1,0.5,4])

files = ["momentum_SGD.txt", "plain_SGD.txt", "adagrad_SGD.txt", "rmsprop_SGD.txt", "adam_SGD.txt"]
labels =  ["momentum SGD", "plain SGD", "AdaGrad", "RMSProp", "Adam"]

# PLOT.simple_regression_errors(files, labels, pdfname="errors_gradient_descent", savepush=True, show=True)
# PLOT.set_pdf_info("errors_gradient_descent", method='SGD', eta='...', theta0=theta0) # more here?

# PLOT.simple_regression_polynomial(X, y, files, labels, pdfname="polynomial_gradient_descent", savepush=True, show=True)
# PLOT.set_pdf_info("polynomial_gradient_descent", method='SGD', eta='...', theta0=theta0) # more here?

# PLOT.simple_regression_errors(["ridge_"+file for file in files], labels, pdfname="ridge_errors_gradient_descent",savepush=True, show=True)
# 


# PLOT.heatmap_plot("EtaLmbdaMSE.pkl")


PLOT.MSEheatmap_plot("EtaLmbdaMSE.pkl")

# PLOT.epoch_plot("actFuncPerEpoch.pkl", pdfname="actFuncPerEpoch")

# PLOT.heatmap_plot("LayerNeuron.pkl", pdfname="LayerNeuron", savepush=False, xlabel="Neurons", ylabel="Hidden layers")
