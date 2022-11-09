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

files = ["ridge_momentum_SGD.txt", "ridge_plain_SGD.txt", "ridge_adagrad_SGD.txt", "ridge_rmsprop_SGD.txt", "ridge_adam_SGD.txt"]
labels =  ["momentum SGD", 'plain SGD', "AdaGrad", "RMSProp", "Adam"]
PLOT.simple_regression_errors(files, labels, savepush=False, show=True)
# 


# PLOT.heatmap_plot("EtaLmbdaMSE.txt", savepush=False)

PLOT.update()