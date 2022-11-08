
from src.utils import *
import src.infoFile_ala_Nanna as info

output_path = "../output/data/simple_regression/"
info.init(path=output_path)
info.sayhello()

info.define_categories({
    "method":"method", 
    "opt":"optimiser", 
    "n_obs":r"$n_\mathrm{obs}$", 
    "no_epochs":"#epochs", 
    "no_minibatches":r"$m$",
    "eta":r"$\eta$", 
    "gamma":r"$\gamma$", 
    "rho":r"$\varrho_1$, $\varrho_2$",
    "theta0":r"$\theta_0$",
    "timer":"run time (s)"})



from src.GradientDescent import GD, SGD


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



def simple_analysis(optimiser, filename, params={}, params_info=None):
    test_MSE1 = np.zeros_like(learningRates); test_MSE2 = np.zeros_like(learningRates)
    theta2 = []
    t0 = time()
    for k, eta in enumerate(learningRates):
        print('\n' + '_'*50)
        print(f'    eta = {eta:.2e}   ({k+1}/{n_etas})\n')
        sgd = SGD(X_train, y_train, eta, theta0, noEpochs1, noMinibatches)
        sgd.set_update_rule(optimiser, params)
        sgd.regression_setting()
        sgd(noEpochs1, 6000) 
        test_MSE1[k] = sgd.mean_squared_error(X_test, y_test)
        sgd(noEpochs2-noEpochs1, 6000) 
        test_MSE2[k] = sgd.mean_squared_error(X_test, y_test)
        theta2.append(sgd.theta)
    t1 = time()
    theta2 = theta2[np.argmin(test_MSE2)] # best estimate of theta
    
    np.savetxt(output_path+filename, (learningRates, test_MSE1, test_MSE2), delimiter=", ", header=f"optimal theta = {theta2}", footer=f"run time: {t1-t0:.0f} s")
    
    Params = params_info or params
    info.set_file_info(filename, method="SGD", opt=optimiser, **Params, eta="...", no_epochs=(noEpochs1, noEpochs2), no_minibatches=noMinibatches, n_obs=n_obs, theta0=theta0, timer=f"{t1-t0:.0f}")



# simple_analysis("momentum", "momentum_SGD.txt", {"gamma":0.5})
# simple_analysis("plain", "plain_SGD.txt")
# simple_analysis("adagrad", "adagrad_SGD.txt")
# simple_analysis("RMSprop", "rmsprop_SGD.txt", {"rho":0.9})
# simple_analysis("Adam", "adam_SGD.txt", {"rho1":0.9, 'rho2':0.999}, {'rho':(0.9, 0.999)})



info.additional_information(r"$f(x) = %.2f x + %.2f x^2 + %.2f x^3 \, + \, %.2f \cdot N(0, 1)$"  %(theta_actual[0], theta_actual[1], theta_actual[2], noise_scale))

info.additional_information(r"Considered %i logarithmically spaced learning rates $\eta \in [%.1e, \, %.1e]$." %(n_etas, learningRates[0], learningRates[-1]))

info.update(header="(S)GD with different update rules and hyperparameters")
