
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
    "lmbda":r"$\lambda$",
    "gamma":r"$\gamma$", 
    "rho":r"$\varrho_1$, $\varrho_2$",
    "theta0":r"$\theta_0$",
    "timer":"run time (s)"})



from src.GradientDescent import GD, SGD



noiseScale = 0.1
no_of_etas = 11
learningRates = np.logspace(-5, 0, no_of_etas)     #   the η's we consider
no_of_observations = 400
thetaActual = [2,1.7,-0.4]

def create_mock_data(seed=169, noise_scale=noiseScale, theta_actual=thetaActual, n_obs=no_of_observations):
    np.random.seed(seed)

    x = np.linspace(-1,1, n_obs)
    X = np.zeros((len(x),3))
    X[:,0] = x
    X[:,1] = x**2
    X[:,2] = x**3
    theta_actual = np.asarray(theta_actual)
    y = X@theta_actual + np.random.randn(n_obs)*noise_scale

    X, y, X_train, y_train, X_test, y_test = Z_score_normalise_split(X, y)

    np.savetxt(output_path + 'design_matrix.txt', X, delimiter=", ")
    np.savetxt(output_path + 'target_data.txt', y, delimiter=", ")

    return X_train, y_train, X_test, y_test



X_train, y_train, X_test, y_test = create_mock_data()

noEpochs1, noEpochs2 = 25, 50
no_of_minibatches = no_of_observations//10
np.random.seed(269)
theta0 = np.random.randn(3)



def simple_analysis(optimiser, filename, params={}, params_info=None, lmbda=0):
    test_MSE1 = np.zeros_like(learningRates); test_MSE2 = np.zeros_like(learningRates)
    theta2 = []
    t0 = time()
    last = []
    for k, eta in enumerate(learningRates):
        print('\n' + '_'*50)
        print(f'    eta = {eta:.2e}   ({k+1}/{no_of_etas})\n')
        sgd = SGD(X_train, y_train, eta, theta0, noEpochs1, no_of_minibatches)
        sgd.set_update_rule(optimiser, params)
        sgd.regression_setting(lmbda)
        sgd(noEpochs1, 6000) 
        test_MSE1[k] = sgd.mean_squared_error(X_test, y_test)
        if sgd.current_epoch == noEpochs1-1:
            sgd(noEpochs2-noEpochs1, 6000) 
        test_MSE2[k] = sgd.mean_squared_error(X_test, y_test)
        last.append(sgd.current_epoch)
        theta2.append(sgd.theta)
    t1 = time()
    theta2 = theta2[np.argmin(test_MSE2)] # best estimate of theta
    
    np.savetxt(output_path+filename, (learningRates, test_MSE1, test_MSE2), delimiter=", ", header=f"optimal theta = {theta2}", footer=f"\nrun time: {t1-t0:.0f} s")
    
    Params_ = params_info or params
    Params = Params_.copy()
    Params['lmbda'] = lmbda
    info.set_file_info(filename, method="SGD", opt=optimiser, **Params, eta="...", no_epochs=(noEpochs1, noEpochs2), no_minibatches=no_of_minibatches, n_obs=no_of_observations, theta0=theta0, timer=f"{t1-t0:.0f}")




# OLS: 

# simple_analysis("momentum", "momentum_SGD.txt", {"gamma":0.5})
# simple_analysis("plain", "plain_SGD.txt")
# simple_analysis("adagrad", "adagrad_SGD.txt")
# simple_analysis("RMSprop", "rmsprop_SGD.txt", {"rho":0.9})
# simple_analysis("Adam", "adam_SGD.txt", {"rho1":0.9, 'rho2':0.999}, {'rho':(0.9, 0.999)})


# Ridge:
lmbda0 = 0.1
# simple_analysis("momentum", "ridge_momentum_SGD.txt", {"gamma":0.5}, lmbda=lmbda0)
# simple_analysis("plain", "ridge_plain_SGD.txt", lmbda=lmbda0)
# simple_analysis("adagrad", "ridge_adagrad_SGD.txt", lmbda=lmbda0)
# simple_analysis("RMSprop", "ridge_rmsprop_SGD.txt", {"rho":0.9}, lmbda=lmbda0)
# simple_analysis("Adam", "ridge_adam_SGD.txt", {"rho1":0.9, 'rho2':0.999}, {'rho':(0.9, 0.999)}, lmbda=lmbda0)



info.additional_information(r"$f(x) = %.2f x + %.2f x^2 + %.2f x^3 \, + \, %.2f \cdot N(0, 1)$"  %(thetaActual[0], thetaActual[1], thetaActual[2], noiseScale))

info.additional_information(r"Considered %i logarithmically spaced learning rates $\eta \in [%.1e, \, %.1e]$." %(no_of_etas, learningRates[0], learningRates[-1]))

info.update(header="Results from simple regression analysis using (S)GD")
