import src.infoFile_ala_Nanna as info



info.sayhello(path="/../output/data/simple_regression")

info.define_categories({
    "method":"method", 
    "opt":"optimiser", 
    "n_obs":r"$n_\mathrm{obs}$", 
    "no_epochs":"#epochs", 
    "no_minobatches":r"$m$",
    "eta":r"$\eta$", 
    "gamma":r"$\gamma$", 
    "rho":r"$\varrho_1$, $\varrho_2$"})




learning_rates = np.logspace(-5, -1, 10)


info.additional_information(r"%i log. spaced learning rates $\eta \in [%]$")

info.update(header="(S)GD with different update rules and hyperparameters")
