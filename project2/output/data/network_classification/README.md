|                           | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs   |   $m$ | $\eta$                  | $\lambda$               | $L-1$        | $N_l$         | $g$     |   train time (s) |   $\gamma$ | $\varrho_1$, $\varrho_2$   |   $\theta_0$ | note      |
|:--------------------------|:---------|:------------|-------------------:|:----------|------:|:------------------------|:------------------------|:-------------|:--------------|:--------|-----------------:|-----------:|:---------------------------|-------------:|:----------|
| actFuncPerEpochCancer.pkl | SGD      | RMSProp     |                569 | ...       |     5 | 0.1                     | 1e-05                   | 1            | 5             | ...     |              nan |        nan | (0.9, 0.999)               |          nan |           |
| dummy.jpg                 | nan      | nan         |                nan | nan       |   nan | nan                     | nan                     | nan          | nan           | nan     |              nan |        nan | nan                        |          nan | delete me |
| EtaLmbdaMSECancer.pkl     | SGD      | RMSProp     |                569 | 250       |     5 | $[$10^{-5}$, $10^{1}$]$ | $[$10^{-5}$, $10^{1}$]$ | 1            | 5             | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |           |
| LayerNeuronCancer.pkl     | SGD      | RMSProp     |                569 | 250       |     5 | 0.1                     | 1e-05                   | $[{0}, {9}]$ | $[{5}, {50}]$ | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |           |


# Results from cancer classification analysis using NN


## Additional information:

* Loss function: cross entropy (with regularisation
