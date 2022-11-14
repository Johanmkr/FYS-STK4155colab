|                         | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs           | $m$           | $\eta$                  | $\lambda$               | $L-1$        | $N_l$         | $g$     |   train time (s) |   $\gamma$ | $\varrho_1$, $\varrho_2$   |   $\theta_0$ | note   |
|:------------------------|:---------|:------------|-------------------:|:------------------|:--------------|:------------------------|:------------------------|:-------------|:--------------|:--------|-----------------:|-----------:|:---------------------------|-------------:|:-------|
| actFuncPerEpoch.pkl     | SGD      | RMSProp     |                400 | ...               | 5             | 0.01                    | 1e-05                   | 3            | 40            | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| actFuncPerEpoch1000.pkl | SGD      | RMSProp     |                400 | ...               | 5             | 0.01                    | 1e-05                   | 3            | 40            | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| actFuncPerEpoch250.pkl  | SGD      | RMSProp     |                400 | ...               | 5             | 0.01                    | 1e-05                   | 3            | 40            | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| EpochMinibatch.pkl      | SGD      | RMSProp     |                400 | $[{100}, {1000}]$ | $[{1}, {10}]$ | 0.01                    | 1e-05                   | 3            | 40            | tanh    |              nan |        nan | (0.9, 0.999)               |          nan |        |
| EtaLmbdaMSE.pkl         | SGD      | RMSProp     |                400 | 250               | 5             | $[$10^{-8}$, $10^{1}$]$ | $[$10^{-8}$, $10^{1}$]$ | 1            | 5             | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |
| LayerNeuron.pkl         | SGD      | RMSProp     |                400 | 250               | 5             | 0.01                    | 1e-05                   | $[{0}, {9}]$ | $[{5}, {50}]$ | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |


# Results from Franke regression analysis using NN


## Additional information:

* Loss function: mean squared error (with regularisation)
