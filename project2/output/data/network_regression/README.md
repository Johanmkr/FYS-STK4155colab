|                         | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs           | $m$           | $\eta$                  | $\lambda$               | $L-1$        | $N_l$         | $g$     |   train time (s) |   $\gamma$ | $\varrho_1$, $\varrho_2$   |   $\theta_0$ | note   |
|:------------------------|:---------|:------------|-------------------:|:------------------|:--------------|:------------------------|:------------------------|:-------------|:--------------|:--------|-----------------:|-----------:|:---------------------------|-------------:|:-------|
| actFuncPerEpoch.pkl     | SGD      | RMSProp     |                400 | ...               | 5             | 0.01                    | 1e-05                   | 3            | 40            | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| actFuncPerEpoch1000.pkl | SGD      | RMSProp     |                400 | ...               | 3             | 0.1                     | 0.0001                  | 1            | 30            | ...     |              nan |        nan | 0.9                        |          nan |        |
| actFuncPerEpoch250.pkl  | SGD      | RMSProp     |                400 | ...               | 3             | 0.1                     | 0.0001                  | 1            | 30            | ...     |              nan |        nan | 0.9                        |          nan |        |
| compareModel.pkl        | SGD      | RMSProp     |                400 | 700               | 2             | 0.1                     | 0.0001                  | 1            | 30            | sigmoid |              nan |        nan | 0.9                        |          nan |        |
| EpochMinibatch.pkl      | SGD      | RMSProp     |                400 | $[{100}, {1000}]$ | $[{1}, {10}]$ | 0.1                     | 0.0001                  | 1            | 30            | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |
| EtaLmbdaMSE.pkl         | SGD      | RMSProp     |                400 | 250               | 3             | $[$10^{-9}$, $10^{0}$]$ | $[$10^{-9}$, $10^{0}$]$ | 3            | (15, 10, 5)   | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |
| LayerNeuron.pkl         | SGD      | RMSProp     |                400 | 250               | 3             | 0.1                     | 0.0001                  | $[{0}, {9}]$ | $[{5}, {50}]$ | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |


# Results from Franke regression analysis using NN


## Additional information:

* Loss function: mean squared error (with regularisation)
