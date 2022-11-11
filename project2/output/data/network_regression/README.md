|                     | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs           | $m$           | $\eta$                  | $\lambda$               | $L-1$        | $N_l$         | $g$     |   train time (s) |   $\gamma$ |   $\varrho_1$, $\varrho_2$ |   $\theta_0$ | note   |
|:--------------------|:---------|:------------|-------------------:|:------------------|:--------------|:------------------------|:------------------------|:-------------|:--------------|:--------|-----------------:|-----------:|---------------------------:|-------------:|:-------|
| actFuncPerEpoch.pkl | SGD      | RMSProp     |                nan | ...               | 5             | 0.01                    | 1e-05                   | 1            | 5             | ...     |              nan |        nan |                        nan |          nan |        |
| EpochMinibatch.pkl  | SGD      | RMSProp     |                nan | $[{100}, {1000}]$ | $[{5}, {50}]$ | 0.01                    | 1e-05                   | 1            | 5             | tanh    |              nan |        nan |                        nan |          nan |        |
| EtaLmbdaMSE.pkl     | SGD      | RMSProp     |                nan | 250               | 5             | $[$10^{-9}$, $10^{0}$]$ | $[$10^{-9}$, $10^{0}$]$ | 1            | 5             | sigmoid |              nan |        nan |                        nan |          nan |        |
| LayerNeuron.pkl     | SGD      | RMSProp     |                nan | 250               | 5             | 0.01                    | 0.001                   | $[{0}, {9}]$ | $[{5}, {50}]$ | sigmoid |              nan |        nan |                        nan |          nan |        |


# Results from Franke regression analysis using NN

