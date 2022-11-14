|                               | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs           | $m$          | $\eta$                  | $\lambda$               | $L-1$        | $N_l$         | $g$     |   train time (s) |   $\gamma$ | $\varrho_1$, $\varrho_2$   |   $\theta_0$ | note   |
|:------------------------------|:---------|:------------|-------------------:|:------------------|:-------------|:------------------------|:------------------------|:-------------|:--------------|:--------|-----------------:|-----------:|:---------------------------|-------------:|:-------|
| actFuncPerEpochCancer.pkl     | SGD      | RMSProp     |                569 | ...               | 5            | 0.1                     | 1e-05                   | 1            | 5             | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| actFuncPerEpochCancer1000.pkl | SGD      | RMSProp     |                569 | ...               | 5            | 0.001                   | 0.0001                  | 2            | 30            | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| actFuncPerEpochCancer250.pkl  | SGD      | RMSProp     |                569 | ...               | 5            | 0.001                   | 0.0001                  | 2            | 30            | ...     |              nan |        nan | (0.9, 0.999)               |          nan |        |
| EpochMinibatchCancer.pkl      | SGD      | RMSProp     |                569 | $[{100}, {1000}]$ | $[{0}, {9}]$ | 0.001                   | 0.0001                  | 2            | 30            | tanh    |              nan |        nan | (0.9, 0.999)               |          nan |        |
| EtaLmbdaMSECancer.pkl         | SGD      | RMSProp     |                569 | 250               | 5            | $[$10^{-8}$, $10^{1}$]$ | $[$10^{-8}$, $10^{1}$]$ | 1            | 5             | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |
| LayerNeuronCancer.pkl         | SGD      | RMSProp     |                569 | 250               | 5            | 0.001                   | 0.0001                  | $[{0}, {9}]$ | $[{5}, {50}]$ | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |
| logistic.pkl                  | SGD      | adaGrad     |                569 | 250               | 5            | $[$10^{-8}$, $10^{1}$]$ | $[$10^{-8}$, $10^{1}$]$ | 0            |               | sigmoid |              nan |        nan | (0.9, 0.999)               |          nan |        |


# Results from cancer classification analysis using NN


## Additional information:

* Loss function: cross entropy (with regularisation)
