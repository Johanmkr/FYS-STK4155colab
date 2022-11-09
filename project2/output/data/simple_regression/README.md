|                        | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs     |   $m$ | $\eta$   |   $\gamma$ | $\varrho_1$, $\varrho_2$   | $\theta_0$    | note   |   $\lambda$ |   run time (s) |
|:-----------------------|:---------|:------------|-------------------:|:------------|------:|:---------|-----------:|:---------------------------|:--------------|:-------|------------:|---------------:|
| adagrad_SGD.txt        |          | adagrad     |               1000 | (500, 1000) |    50 | ...      |            |                            | [1.  0.5 4. ] |        |       nan   |            nan |
| adam_SGD.txt           |          | Adam        |               1000 | (500, 1000) |    50 | ...      |            | (0.9, 0.999)               | [1.  0.5 4. ] |        |       nan   |            nan |
| momentum_SGD.txt       |          | momentum    |               1000 | (500, 1000) |    50 | ...      |        0.5 |                            | [1.  0.5 4. ] |        |       nan   |            nan |
| plain_SGD.txt          |          | plain       |               1000 | (500, 1000) |    50 | ...      |            |                            | [1.  0.5 4. ] |        |       nan   |            nan |
| ridge_adagrad_SGD.txt  | SGD      | adagrad     |               1000 | (500, 1000) |    50 | ...      |            |                            | [1.  0.5 4. ] |        |         0.1 |            467 |
| ridge_adam_SGD.txt     | SGD      | Adam        |               1000 | (500, 1000) |    50 | ...      |            | (0.9, 0.999)               | [1.  0.5 4. ] |        |         0.1 |            338 |
| ridge_momentum_SGD.txt | SGD      | momentum    |               1000 | (500, 1000) |    50 | ...      |        0.5 |                            | [1.  0.5 4. ] |        |         0.1 |            467 |
| ridge_plain_SGD.txt    | SGD      | plain       |               1000 | (500, 1000) |    50 | ...      |            |                            | [1.  0.5 4. ] |        |         0.1 |            536 |
| ridge_rmsprop_SGD.txt  | SGD      | RMSprop     |               1000 | (500, 1000) |    50 | ...      |            | 0.9                        | [1.  0.5 4. ] |        |         0.1 |            335 |
| rmsprop_SGD.txt        |          | RMSprop     |               1000 | (500, 1000) |    50 | ...      |            | 0.9                        | [1.  0.5 4. ] |        |       nan   |            nan |


# (S)GD with different update rules and hyperparameters


## Additional information:

* $f(x) = 2.00 x + 1.70 x^2 + -0.40 x^3 \, + \, 0.10 \cdot N(0, 1)$
* Considered 10 logarithmically spaced learning rates $\eta \in [1.0e-05, \, 1.0e-01]$.
