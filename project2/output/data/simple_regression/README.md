|                  | method   | optimiser   |   $n_\mathrm{obs}$ | #epochs     |   $m$ | $\eta$   |   $\gamma$ | $\varrho_1$, $\varrho_2$   | $\theta_0$    | note   |
|:-----------------|:---------|:------------|-------------------:|:------------|------:|:---------|-----------:|:---------------------------|:--------------|:-------|
| adagrad_SGD.txt  |          | adagrad     |               1000 | (500, 1000) |    50 | ...      |            |                            | [1.  0.5 4. ] |        |
| adam_SGD.txt     |          | Adam        |               1000 | (500, 1000) |    50 | ...      |            | (0.9, 0.999)               | [1.  0.5 4. ] |        |
| momentum_SGD.txt |          | momentum    |               1000 | (500, 1000) |    50 | ...      |        0.5 |                            | [1.  0.5 4. ] |        |
| plain_SGD.txt    |          | plain       |               1000 | (500, 1000) |    50 | ...      |            |                            | [1.  0.5 4. ] |        |
| rmsprop_SGD.txt  |          | RMSprop     |               1000 | (500, 1000) |    50 | ...      |            | 0.9                        | [1.  0.5 4. ] |        |


# (S)GD with different update rules and hyperparameters


## Additional information:

* $f(x) = 2.00 x + 1.70 x^2 + -0.40 x^3 \, + \, 0.10 \cdot N(0, 1)$
* Considered 10 logarithmically spaced learning rates $\eta \in [1.0e-05, \, 1.0e-01]$.
