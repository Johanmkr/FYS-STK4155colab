|                        | scheme | mode   | $d$ | $\lambda$             | resampling (iter) | mark                                              |
| :----------------------- | :------- | :------- | :---- | :---------------------- | :------------------ | :-------------------------------------------------- |
| beta_hist_ols.pdf      | OLS    | manual | 5   | 0                     | BS (400)          |                                                   |
| beta_hist_ridge.pdf    | Ridge  | manual | 5   | 0.0008858667904100823 | BS (400)          |                                                   |
| beta_ols.pdf           | OLS    | manual |     | 0                     |                   |                                                   |
| data3D.pdf             |        |        |     |                       |                   | $η=0.1$                                          |
| data3D_no_noise.pdf    |        |        |     |                       |                   | $η=0$                                            |
| error_vs_noise_ols.pdf | ols    | manual |     | 0                     |                   | η = 1.0e-04, 1.0e-03, 1.0e-02, 1.0e-01, 1.0e+00, |
| MSE_hist_ols.pdf       | OLS    | manual | 5   | 0                     | BS (400)          |                                                   |
| MSE_hist_ridge.pdf     | Ridge  | manual | 5   | 0.0008858667904100823 | BS (400)          |                                                   |
| MSE_lasso_BS.pdf       | lasso  | skl    | 5   | ...                   | BS (10)           |                                                   |
| MSE_lasso_CV.pdf       | lasso  | skl    | 5   | ...                   | CV (8)            |                                                   |
| MSE_ols_BS.pdf         | OLS    | manual | ... | 0                     | BS (400)          |                                                   |
| MSE_ols_CV.pdf         | OLS    | manual | ... | 0                     | CV (8)            |                                                   |
| MSE_R2_scores_ols.pdf  | OLS    | manual |     | 0                     |                   |                                                   |
| MSE_ridge_BS.pdf       | Ridge  | manual | 5   | ...                   | BS (400)          |                                                   |
| MSE_ridge_CV.pdf       | Ridge  | manual | 5   | ...                   | CV (8)            |                                                   |
| tradeoff_lasso.pdf     | lasso  | skl    |     |                       | BS (10)           |                                                   |
| tradeoff_ols.pdf       | OLS    | manual |     |                       | BS (400)          |                                                   |
| tradeoff_ridge.pdf     | Ridge  | manual |     |                       | BS (400)          |                                                   |

# Information about plots in `/output/figures/Franke/`

## Additional information:

* xy-grid: N x N = 20 x 20
* noise level: η = 0.1
* Considered 12 polynomial degrees between d = 1 and d = 12 (linarly spaced).
* Considered 20 λ-values between λ = 1.0e-06 and λ = 1.0e-02 (logarithmically spaced).