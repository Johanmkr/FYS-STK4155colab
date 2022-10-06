|                                   | scheme   | mode   | $d$   | $\lambda$   | resampling (iter)   | mark                               |
|:----------------------------------|:---------|:-------|:------|:------------|:--------------------|:-----------------------------------|
| beta_hist_ols.pdf                 | OLS      | own    | 6     | 0           | BS (200)            |                                    |
| beta_ols.pdf                      | OLS      | manual |       | 0           |                     | $β$'s grouped by order $d$         |
| beta_polydeg_hist_ols.pdf         | OLS      | own    | 8     | 0           | BS (200)            |                                    |
| comparison3D.pdf                  |          |        |       |             |                     | visualise x, y, z data             |
| comparison3D_ols.pdf              | OLS      | manual | 8     | 0           |                     | prediction set                     |
| comparison3D_ridge.pdf            | Ridge    | manual | 15    | 1.08e-08    |                     | prediction set                     |
| comparison3D_ridgetrain.pdf       | Ridge    | manual | 15    | 1.08e-08    |                     | training set                       |
| data3D.pdf                        |          |        |       |             |                     | visualise x, y, z data             |
| MSE_heatmap_lasso_CV.pdf          | Lasso    | skl    |       |             | CV (10)             | 10 $λ$'s from 1.00e-05 to 1.00e-05 |
| MSE_heatmap_ridge_CV.pdf          | Ridge    | skl    |       |             | CV (10)             | 12 $λ$'s from 1.00e-05 to 1.00e-05 |
| MSE_heatmap_ridge_CV_loworder.pdf | Ridge    | skl    |       |             | CV (9)              | 60 $λ$'s from 1.00e-06 to 1.00e-06 |
| MSE_hist_ols.pdf                  | ols      | own    | 8     | 0           | BS (100)            |                                    |
| MSE_hist_olsTrue.pdf              | OLS      | own    | 8     | 0           | BS (200)            |                                    |
| MSE_ols_BS.pdf                    | OLS      | own    | ...   | 0           | BS (10)             |                                    |
| MSE_ols_CV.pdf                    | nan      | nan    | nan   | nan         | nan                 |                                    |
| MSE_R2_scores_ols.pdf             | OLS      | manual |       | 0           |                     |                                    |
| MSE_ridge_BS.pdf                  | Ridge    | skl    | 18    | ...         | BS (200)            |                                    |
| MSE_ridge_CV.pdf                  | Ridge    | skl    | 18    | ...         | CV (10)             |                                    |
| tradeoff_ols.pdf                  | OLS      | own    | ...   | 0           | BS (200)            |                                    |
| tradeoff_ridge.pdf                | Ridge    | skl    | 18    | ...         | BS (200)            |                                    |


# Information about plots in `/output/figures/terrain/`


## Additional information:

* xy-grid: (Nx) x (Ny) = 30 x 30
* Considered 20 polynomial degrees between d = 1 and d = 20 (linarly spaced).
* Considered 12 λ-values between λ = 1.0e-05 and λ = 1.0e-02 (logarithmically spaced).
