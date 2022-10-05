|                           | scheme   | mode   | $d$   | $\lambda$   | resampling (iter)   | mark                               |
|:--------------------------|:---------|:-------|:------|:------------|:--------------------|:-----------------------------------|
| beta_hist_ols.pdf         | OLS      | own    | 8     | 0           | BS (200)            |                                    |
| beta_ols.pdf              | OLS      | manual |       | 0           |                     | $β$'s grouped by order $d$         |
| beta_polydeg_hist_ols.pdf | OLS      | own    | 8     | 0           | BS (200)            |                                    |
| comparison3D.pdf          |          |        |       |             |                     | visualise x, y, z data             |
| comparison3D_ols.pdf      | OLS      | manual | 8     | 0           |                     | prediction set                     |
| comparison3D_ridge.pdf    | Ridge    | manual | 18    | 0.0001      |                     | prediction set                     |
| data3D.pdf                |          |        |       |             |                     | visualise x, y, z data             |
| MSE_heatmap_ridge_CV.pdf  | Ridge    | skl    |       |             | CV (9)              | 30 $λ$'s from 1.00e-09 to 1.00e-09 |
| MSE_hist_ols.pdf          | ols      | own    | 8     | 0           | BS (100)            |                                    |
| MSE_hist_olsTrue.pdf      | OLS      | own    | 8     | 0           | BS (200)            |                                    |
| MSE_ols_BS.pdf            | OLS      | own    | ...   | 0           | BS (9)              |                                    |
| MSE_ols_CV.pdf            | nan      | nan    | nan   | nan         | nan                 |                                    |
| MSE_R2_scores_ols.pdf     | OLS      | manual |       | 0           |                     |                                    |
| MSE_ridge_BS.pdf          | Ridge    | own    | 8     | ...         | BS (200)            |                                    |
| MSE_ridge_CV.pdf          | Ridge    | own    | 8     | ...         | CV (5)              |                                    |
| tradeoff_ols.pdf          | OLS      | own    | ...   | 0           | BS (200)            |                                    |
| tradeoff_ridge.pdf        | Ridge    | own    | 8     | ...         | BS (200)            |                                    |


# Information about plots in `/output/figures/terrain/`


## Additional information:

* xy-grid: (Nx) x (Ny) = 30 x 30
* Considered 20 polynomial degrees between d = 1 and d = 20 (linarly spaced).
* Considered 30 λ-values between λ = 1.0e-09 and λ = 1.0e-04 (logarithmically spaced).
