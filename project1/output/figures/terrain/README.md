|                    | scheme   | mode   | $d$   | $\lambda$   | resampling (iter)   | mark                   |
|:-------------------|:---------|:-------|:------|:------------|:--------------------|:-----------------------|
| beta_hist_ols.pdf  | ols      | own    | 8     | 0           | BS (100)            |                        |
| data3D.pdf         |          |        |       |             |                     | visualise x, y, z data |
| MSE_hist_ols.pdf   | ols      | own    | 8     | 0           | BS (100)            |                        |
| MSE_ols_BS.pdf     | ols      | own    | ...   | 0           | BS (5)              |                        |
| MSE_ols_CV.pdf     | nan      | nan    | nan   | nan         | nan                 |                        |
| MSE_ridge_BS.pdf   | ridge    | own    | 8     | ...         | BS (100)            |                        |
| MSE_ridge_CV.pdf   | ridge    | own    | 8     | ...         | CV (5)              |                        |
| tradeoff_ols.pdf   | ols      | own    |       |             | BS (100)            |                        |
| tradeoff_ridge.pdf | ridge    | own    |       |             | BS (100)            |                        |


# Information about plots in `/output/figures/terrain/`


## Additional information:

* xy-grid: (Nx) x (Ny) = 30 x 30
* Considered 18 polynomial degrees between d = 1 and d = 18 (linarly spaced).
* Considered 10 λ-values between λ = 1.0e-06 and λ = 1.0e-01 (logarithmically spaced).
