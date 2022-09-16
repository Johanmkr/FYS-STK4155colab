# Project 1

## Plan (Wednesday 14.09)

### Questions
* How to properly visualise $\beta$? OK
* How to properly scale $z$? Why?
* How to plot `z_train`-points?
* Should we stop at polynomial degree $n=5$? (Part c))

### To do
* Speak to CM or Anna about pt. c) (show that...). **Nansen**
* Check that pt. a) is correct. **Jojo**
* Compare own OLS code to sklearn code. **Jojo**
* Write method (formula for MSE, ...) in report. **Jojo/Nansen**
* Begin with clever code structure (idea: plot.py, utils.py, main.py). **Nansen**

## Code (plan & structure)

Planning to have a clever code structure. Method: trial & failure, mostly the latter:(


### Tentative plan for `code/src/`:
- `designMatrix.py` presents a class for a design matrix-object, first and foremost that of an $n$th order polynomial in $x$ and $y$.
- `Resampling.py` should have classes and methods for resampling by bootstrapping and cross-validation
- `Regression.py` will contain methods for optimizing $\beta$ using OLS, Ridge and Lasso