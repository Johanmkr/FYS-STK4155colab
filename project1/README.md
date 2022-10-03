# **Project 1**: Regression analysis and resampling methods

We present the work of Johan Mylius Kroken and Nanna Bryne in the first project in Applied Data Analysis and Machine Learning (autumn 2022). The report is found in the `latex`-folder. The input data is located in `data` and you will find the output plots in `output/figures/**`. (__explanation????__)


## Code 

### How to run

To properly run the code, make sure you are in the `code` directory.

(EXPLAIN HOW TO DO THE DIFFERENT PARTS: `main.py`(?), `FrankeAnalysis.py`, `terrainAnalysis.py`)

**PREREQUISITES**:
- `numpy`, `matplotlib`, etc. (FIX)
- `imageio`
- `pandas`
- `sklearn`

The scripts for plotting are stashed away into `plot.py` for your visual pleasure. The file includes, amongst other functionalities, options for saving figures into sub-directories (`output/figures/Franke` or `output/figures/terrain`) toghether with useful information about how they were obtained to a README-file.

### Structure

The main `.py`-files import various source files. We present a short description of each of them.

**`code/src/utils.py`**

S

**`code/src/objects.py`**
 
- `designMatrix.py`
- `targetVector.py`
- `parameterVector.py`




# Plan (Wednesday 14.09)

### Questions
* How to properly visualise $\beta$? OK
* How to properly scale $z$? Why?
* How to plot `z_train`-points?
* Should we stop at polynomial degree $n=5$? (Part c))

### To do
* Tex resampling. 
* Figure out scaling 
* Tex scaling
* Tex tex tex


## Code (plan & structure)

Planning to have a clever code structure. Method: trial & failure, mostly the latter:(


### Tentative plan for `code/src/`:
- `designMatrix.py` presents a class for a design matrix-object, first and **foremost** that of an $n$th order polynomial in $x$ and $y$.
- `Resampling.py` should have classes and methods for resampling by bootstrapping and cross-validation
- `Regression.py` will contain methods for optimizing $\beta$ using OLS, Ridge and Lasso


