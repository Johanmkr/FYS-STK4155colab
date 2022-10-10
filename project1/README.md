# **Project 1**: Regression analysis and resampling methods

We present the work of Johan Mylius Kroken and Nanna Bryne in the first project in Applied Data Analysis and Machine Learning (autumn 2022). The project report (creatively named `project1.pdf`) is found in the `latex`-folder. 

The input data is located in `data` and you will find the output plots in `output/figures/**`.


## **CODE**

In the following, we assume `code` to be the home directory.
## How to run

To properly run the code, make sure you are in the `code` directory. 

The main scripts `FrankeAnalysis.py` and `terrainAnalysis.py` are written such that in order to run, they require a case-insesitive command line argument 
* `OLS` to perform the analysis using the Ordinary least squares scheme,
* `Ridge` to perform the analysis using the Ridge minimisation scheme,
* `Lasso` to perform the analysis using the Lasso minimisation scheme and
* `final` to  visualise the model we decide on,

but will ask for input if these are not given. In addition, `none` as command line or input argument will run miscellaneous code, e.g. generate 3D visualisations. To run all of the script, use `all` instead.


### Examples
```
python3 terrainAnalysis.py riDge 
### code concerning Ridge regression for terrain data is run
```

```
python3 FrankeAnalysis.py all 
### code for analysis of Franke-function is run
```

### Prerequisities
In addition to the regular libraries, such as `numpy` and `matplotlib`, the user needs the below-listed libraries downloaded on beforehand to properly run the codes.
- `sklearn`
- `pandas`
- `seaborn`
- `iamgeio`
- `tabulate`
- `collections`
- `copy`

## Structure
 
The scripts for plotting are stashed away into `plot.py` for your visual pleasure. The file includes, amongst other functionalities, options for saving figures into sub-directories (`../output/figures/Franke` or `../output/figures/terrain`) toghether with useful information about how they were obtained to a README-file. The plotting code is polluted with many quite unnecessary functionalities which may have been useful in the trying-and-failing-part of the process. An attempt to understand this task-specific code is not recommended and at your sole risk.

The main `.py`-files import various source files. We present a short description of each of them.

### **`src/utils.py`**

Useful imports, random seed, miscellaneous functions etc. is stored in this file for consistency-purposes.

### **`src/objects.py`**

Here lies a few classes for the necessary objects in linear regression, i.e. the most important contents of
$$\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon} = \tilde{\mathbf{y}} + \boldsymbol{\varepsilon}.$$
 
- $X\leftrightarrow$ `designMatrix`$\quad$ (scaled feature matrix and useful information s.a. corresponding polynomial degree)
- $\boldsymbol{\beta}\leftrightarrow$ `parameterVector`$\quad$ (scaled parameter coefficients)
- $\mathbf{y}\leftrightarrow$ `targetVector`$\quad$ (scaled data to reproduce with the model)
- $\tilde{\mathbf{y}}\leftrightarrow$ `modelVector`$\quad$ (replica of `targetVector`, sole purpose of fancy-ness)

All input is assumed to be scaled, but the most important to remember is that the input design matrix comes without intecept coloumn.

When actually doing the analysis from the main scripts, the only import needed is of the `dataPrepper`-class. Essentially, one only needs write

```
from srd.objects import dataPrepper
prepper = dataPrepper(x, y, z) # some x, y, z data 
prepper.prep()
```
and the data is splitted, scaled and created for given polynomial degrees. One can esily extract the train and test sets for desired polynomial degree.

### **`src/Regression.py`**

This `linearRegresson`-class has everything we need for performing linear regression analysis with the three desired schemes, both using `scikit-learn` and own scripts. Its unconventionally structured subclass extensions `Training` and `Prediction` (room for improvement) are initiated (for instance by `dataPrepper`) first and given as input to a motherclass object.  
### **`src/Resampling.py`**

The `Bootstrap`- and `CrossValidation`-classes inherit from a motherclass and are essentially the same, except for their `advance()`-method. In said method lies the resampling technique specific (self-explanatory) algorithm for one iteration.