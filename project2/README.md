# **Project 2**: Classification and regression

We present the work of Johan Mylius Kroken and Nanna Bryne in the second project in Applied Data Analysis and Machine Learning (autumn 2022). The project report (creatively named `project2.pdf`) is found in the `latex`-folder. 

The results are located in `output/data/**`, where you find information about how they where obtained in `output/data/**/README.md`. You will find the output plots in `output/figures` with associated key parameters described in `output/figures/README.md`.


## **CODE**

In the following, we assume `code` to be the home directory. 
## How to run

To properly run the code, make sure you are in the `code` directory. 

DESCRIBE HOW TO USE MAKEFILE!

### Examples
```
make plots
### ???
```


### Prerequisities
In addition to the regular libraries, such as `numpy` and `matplotlib`, the user needs the below-listed libraries downloaded on beforehand to properly run the codes.
- `pandas`
- `seaborn`
- `tabulate`
- `collections`
- `copy`

## Structure
 
`generate_plots.py` uses functions from `plot.py` to create all plots. These functions read relevant information from `../output/data/**/info.pkl` so that it is saved to `../output/figures/info.pkl` and presented in `../output/figures/README.md`.

The main programs are run from the following scripts:

* `simple_regression.py`: SGD 3rd order polynomial fit (results in `../output/data/simple_regression`)
* `regression.py`: NN Franke function fit (results in `../output/data/network_regression`)
* `classification.py`: NN Cancer data fit( results in `../output/data/network_classification`)

The main `.py`-files import various source files. We present a short description of each of them.

### **`src/utils.py`**

Useful imports, random seed, miscellaneous functions etc. is stored in this file for consistency-purposes.
#FIXME

### **`src/GradientDescent.py`**

SHORT DESC.

### **`src/NeuralNetwork.py`**

SHORT DESC.