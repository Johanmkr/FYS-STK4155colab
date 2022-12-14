# **Project 3**: Something something 

In collaboration with three others, found [here](https://github.com/hkve/FYS-STK4155/tree/main/Project3).


# Delete the below


OBSOBS! Not ready at all. Remember to fix before deadline.

We present the work of Johan Mylius Kroken and Nanna Bryne in the third and last project in Applied Data Analysis and Machine Learning (autumn 2022). The project report (creatively named `project2.pdf`) is found in the `latex`-folder, or [here](https://github.com/Johanmkr/FYS-STK4155colab/blob/main/project2/latex/project3.pdf).

### Abstract

> We build a versatile neural ... 

## **RESULTS**

The results are located in `output/data/**`, where you find information about how they where obtained in `output/data/**/README.md`. You will find the output plots in `output/figures` with associated key parameters described in `output/figures/README.md`.

## **CODE**

In the following, we assume `code` to be the home directory.

## How to run

To properly run the code, make sure you are in the `code` directory. The `makefile` makes it easy to rerun the analyses and recreate the plots. Note that all runs will require input argument e.g. "`yes`" and it should be easy to follow the instructions given in the terminal.

* Run the SGD regression part (simple one-dimensional polynomial)
  ~~~
  make SGDregression
  ~~~
* Run the NN regression part (Franke function)
  ~~~
  make NNregression
  ~~~
* Run the classification part (Cancer data)
  - with NN
    ~~~
    make NNclassification
    ~~~
  - with logistic regression
    ~~~
    make logistic
    ~~~

And finally,

* to generate and show plots, type:
  ~~~
  make plots
  ~~~

## Prerequisities

In addition to the regular libraries, such as `numpy` and `matplotlib`, the user needs the below-listed libraries downloaded on beforehand to properly run the codes.

- `pandas`
- `seaborn`
- `tabulate`
- `collections`
- `copy`
- `autograd`
- `time`
- `tqdm`

## Structure

`generate_plots.py` uses functions from `plot.py` to create all plots. These functions read relevant information from `../output/data/**/info.pkl` so that it is saved to `../output/figures/info.pkl` and presented in `../output/figures/README.md`.

The main programs are run from the following scripts, each considering their own dataset:

* `simple_regression.py`: SGD 3rd order polynomial fit (results in `../output/data/simple_regression`)
* `regression.py`: NN Franke function fit (results in `../output/data/network_regression`)
* `classification.py`: NN Cancer data fit (results in `../output/data/network_classification`)

The main `.py`-files import various source files. We present a short description of each of them.

#### **`src/utils.py`**

Useful imports, random seed, miscellaneous functions etc. is stored in this file for consistency-purposes.

#### **`src/GradientDescent.py`**

Script for performing different sorts of gradient descent algorithms. A simple usage of the main class `SGD` is implemented in `devNeuralNetwork`.

#### **`src/devNeuralNetwork.py`**

Main neural network class. Has no external functions, does only contain the neural network class.

#### **`src/LossFunctions.py`**

Library of loss functions.

#### **`src/ActivationFunction.py`**

Library of activation functions.

#### **`src/ExtendAndCollapse.py`**

Class to hold the weight and biases of every layer, expanded into one large array for weights and one for biases. The class has functionality to compute the gradients of the whole network simultaneously, and collapse the weights and biases back into their layer individuals.

#### **`src/Layer.py`**

Class to contain the layer objects. It has class variables to keep track of the number of neurons, their value, their biases and the weights feeding forward to this layer.

#### **`src/infoFile_ala_Nanna.py`**

Working title for a work-in-progress. Unimportant for the actual running and results. Purpose is to save information about results and overwrite only the relevant features.