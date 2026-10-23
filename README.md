# Lab 2: Deep Learning

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b1/MNIST_dataset_example.png" height="300">

## Table of Contents

- [Introduction](#introduction)
- [Getting started](#getting-started)
    - [Creating a virtual environment](#creating-a-virtual-environment)
    - [Installing project dependencies](#installing-project-dependencies)
    - [Working with Jupyter Notebooks](#working-with-jupyter-notebooks)
- [NumPy & PyTorch tutorial](#numpy--pytorch-tutorial)
- [Part I: First principles](#part-i-first-principles)
    - [Exercise 1: Linear model](#exercise-1-linear-model)
    - [Exercise 2: Activation function](#exercise-2-activation-function)
- [Part II: Linear regression](#part-ii-linear-regression)
    - [Exercise 3: Define the regression model](#exercise-3-define-the-regression-model)
    - [Exercise 4: Train and evaluate the regression model](#exercise-4-train-and-evaluate-the-regression-model)
    - [Exercise 5: Run the regression experiment](#exercise-5-run-the-regression-experiment)
    - [Exercise 6: Investigate the role of hyperparameters](#exercise-6-investigate-the-role-of-hyperparameters)
- [Part III: Image classification](#part-iii-image-classification)
    - [Exercise 7: Define the classification model](#exercise-7-define-the-classification-model)
    - [Exercise 8: Train and evaluate the classification model](#exercise-8-train-and-evaluate-the-classification-model)
    - [Exercise 9: Run the classification experiment](#exercise-9-run-the-classification-experiment)
    - [Exercise 10: Investigate the role of hyperparameters](#exercise-10-investigate-the-role-of-hyperparameters)
- [Extra: Advanced image classification](#extra-advanced-image-classification)
    - [Exercise 11: CIFAR-10](#exercise-11-cifar-10)
    - [Exercise 12: CIFAR-100](#exercise-12-cifar-100)

## Introduction

In this lab, you will study the complete deep-learning workflow at increasing levels of abstraction. You will first implement the essential learning operations with basic tensor calculations, then use PyTorch's model, automatic differentiation, and optimization tools for a regression problem, and finally train and tune a neural network for image classification.

The repository contains starter code for the models and learning procedures, along with data-loading, visualization, and checkpointing utilities. Parts II and III follow a similar structure but address independent learning tasks. Throughout the lab, pay close attention to tensor shapes, the responsibilities of the model and learner classes, and the distinction between training, validation, and testing.

> **Important:** For the required implementations, only replace code inside the sections marked with `TODO`. You may adjust the settings identified in the experimentation exercises, but do not change the names, parameters, or scope of the provided functions and classes.

## Getting started

There are a few steps to take to run the code for this lab on your machine.


### Creating a virtual environment

First, it is important to isolate the code and the packages it depends upon from those of other projects on your machine. Therefore we strongly encourage you to use a virtual environment.

You can create a virtual environment with:

```
python -m venv venv
```

This command will create an empty virtual environment named `venv` in the current directory. After creation, you can activate the virtual environment.

Using Pycharm Terminal:
```
.\venv\Scripts\activate
```

Using Windows CMD:
```
cd path_to_project_folder
.\venv\Scripts\activate.bat
```

This environment has its own independent set of packages, installed in the `venv` directory, and is isolated from the base environment and other virtual environments.

You can deactivate the virtual environment with:

```
deactivate
```

For further information, read the [venv documentation](https://docs.python.org/3/library/venv.html).


### Installing project dependencies

Next, you will need to install the following packages in the newly created virtual environment to run the code for this lab.

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

Use the following command to install these dependencies:

```
pip install torch torchvision matplotlib numpy
```

**Make sure the virtual environment is activated before installing these packages!**

If you would like to install PyTorch with GPU hardware acceleration enabled, consult the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the most up-to-date information.


### Working with Jupyter Notebooks

Finally, you will need to install the IPython kernel and configure your IDE to work with Jupyter Notebooks.

Make sure the virtual environment is activated. Then install the `ipykernel` package with:

```
pip install ipykernel
```

PyCharm Community Edition does not support Jupyter Notebooks. However, you can run them locally in the browser using the `jupyter` package. Install this package with:

```
pip install jupyter
```

Next, run Jupyter in your browser by running the following command in the terminal. This will open a browser-based application where you can open, edit, and run the Jupyter Notebook files (.ipynb).

```
jupyter notebook
```

Visual Studio Code has native support for working with Jupyter Notebooks. Follow the instructions in [Manage Jupyter Kernels in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) to select the kernel from the virtual environment. Editing and running notebooks is explained in [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

For further information on the ins and outs of Jupyter Notebooks, explore the [Jupyter website](https://jupyter.org/) or read the [IPython kernel documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## NumPy & PyTorch tutorial

Before implementing a neural network, you should familiarize yourself with NumPy and PyTorch, two core libraries for numerical computing and deep learning. In deep learning, data and model parameters are represented by multidimensional collections of numbers. NumPy calls these objects **arrays**, while PyTorch calls them **tensors**.

The [NumPy & PyTorch tutorial](tutorial.ipynb) introduces these concepts through practical examples and explores how multidimensional data is represented, inspected, transformed, and used in calculations. This will help you reason about the data and operations in the exercises that follow.

Work through the notebook as follows:

1. Open [`tutorial.ipynb`](tutorial.ipynb) and select the kernel from the virtual environment you configured above.
2. Start at the top and run the cells in order. Later examples build on terminology and techniques introduced earlier.
3. For each example, identify what every dimension represents. Before running a code cell, try to predict the result, its shape, and its data type; then check your prediction against the output.
4. Pay attention to how each operation affects the data's values and dimensions. Experiment with copies of the code cells by changing values or shapes, and use the resulting output or error message to test your understanding.

## Part I: First principles

In the first part of the lab, you will build and train two small machine-learning models using basic tensor operations. The purpose is to understand every step of the process: initializing weights, calculating an output and loss, deriving gradients, updating weights, and repeating this process during training.

You'll start with implementing and train a two-neuron model with a pure linear output in excercise 1. Next, you'll extend the linear model with a sigmoid activation function in excercise 2.

You'll work with the following files:

- [`first_principles_models.py`](first_principles_models.py) contains the model classes to be completed
- [`first_principles.ipynb`](first_principles.ipynb) provides implementation guidance, visualizations, and intermediate checks for each exercise

Complete the TODOs in [`first_principles_models.py`](first_principles_models.py) in the order in which they are introduced in the notebook [`first_principles.ipynb`](first_principles.ipynb). Use the notebook to run each stage interactively, inspect the tensors, and verify that your implementations behave correctly. After changing the module, restart the notebook kernel and run all cells from the beginning so that the updated classes are imported.

> **Note:** For these exercises you should not use pre-implemented models or layers from `torch.nn` or their functional equivalents from `torch.nn.functional`. Also, you should not use `backward`, `autograd.grad`, and the `.grad` tensor attribute to calculate the gradients or an optimizer from `torch.optim` to update the weights. The purpose of these excercises is to implement these functionalities from scratch.

### Exercise 1: Linear model

Implement the `LinearModel` class in [`first_principles_models.py`](first_principles_models.py). The model maps one RGB image to the two values in the provided target tensor.

Complete the following methods:

- `init_weights`: initialize weights for every input-output connection.
- `calculate_output`: calculate the linear output from the multidimensional input.
- `calculate_loss`: calculate the mean squared error between the output and target.
- `calculate_gradients`: calculate the gradients of the loss with respect to the weights.
- `update_weights`: update the weights with one gradient-descent step.

### Exercise 2: Activation function

Implement the `LinearSigmoidModel` class that extends the linear model by applying a sigmoid activation function to its linear output. Reuse the behavior inherited from `LinearModel` where appropriate.

Complete the following methods:

- `calculate_output`: extend the output calculation with the activation function.
- `calculate_gradients`: extend the gradient calculation to account for the activation function.

## Part II: Linear regression

In this part, you will use PyTorch's model, automatic-differentiation, and optimization tools to implement a linear regression model to estimate the price of a house from its size.

The provided `HousesDataset` generates noisy samples from an "unknown" linear relationship. Your task is to fit a linear model to these observations and recover the underlying relationship as accurately as possible despite the noise. Although this simple regression problem does not require a deep-learning framework, it provides a clear setting in which to learn the core PyTorch concepts and its standard training workflow.

You'll work with the following files:

- [`data.py`](data.py) defines the houses dataset
- [`models.py`](models.py) defines the linear regression model
- [`learners.py`](learners.py) contains the regression training and evaluation procedures
- [`run_linear_regression.py`](run_linear_regression.py) configures and runs the experiment and visualizes its results

### Exercise 3: Define the regression model

Complete the `LinearRegressionModel` class in [`models.py`](models.py). Define a linear layer with one input feature and one output value in the constructor, and implement the `forward` method so that it returns the layer's prediction for a batch of house sizes. For this step, you might want to consult the documentation of the [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) module.

> **Note:** The prediction and target tensors must have the same shape. 

### Exercise 4: Train and evaluate the regression model

Complete the `RegressionLearner` class in [`learners.py`](learners.py).

You will need to complete the `train` method with the update step for each batch using the configured model, the criterion (loss function), and the optimizer.

Furthermore, you will need to complete the `test` method. This should calculate and return the root mean squared error (RMSE) of the model on the given dataset without tracking gradients. RMSE penalizes large errors similar to the mean squared error (MSE), but is expressed in the same units as the target data, so the reported result can be interpreted as an error in euros.

> **Note:** Make sure that test loss values are aggregated over the number of samples rather than the number of batches.

### Exercise 5: Run the regression experiment

Next, complete the TODOs in [`run_linear_regression.py`](run_linear_regression.py) by instantiating the `LinearRegressionModel` and configuring an appropriate criterion (loss function) and optimizer.

The criterion is a function that calculates the training error between model outputs and targets. See the documentation on the [torch.nn](https://docs.pytorch.org/docs/2.14/nn.html#loss-functions) and [torch.nn.functional](https://docs.pytorch.org/docs/2.14/nn.functional.html#loss-functions) libraries for more information about the loss functions provided by PyTorch.

The optimizer stores references to the model parameters and updates them after gradients have been calculated. Common choices include stochastic gradient descent (`torch.optim.SGD`) and Adam (`torch.optim.Adam`). See the documentation on the [torch.optim](https://docs.pytorch.org/docs/2.14/optim.html) library for more information about the optimization algorithms provided by PyTorch.

After completing the pipeline, run [`run_linear_regression.py`](run_linear_regression.py). Examine the training curve, compare the training and test RMSE values, and inspect the learned regression line. Because the dataset contains noise, the model is not expected to reproduce every data point exactly. Instead, verify that the training loss generally decreases and that the learned function approaches the underlying function as training progresses.

### Exercise 6: Investigate the role of hyperparameters

Investigate how the following hyperparameters affect the model's training behavior and performance:

- the learning rate
- the batch size
- the number of training epochs
- the size of the training dataset

These hyperparameters are defined near the top of [`run_linear_regression.py`](run_linear_regression.py).

Change one setting at a time so that the results remain comparable. After each run, examine the training curve and compare the training and test RMSE values. Explain how the change affected the speed and stability of training, as well as the model's performance on unseen data.

## Part III: Image classification

In this part, you will use PyTorch's neural-network components to implement a fully connected model that classifies handwritten digits from the MNIST dataset.

MNIST contains 28-by-28-pixel grayscale images of the digits zero through nine. In the following exercises, you will define a three-layer classification model and implement its complete training and evaluation workflow using a multiclass loss, validation after each epoch, and accuracy as an evaluation metric.

You'll work with the following files:

- [`models.py`](models.py) defines the classification models
- [`learners.py`](learners.py) contains the training, validation, and test procedures
- [`run_classification.py`](run_classification.py) configures and runs the experiment and visualizes its results

### Exercise 7: Define the classification model

Complete the `NeuralNetworkClassificationModel` class in [`models.py`](models.py).

First, define a fully connected architecture with three non-linear layers in the constructor. Take into account the dimensionality of the inputs and the number of classes for which the model must produce an output. Determine suitable sizes for the two hidden layers and select appropriate activation functions.

Then, implement the `forward` method so that it returns a prediction for each image in a batch, using the three layers. Here is also the place to implement any final data transformations, such as flattening the images. For this step, You might want to consult the documentation on the built-in modules provided by the [`torch.nn`](https://pytorch.org/docs/stable/nn.html) library.

> **Important:** Ensure the model preserves the batch dimension.

> **Hint:** When choosing the activation function for the final layer, consider how the criterion (loss function) will use the model's output values. Consult the loss function's documentation to determine the form of input it expects, and make sure that this matches what the model returns.

### Exercise 8: Train and evaluate the classification model

Complete the `ClassificationLearner` class in [`learners.py`](learners.py).

You will need to complete the `train` method with the update step for each batch using the configured model, criterion, and optimizer.

Furthermore, you will need to complete the `_evaluate` method. This method should evaluate every batch without tracking gradients and return the mean loss and accuracy over the complete dataset. Select the predicted class for each image from the model output.

The learner uses `_evaluate` to measure performance on the validation subset after each training epoch and saves the model with the lowest validation loss. The `test` method reuses the same evaluation procedure for the test set.

> **Note:** Make sure that both loss and accuracy are aggregated over the number of samples rather than the number of batches.

### Exercise 9: Run the classification experiment

Next, complete the TODOs in [`run_classification.py`](run_classification.py) by instantiating the `NeuralNetworkClassificationModel` and configuring an appropriate criterion and optimizer.

The criterion must be suitable for multiclass classification and compatible with the model output defined in Exercise 7. The optimizer must be configured with the model parameters and the provided learning rate. Consult the PyTorch documentation for [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) and [`torch.optim`](https://pytorch.org/docs/stable/optim.html) when selecting these components.

After completing the pipeline, run [`run_classification.py`](run_classification.py). Examine the training and validation curves, report the test loss and accuracy, and compare the displayed predictions with their labels. Verify that the model with the lowest validation loss is loaded before evaluating the test set.

### Exercise 10: Investigate the role of hyperparameters

Investigate how the following hyperparameters affect the model's training behavior and performance:

- the learning rate
- the batch size
- the number of training epochs
- the number of hidden layers
- the size of each hidden layer

The learning rate, batch size, and number of epochs are defined near the top of [`run_classification.py`](run_classification.py). Adapt the architecture of `NeuralNetworkClassificationModel` to vary the number and size of the hidden layers.

Change one setting at a time so that the results remain comparable. After each run, examine the training and validation curves and compare the validation loss and accuracy. Explain how the change affected the speed and stability of training, as well as the model's performance on unseen data.

Use the validation results to select your preferred configuration, and evaluate that configuration on the test set only after tuning is complete. Repeatedly using the test dataset to adjust the model would make the test set part of the development process.

## Extra: Advanced image classification

### Exercise 11: CIFAR-10

Create a copy of [`run_classification.py`](run_classification.py) and adapt it to train and evaluate a classifier on the CIFAR-10 dataset. This dataset is provided by the `torchvision.datasets` library, similar to the MNIST dataset. Account for the different input shape and retain ten output classes. You may modify or add model code if your chosen architecture requires it.

### Exercise 12: CIFAR-100

Create a copy of [`run_classification.py`](run_classification.py) and adapt it to train and evaluate a classifier on the CIFAR-100 dataset provided by the `torchvision.datasets` library. Obtaining competitive accuracies on this dataset will require a more suitable neural network architecture for learning from images and additional hyperparameter tuning.
