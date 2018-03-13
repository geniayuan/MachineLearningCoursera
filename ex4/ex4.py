## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import optimize

from displayData import displayData
from nnCostFunction import nnCostFunction, nnCostFunctionGrad
from flatAndReshape import flatTheta, reshapeTheta
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from computeNumericalGradient import computeNumericalGradient
from predict import predict

## Initialization
#clear ; close all; clc

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex4data1.mat') # training data stored in arrays X, y
X = mat['X'] # size 5000*400
y = mat['y'] # size 5000*1

m = X.shape[0]
n = X.shape[1]
# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]
displayData(sel);
plt.show()

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.
print('\nLoading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weightMat = scipy.io.loadmat('ex4weights.mat')
Theta1 = weightMat['Theta1']
Theta2 = weightMat['Theta2']

pred = predict(Theta1, Theta2, X)
pp = (pred==y)
print("Given Theta Set Accuracy:", np.average(pp))

## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)
myThetaList = [Theta1, Theta2]
## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...')
# Weight regularization parameter (we set this to 0 here).
lambd = 0

J = nnCostFunction(flatTheta(myThetaList), \
                    input_layer_size, \
                    hidden_layer_size, \
                    num_labels, \
                    X, y, lambd);

print("Cost at parameters, this value should be about 0.287629)", J)

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
print('\nChecking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
lambd = 1
J = nnCostFunction( flatTheta(myThetaList), \
                    input_layer_size, \
                    hidden_layer_size,\
                    num_labels, \
                    X, y, lambd);

print('Cost at parameters,(this value should be about 0.383770)', J)

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#print('\nEvaluating sigmoid gradient...')

#g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]));
#print("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:", g)


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)
print('\nRandom Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

pred = predict(initial_Theta1, initial_Theta2, X)
pp = (pred==y)
print('Random Theta Set Accuracy:', np.average(pp))

# Unroll parameters
initial_nn_params = flatTheta([initial_Theta1,initial_Theta2])

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation...')

#  Check gradients by running checkNNGradients
#checkNNGradients()

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#
print('\nChecking Backpropagation (w/ Regularization) ...')

#  Check gradients by running checkNNGradients
lambd = 3
#checkNNGradients(lambd);

# Also output the costFunction debugging values
debug_J  = nnCostFunction(flatTheta(myThetaList), input_layer_size, \
                          hidden_layer_size, num_labels, X, y, lambd);

print('Cost at (fixed) debugging parameters (for lambda = 3) is', debug_J, \
        'this value should be about 0.576051',)

## ================= Part 8: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.
print('\nVisualizing Neural Network...')
displayData(Theta1[:, 1:])
plt.show()

## =================== Part 9: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network...')

#  You should also try different values of lambda
lambd = 0

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
myoption={'gtol': 1e-5, 'disp': True, 'maxiter':50}

thisResult = optimize.minimize( nnCostFunction, \
                                initial_nn_params, \
                                method = 'CG', \
                                jac = nnCostFunctionGrad, \
                                args = (input_layer_size, \
                                        hidden_layer_size, \
                                        num_labels, \
                                        X, y, lambd), \
                                options = myoption)

# Obtain Theta1 and Theta2 back from nn_params
trainThetaList = reshapeTheta(thisResult.x, input_layer_size, hidden_layer_size, num_labels)
trainTheta1 = trainThetaList[0] # 25*401
trainTheta2 = trainThetaList[1] # 10*26

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
pred = predict(trainTheta1, trainTheta2, X)
pp = (pred==y)
print('Training Set Accuracy:', np.average(pp))
