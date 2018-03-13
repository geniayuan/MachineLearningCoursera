## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
#clear ; close all; clc

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#from matplotlib import cm
#from mpl_toolkits.mplot3d import axes3d, Axes3D

from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from costFunction import costFunction, costFunctionGradient
from mapFeature import mapFeature
from predict import predict

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

#filename = input("please enter the file name: ")
filename = "ex2data2.txt"
data = np.loadtxt(filename, delimiter = ',')
m = data.shape[0] # of training examples
n = data.shape[1] -1# number of features

x = data[:, :-1]
y = data[:,n]

## ==================== Part 0: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
plotData(x, y)
plt.ylabel('Microchip Test 1')
plt.xlabel('Microchip Test 2')

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#
y = y.reshape(m,1)

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(x[:,0], x[:,1])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
lambd = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunction(initial_theta, X, y, lambd)
grad = costFunctionGradient(initial_theta, X, y, lambd)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:', grad[0:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1],1))
cost = costFunction(test_theta, X, y, 10)
grad = costFunctionGradient(test_theta, X, y, 10)

print('Cost at test theta (with lambda = 10):', cost)
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only:', grad[0:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambd = 1;

# Set Options
#options = optimset('GradObj', 'on', 'MaxIter', 400);
myoption = {"maxiter":500, "disp":False}
result = optimize.minimize(costFunction, initial_theta, args=(X,y,lambd),  method='BFGS', options = myoption)
theta = np.array([result.x]) # row vector
cost = result.fun
# Optimize
#[theta, J, exit_flag] = ...
#	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
#theta = optimize.fmin_bfgs(costFunction, x0=initial_theta, fprime = costFunctionGradient, maxiter=400, args = (X,y,lambd))
# Plot Boundary
plotDecisionBoundary(theta, X, y);
# Labels and Legend
plt.show()

# Compute accuracy on our training set
p = predict(theta.T, X)
pp = (p==y)
print('Train Accuracy: ', np.average(pp) * 100);
print('Expected accuracy (with lambda = 1): 83.1 (approx)');
