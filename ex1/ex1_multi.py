# Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize
from normalEqn import normalEqn

## ================ Part 1: Feature Normalization ================
print('Loading data ...')

#filename = input("please enter the file name: ")
filename = "ex1data2.txt"

data = np.loadtxt(filename, delimiter = ',')
m = data.shape[0]

x = data[:,0:2]
y = data[:,2]

print('Normalizing Features ...')
x_norm, x_mu, x_std = featureNormalize(x)

# Add intercept term to X
X = np.c_[np.ones(m), x_norm]
y = y.reshape(m,1)

## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#
print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.8;
num_iters = 50;

# Init Theta and Run Gradient Descent
theta = np.zeros( (X.shape[1], 1) );
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters);

# Plot the convergence graph
plt.figure(1, figsize=(8,6));
plt.plot(J_history, '-b', linewidth=2);
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent:', theta);

# Estimate the price of a 1650 sq-ft, 3 br house
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
example = (np.array([1650,3]) - x_mu) / x_std
x_test = np.insert(example, 0, 1)

price = float(np.dot(x_test,theta))
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...')
# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

# Add intercept term to X
X = np.insert(x, 0, 1, axis = 1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
example = np.array([1650,3])
x_test = np.insert(example, 0, 1)

price = float( np.dot(x_test,theta) )
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price);
