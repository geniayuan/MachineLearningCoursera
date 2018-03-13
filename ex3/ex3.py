## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
#clear ; close all; clc
import numpy as np
import scipy.io
from scipy import optimize
import matplotlib.pyplot as plt

from displayData import displayData
from lrCostFunction import lrCostFunction,lrCostFunGrad
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex3data1.mat'); # training data stored in arrays X, y
X = mat['X'] # size 5000*400
y = mat['y'] # size 5000*1

m = X.shape[0]
n = X.shape[1]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]
#print(sel.shape)

displayData(sel);
plt.show()

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print('Testing lrCostFunction() with regularization......')

theta_t = np.array([[-2], [-1], [1], [2]])
#X_t = [ones(5,1) reshape(1:15,5,3)/10];
# reshape in numpy is row first, matlab col first
X_t = np.arange(1,16).reshape(3,5)/10
X_t = np.insert(X_t.T, 0, 1, axis=1)
y_t = np.array([[1],[0],[1],[0],[1]]) >= 0.5
lambda_t = 3;
J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad  = lrCostFunGrad(theta_t, X_t, y_t, lambda_t)

print('Cost:', J)
print('Expected cost: 2.534819')
print('Gradients:', grad)
print('Expected gradients:');
print(' 0.146561, -0.548558, 0.724722, 1.398003')

## ============ Part 2b: One-vs-All Training ============
print('Training One-vs-All Logistic Regression......')

lambd = 0.1
all_theta = oneVsAll(X, y, num_labels, lambd)

## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)
pp = (pred==y)

print('Training Set Accuracy: %f', np.average(pp) * 100)

wrongcol = np.nonzero(pred != y)[0]
wrongNum = len(wrongcol)
print("wrong predict number:", wrongNum)

rand_indices = np.random.permutation(wrongNum)
selCol = wrongcol[rand_indices[:100]]
#print(sel.shape)
sel = X[selCol, :]
print("prediction:")
print(pred[selCol, :].reshape(10,10))
print("true value:")
print(y[selCol, :].reshape(10,10))

displayData(sel)
plt.show()
