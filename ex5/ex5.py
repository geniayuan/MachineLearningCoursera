## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import optimize

from linearRegCostFunction import linearRegCostFunction,linearRegCostFunGrad
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from validationCurve import validationCurve
## Initialization
#clear ; close all; clc

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
mat = scipy.io.loadmat('ex5data1.mat')
X = mat['X']
y = mat['y']
Xval = mat['Xval']
yval = mat['yval']
Xtest = mat['Xtest']
ytest = mat['ytest']

# m = Number of examples
m = X.shape[0]

# Plot training data
#plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
#plt.grid(True) #Always plot.grid true
#plt.ylabel('Change in water level (x)')
#plt.xlabel('Water flowing out of the dam (y)')
#plt.show()

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
#
theta = np.array([[1],[1]])
J = linearRegCostFunction(theta, X, y, 1)

print('Cost at theta = [1 ; 1]: %f '\
         '(this value should be about 303.993192)' % J)

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
#
grad = linearRegCostFunGrad(theta, X, y, 1)

print('Gradient at theta = [1 ; 1]:  [%f; %f] '\
         '(this value should be about [-15.303016; 598.250744])'\
         % (grad[0], grad[1]) )

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#
print('Train Linear Regression ...\n')
#  Train linear regression with lambda = 0
lambd = 0
theta, pred = trainLinearReg(X, y, lambd)

#  Plot fit over the data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, pred, '--', lineWidth=2)
plt.grid(True)
plt.show()


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf
#
lambd = 0;
error_train, error_val = learningCurve(X, y, Xval, yval, lambd)

xx = np.arange(m)+1
plt.plot(xx, error_train, label='Train')
plt.plot(xx, error_val, label='Cross Validation')
plt.legend()
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.grid(True)
plt.show()
#plt.axis([0 13 0 150])

print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))


## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#
p = 5
# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
#X_poly = [ones(m, 1), X_poly];                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
#X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
#X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           # Add Ones

print('Normalized Training Example 1:')
print(X_poly[1, :])

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#
lambd = 0
theta, pred = trainLinearReg(X_poly, y, lambd)

myX = np.linspace(1.2*min(X), 1.5*max(X), num = 100)
myX_poly = polyFeatures(myX, p)
myX_poly = (myX_poly - mu) / sigma
myX_poly = np.insert(myX_poly,0,1,axis=1)
preditY = myX_poly.dot(theta)

# Plot training data and fit
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.plot(myX, preditY, '--', linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit (lambda = %f)' % lambd)
plt.grid(True)
plt.show()

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambd)
plt.plot(xx, error_train, label='Train')
plt.plot(xx, error_val, label='Cross Validation')
plt.legend()
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.grid(True)
plt.show()
#plt.axis([0 13 0 150])
print('Polynomial Regression (lambda = %f)' % lambd)
print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = \
                        validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train, label='Train')
plt.plot(lambda_vec, error_val,label='Cross Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')
plt.grid(True)
plt.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(lambda_vec.size):
	print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]) )
