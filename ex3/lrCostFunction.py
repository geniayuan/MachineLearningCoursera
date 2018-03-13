#function [J, grad] = lrCostFunction(theta, X, y, lambda)
#LRCOSTFUNCTION Compute cost and gradient for logistic regression with
#regularization
#   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters.
import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, lambd):
    m = X.shape[0]
    n = X.shape[1]
    theta = theta.reshape(n,1)
    y = y.reshape(m,1)

    h = sigmoid(X.dot(theta)).reshape(m,1)
    p1 = - np.dot(y.T, np.log(h))
    p2 = - np.dot(1-y.T, np.log(1-h))
    #print("h size", h.shape, "p1 size", p1.shape, "p2 size", p2.shape)
    reg = lambd/2 * np.dot(theta[1:,].T, theta[1:,])

    return float(p1 + p2 + reg)/m

def lrCostFunGrad(theta, X, y, lambd):
    m = X.shape[0]
    n = X.shape[1]
    theta = theta.reshape(n,1)
    y = y.reshape(m,1)

    h = sigmoid(X.dot(theta)).reshape(m,1)
    grad = np.dot(X.T, h-y)
    regGrad = lambd * theta
    regGrad[0] = 0
    #print("grad size", grad.shape)
    #print("regGrad size", regGrad.shape)
    temp = (grad + regGrad)/m

    return temp[:,0]

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations.
#
# Hint: When computing the gradient of the regularized cost function,
#       there're many possible vectorized solutions, but one solution
#       looks like:
#           grad = (unregularized gradient for logistic regression)
#           temp = theta;
#           temp(1) = 0;   # because we don't add anything for j = 0
#           grad = grad + YOUR_CODE_HERE (using the temp variable)
#
