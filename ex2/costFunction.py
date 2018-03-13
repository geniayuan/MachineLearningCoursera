#   COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y, lbd=0):
    m = len(y)

    h = sigmoid(X.dot(theta))
    p1 = - np.dot(y.T, np.log(h))
    p2 = - np.dot(1-y.T, np.log(1-h))
    reg = lbd/2 * np.dot(theta[1:,].T, theta[1:,])

    return float(p1+p2+reg)/m

def costFunctionGradient(theta, X, y, lbd=0):
    m = len(y)
    n = len(theta)

    temp = sigmoid(X.dot(theta)) - y
    regdev = lbd * theta
    regdev[0] = 0

    return (np.dot(np.transpose(X), temp) + regdev)/m
