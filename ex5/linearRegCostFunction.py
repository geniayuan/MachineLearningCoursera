#LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
#regression with multiple variables
#   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
#   cost of using theta as the parameter for linear regression to fit the
#   data points in X and y. Returns the cost in J and the gradient in grad
import numpy as np

def linearRegCostFunction(theta, X, y, lambd):
    # Initialize some useful values
    m = X.shape[0] # number of training examples
    # Add intercept term to X
    Xtemp = np.insert(X,0,1,axis=1)
    theta = theta.reshape(Xtemp.shape[1],1)

    # You need to return the following variables correctly
    temp = Xtemp.dot(theta) - y

    J = np.sum(temp*temp)
    reg = lambd * np.sum(theta[1:]*theta[1:])

    J = (J+reg)/(2*m)
    #print("cost", J)

    return J

def linearRegCostFunGrad(theta, X, y, lambd):
    # Initialize some useful values
    m = X.shape[0] # number of training examples
    # Add intercept term to X
    Xtemp = np.insert(X,0,1,axis=1)
    y = y.reshape(m,1)
    theta = theta.reshape(Xtemp.shape[1],1)

    # You need to return the following variables correctly
    grad = np.dot(Xtemp.T, Xtemp.dot(theta)-y)
    reg = lambd * theta
    reg[0] = 0

    grad = 1/m * (grad+reg).flatten() # as an array
    #print("grad", grad)
    #print("Xtemp size:", Xtemp.shape)
    #print("theta size:", theta.shape)
    #print("grad size:", grad.shape)
    #print("reg size:", reg.shape)

    return grad

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #
