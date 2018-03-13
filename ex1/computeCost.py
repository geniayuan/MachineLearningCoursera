#COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
import numpy as np

def computeCost(X, y, theta):
    # Initialize some useful values
    m = len(y); # number of training examples
    n = len(theta); # number of variables + 1
    # X is m * n, the first column are all 1
    # y is m * 1
    # theta is n * 1

    temp = np.dot(X, theta) - y

    J = np.dot(np.transpose(temp), temp)
    J /= 2*m

    return float(J)
