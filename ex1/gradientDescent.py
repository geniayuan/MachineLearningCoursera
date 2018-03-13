import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    # Initialize some useful values
    m = len(y); # number of training examples
    n = len(theta); # number of variables + 1
    # X is m * n, the first column are all 1
    # y is m * 1
    # theta is n * 1

    #thetaHistory = []
    values = []
    for i in range(iterations):
        J = computeCost(X, y, theta)
        #thetaHistory.append(theta)
        values.append(J)

        temp = np.dot(X, theta) - y
        theta = theta - alpha/m * np.dot(np.transpose(X), temp)

    J = computeCost(X, y, theta)
    values.append(J)

    return theta, values
