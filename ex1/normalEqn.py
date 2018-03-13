#   NORMALEQN Computes the closed-form solution to linear regression
#   NORMALEQN(X,y) computes the closed-form solution to linear
#   regression using the normal equations.
import numpy as np

def normalEqn(X, y):

    tempInv = np.linalg.inv( np.dot(np.transpose(X), X) )
    temp = np.dot( tempInv, np.transpose(X))
    theta = np.dot(temp,y)

    return theta
