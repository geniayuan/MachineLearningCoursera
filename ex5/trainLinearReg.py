#TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
#regularization parameter lambda
#   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
#   the dataset (X, y) and regularization parameter lambda. Returns the
#   trained parameters theta.
#
import numpy as np
from scipy import optimize

from linearRegCostFunction import linearRegCostFunction,linearRegCostFunGrad

def trainLinearReg(X, y, lambd):
    # size info, indlue intercept term
    n = X.shape[1]+1

    # Initialize Theta
    # optimize.minimize takes a flattened array
    initial_theta = np.zeros(n)
    #epsilon = 0.5
    #initial_theta = np.random.rand(1,n) * 2 * epsilon - epsilon

    myoption={'gtol': 1e-10, 'disp': True, 'maxiter':1000}
    thisResult = optimize.minimize( linearRegCostFunction, \
                                    initial_theta,\
                                    method = 'CG', \
                                    jac = linearRegCostFunGrad, \
                                    args = (X, y, lambd), \
                                    options = myoption)

    fit_theta = thisResult.x.reshape(n,1)

    pred = np.insert(X,0,1,axis=1).dot(fit_theta)

    return fit_theta, pred
