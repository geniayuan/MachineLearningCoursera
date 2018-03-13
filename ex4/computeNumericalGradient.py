#COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
#and gives us a numerical estimate of the gradient.
#   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
#   gradient of the function J around theta. Calling y = J(theta) should
#   return the function value at theta.

# Notes: The following code implements numerical gradient checking, and
#        returns the numerical gradient.It sets numgrad(i) to (a numerical
#        approximation of) the partial derivative of J with respect to the
#        i-th input argument, evaluated at theta. (i.e., numgrad(i) should
#        be the (approximately) the partial derivative of J with respect
#        to theta(i).)
#

import numpy as np

def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    e = 1e-4
    #col = np.array([7396, 5523, 9584, 1020,5080,8017,482,393,3981,9346])
    #for p in col:
    for p in range(theta.size):
        # Set perturbation vector
        perturb = np.zeros(theta.shape)
        perturb[p] = e

        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

    return numgrad
