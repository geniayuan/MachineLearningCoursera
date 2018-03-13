# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
#
import numpy as np

# only implemented for vector
def mapFeature(X1, X2):
    m = len(X1)
    degree = 6;
    out = np.ones( (m,28) )

    k = 1
    for i in range(1,degree+1):
        for j in range(i+1):
            col = (X1**(i-j))*(X2**j)
            col.reshape((m,1))
            out[:,k] = col
            k += 1

    return out
