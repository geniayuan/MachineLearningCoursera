#   SIGMOID Compute sigmoid function
#   g = SIGMOID(z) computes the sigmoid of z.

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).
import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
