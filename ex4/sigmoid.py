#function g = sigmoid(z)
#SIGMOID Compute sigmoid functoon
#   J = SIGMOID(z) computes the sigmoid of z.
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
