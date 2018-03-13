#RBFKERNEL returns a radial basis function kernel between x1 and x2
#   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
#   and returns the value in sim
import numpy as np

def gaussianKernel(x1, x2, sigma):
    #Ensure that x1 and x2 are column vectors
    x1temp = x1.reshape(-1,1)
    x2temp = x2.reshape(-1,1)

    temp = x1temp - x2temp

    # You need to return the following variables correctly.
    return np.exp( -np.sum(temp*temp)/(2*sigma**2) )
