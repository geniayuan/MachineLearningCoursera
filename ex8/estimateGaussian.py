#ESTIMATEGAUSSIAN This function estimates the parameters of a
#Gaussian distribution using the data in X
#   [mu sigma2] = estimateGaussian(X),
#   The input X is the dataset with each n-dimensional data point in one row
#   The output is an n-dimensional vector mu, the mean of the data set
#   and the variances sigma^2, an n x 1 vector
#
import numpy as np

def estimateGaussian(X, multi = False):
    m = X.shape[0]

    mu = np.mean(X, axis = 0)

    if not multi: # do not consider covariance
        sigma2 = 1/m * np.sum( (X-mu)*(X-mu), axis=0)
    else: # consider covariance
        sigma2 = 1/m * (X-mu).T.dot(X-mu)

    #print("X size", X.shape)
    #print("mu size:", mu.shape, mu)
    #print("sigma2 size:", sigma2.shape, sigma2)
    return mu, sigma2

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu(i) should contain the mean of
    #               the data for the i-th feature and sigma2(i)
    #               should contain variance of the i-th feature.
    #
