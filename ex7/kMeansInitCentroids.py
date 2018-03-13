#KMEANSINITCENTROIDS This function initializes K centroids that are to be
#used in K-Means on the dataset X
#   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
#   used with the K-Means on the dataset X
#
import numpy as np

def kMeansInitCentroids(X, K):
    m = X.shape[0]
    rand_indices = np.random.permutation(m)

    centroids = X[rand_indices[:K], :]

    return centroids

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #
