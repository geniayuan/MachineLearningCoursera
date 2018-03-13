#FINDCLOSESTCENTROIDS computes the centroid memberships for every example
#   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
#   in idx for a dataset X where each row is a single example. idx = m x 1
#   vector of centroid assignments (i.e. each entry in range [1..K])
#
import numpy as np

def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]
    m = X.shape[0]
    # You need to return the following variables correctly.
    idx = np.zeros(m)

    for i in range(m):
        thisX = X[i,:]
        temp = (thisX - centroids)**2
        dis2 = np.sum(temp, axis=1)

        idx[i] = np.argmin(dis2)

    return idx.astype(int)



    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #
