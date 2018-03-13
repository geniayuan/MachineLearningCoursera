#PCA Run principal component analysis on the dataset X
#   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
#   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
#
import numpy as np

def pca(X):
    # Useful values
    m, n = X.shape
    #print("m,n:",m,",",n)
    X_coherent = 1/m * np.dot(X.T, X)
    #print("X_coh:", X_coherent)

    u, s, vh = np.linalg.svd(X_coherent)

    # You need to return the following variables correctly.
    return u, s, vh

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #
