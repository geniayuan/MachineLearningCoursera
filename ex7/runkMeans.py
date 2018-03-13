#RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
#is a single example
#   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
#   plot_progress) runs the K-Means algorithm on data matrix X, where each
#   row of X is a single example. It uses initial_centroids used as the
#   initial centroids. max_iters specifies the total number of interactions
#   of K-Means to execute. plot_progress is a true/false flag that
#   indicates if the function should also plot its progress as the
#   learning happens. This is set to false by default. runkMeans returns
#   centroids, a Kxn matrix of the computed centroids and idx, a m x 1
#   vector of centroid assignments (i.e. each entry in range [1..K])
#
import numpy as np
import matplotlib.pyplot as plt

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from plotProgresskMeans import plotProgresskMeans

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # Plot the data if we are plotting progress
    if plot_progress:
        plt.figure(figsize=(8,6))

    # Initialize values
    m = X.shape[0]
    n = X.shape[1]
    K = initial_centroids.shape[0]

    centroids = initial_centroids
    previous_centroids = centroids
    #idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print('K-Means iteration %d/%d...' % (i, max_iters) )

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    if plot_progress:
        plt.show()

    idx = findClosestCentroids(X, centroids)
    
    return centroids, idx
