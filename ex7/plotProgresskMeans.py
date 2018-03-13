#PLOTPROGRESSKMEANS is a helper function that displays the progress of
#k-Means as it is running. It is intended for use only with 2D data.
#   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
#   points with colors assigned to each centroid. With the previous
#   centroids, it also plots a line between the previous locations and
#   current locations of the centroids.
#
import numpy as np
import matplotlib.pyplot as plt

from plotDataPoints import plotDataPoints

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:,0], centroids[:,1], 'x', \
                            markerEdgeColor = 'k', \
                            markerSize = 6, lineWidth = 6)

    # Plot the history of the centroids with lines
    for j in range(K):
        plt.plot([centroids[j,0], previous[j,0]],\
                 [centroids[j,1], previous[j,1]], 'k-', lineWidth = 1)

    #plt.title('Iteration number %d' % i)
    #show()
