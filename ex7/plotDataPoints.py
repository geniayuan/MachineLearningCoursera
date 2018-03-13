#PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
#index assignments in idx have the same color
#   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
#   with the same index assignments in idx have the same color
import numpy as np
import matplotlib.pyplot as plt

def plotDataPoints(X, idx, K):
    # Create palette
    #palette = hsv(K + 1);
    #colors = palette(idx, :)
    circleSize = 10
    cmap = plt.get_cmap("jet")
    circleColor = cmap(idx / K)

    plt.scatter(X[:,0], X[:,1], s=circleSize, c=circleColor)
