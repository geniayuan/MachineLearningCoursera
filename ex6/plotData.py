#PLOTDATA Plots the data points X and y into a new figure
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.
#
# Note: This was slightly modified such that it expects y = 1 or y = 0
import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    # Find Indices of Positive and Negative Examples
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+', lineWidth=1, markerSize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerFaceColor='y', markerSize=7)
    plt.grid(True)
    #plt.show()
