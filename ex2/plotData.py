#PLOTDATA Plots the data points X and y into a new figure
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.

# Create New Figure
# figure; hold on;

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the positive and negative examples on a
#               2D plot, using the option 'k+' for the positive
#               examples and 'ko' for the negative examples.
#
import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    pos = np.nonzero(y==1)
    neg = np.nonzero(y==0)

    plt.figure(figsize=(8,6))
    plt.plot(X[pos,0], X[pos,1], 'k+', markersize=8)
    plt.plot(X[neg,0], X[neg,1], 'ko', markerfacecolor='y', markersize=6)
    plt.grid(True) #Always plot.grid true!
    #plt.legend(handles=[dot1, dot2])
