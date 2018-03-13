#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
#   positive examples and o for the negative examples. X is assumed to be
#   a either
#   1) Mx3 matrix, where the first column is an all-ones column for the
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

# Plot Data
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import axes3d, Axes3D
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y):
    #plotData(X[:,1:], y)

    n = X.shape[1]
    if n <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([ np.min(X[:,1]),  np.max(X[:,2]) ])
        # Calculate the decision boundary line
        plot_y = (-1./theta[2]) * (theta[1]*plot_x + theta[0])
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y,'b-',linewidth=2,label='Decision Boundary')

        # Legend, specific for the exercise
        #legend('Admitted', 'Not admitted', 'Decision Boundary')
        #axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                Xnew = mapFeature( np.array([u[i]]), np.array([v[j]]) )
                z[i,j] = np.dot( theta,  Xnew.T);

        z = z.transpose() # important to transpose z before calling contour
        uu,vv = np.meshgrid(u,v)
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(uu, vv, z, [0], linewidth=2)
