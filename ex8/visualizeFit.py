#VISUALIZEFIT Visualize the dataset and its estimated distribution.
#   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
#   probability density function of the Gaussian distribution. Each example
#   has a location (x1, x2) that depends on its feature values.
#
import numpy as np
import matplotlib.pyplot as plt
from math import isinf

from multivariateGaussian import multivariateGaussian

def visualizeFit(X, mu, sigma2):

    x1 = np.linspace(0, 35, 71)
    x2 = np.linspace(0, 35, 71)
    X1, X2 = np.meshgrid(x1, x2)

    Xplot = np.column_stack((X1.flatten(), X2.flatten()))

    Z = multivariateGaussian(Xplot, mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:,0], X[:,1], 'b+', markersize = 4, linewidth = 1)

    # Do not plot if there are infinities
    if not isinf( np.sum(Z) ):
        plt.contour( X1, X2, Z, 10.0 ** np.arange(-20, 0, 3).T )
