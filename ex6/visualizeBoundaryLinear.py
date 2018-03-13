#VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
#SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
#   learned by the SVM and overlays the data on it

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def visualizeBoundaryLinear(X, y, linearModel):
    x1p = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    x2p = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    zvals = np.zeros( (x1p.size, x2p.size) )
    for i in range(x1p.size):
        for j in range(x2p.size):
            this_x = np.array( [x1p[i],x2p[j]] )
            zvals[i][j] = float(linearModel.predict( this_x.reshape(1,-1) ))

    xx1, xx2 = np.meshgrid( x1p, x2p )

    plt.contour( xx1, xx2, zvals.T)
