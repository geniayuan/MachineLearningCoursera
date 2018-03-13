#VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision
#   boundary learned by the SVM and overlays the data on it
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def visualizeBoundary(X, y, model):
    x1p = np.linspace(min(X[:,0]), max(X[:,0]), 100) #s1
    x2p = np.linspace(min(X[:,1]), max(X[:,1]), 100) #s2
    x1, x2 = np.meshgrid( x1p, x2p ) # s2*s1

    vals = np.zeros(x1.shape)

    for i in range(x1.shape[1]):
        this_x = np.column_stack((x1[:, i], x2[:, i]))
        vals[:, i] = model.predict(this_x)

    # Plot the SVM boundary
    #contour(x1, x2, vals, [0 0], 'Color', 'b')
    plt.contour(x1, x2, vals)
