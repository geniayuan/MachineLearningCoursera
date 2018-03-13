## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids

from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
#from recoverData import recoverData
from plotDataPoints import plotDataPoints
## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

#close all; close all; clc

# Reload the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = scipy.misc.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat');

A = A / 255
img_size = A.shape

X = A.reshape(img_size[0] * img_size[1], 3)
print("X size:", X.shape)
K = 16
max_iters = 10

initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, \
                            max_iters=max_iters, plot_progress=False)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
rand_indices = np.random.permutation(X.shape[0])
sel = rand_indices[:1000]
#sel = floor(rand(1000, 1) * size(X, 1)) + 1;

#  Setup Color Palette
#palette = hsv(K);
#colors = palette(idx(sel), :);

#  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap("jet")
circleColor = cmap(idx[sel] / K)
ax.scatter(X[sel,0], X[sel,1], X[sel,2], c=circleColor)

#scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()


## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S, Vh = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plotDataPoints(Z[sel, :], idx[sel], K)
#plotDataPoints(Z, idx, K)
plt.title('Pixel dataset plotted in 2D,\n using PCA for dimensionality reduction')
plt.show()
