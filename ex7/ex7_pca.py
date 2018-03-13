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

from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData

from displayData import displayData
## Initialization
#clear ; close all; clc

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('\nVisualizing example dataset for PCA....');

#  The following command loads the dataset. You should now have the
#  variable X in your environment
mat = scipy.io.loadmat('ex7data1.mat')
X = mat['X']

#  Visualize the example dataset
plt.plot(X[:,0], X[:,1], 'bo', fillstyle ='none')
plt.axis([0.5, 6.5, 2, 8])
plt.axis('equal')
plt.grid(True)
#plt.show()


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('\nRunning PCA on example dataset...')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)
#print("mu:", mu)
#print("sigma:", sigma)
#print("X_norm:", X_norm)
#  Run PCA
U, S, Vh = pca(X_norm)
#print('U:', U)
#print('S:', S)
#print('Vh:', Vh)
#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
plt.plot([mu[0], mu[0]+1.5*S[0]*U[0,0]],\
         [mu[1], mu[1]+1.5*S[0]*U[1,0]], 'k-', lineWidth = 2)

plt.plot([mu[0], mu[0]+1.5*S[1]*U[0,1]],\
         [mu[1], mu[1]+1.5*S[1]*U[1,1]], 'k-', lineWidth = 2)

plt.show()

#drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2)
#drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2)
print('Top eigenvector:')
print(' U(:,0) =', U[:,0])
print('(you should expect to see -0.707107 -0.707107)')

## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#  first k eigenvectors. The code will then plot the data in this reduced
#  dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('\nDimension reduction on example dataset...')

#  Plot the normalized dataset (returned from pca)
plt.plot(X_norm[:,0], X_norm[:,1], 'bo', fillstyle ='none')
plt.axis([-4, 3, -4, 3])
plt.axis('equal')

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f' % Z[0] )
print('(this value should be about 1.481274)')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: %f %f' % (X_rec[0,0], X_rec[0,1]) )
print('(this value should be about  -1.047419 -1.047419)')

#  Draw lines connecting the projected points to the original points
plt.plot(X_rec[:,0], X_rec[:,1], 'ro', fillstyle ='none')
for i in range(X_norm.shape[0]):
    start = X_norm[i,:]
    end = X_rec[i,:]
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k--', lineWidth = 1)

plt.show()


## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('\nLoading face dataset....')
#  Load Face dataset
mat = scipy.io.loadmat('ex7faces.mat')
X = mat['X']

#  Display the first 100 faces in the dataset
displayData( X[:100, :])
plt.title('Original first 100 faces')
plt.show()


## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('\nRunning PCA on face dataset, this might take a minute or two ...')

#  Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

# Display normalized data
#plt.subplot(1, 2, 1)
displayData( X_norm[:100,:] )
plt.title('Normalized first 100 faces')
plt.show()

#  Run PCA
U, S, Vh = pca(X_norm)
#  Visualize the top 36 eigenvectors found
displayData(U[:, :36].T)
plt.title('Top 36 eigenvectors')
plt.show()

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
print('\nDimension reduction for face dataset...')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of:', Z.shape)


## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces....')

X_rec  = recoverData(Z, U, K)

# Display reconstructed data from only k eigenfaces
displayData( X_rec[:100, :] )
plt.title('Recovered first 100 faces')
plt.show()

'''
## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

close all; close all; clc

# Reload the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = double(imread('bird_small.png'));

# If imread does not work for you, you can try instead
#   load ('bird_small.mat');

A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16;
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = floor(rand(1000, 1) * size(X, 1)) + 1;

#  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

#  Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
print('Program paused. Press enter to continue.\n');
pause;

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);

# PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

# Plot in 2D
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
print('Program paused. Press enter to continue.\n');
pause;
'''
