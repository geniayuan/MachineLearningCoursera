## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold
## Initialization
#clear ; close all; clc

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easy to
#  visualize.
#
#  Our example case consists of 2 network server statistics across
#  several machines: the latency and throughput of each machine.
#  This exercise will help us find possibly faulty (or very fast) machines.
#
print('\nVisualizing example dataset for outlier detection...');

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = scipy.io.loadmat('ex8data1.mat')
X = mat['X']
Xval = mat['Xval']
yval = mat['yval']

#  Visualize the example dataset
plt.plot(X[:,0], X[:,1], 'b+', markersize = 4, linewidth = 1)
plt.axis([0, 30, 0, 30])
plt.grid(True)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

## ================== Part 2: Estimate the dataset statistics ===================
#  For this exercise, we assume a Gaussian distribution for the dataset.
#
#  We first estimate the parameters of our assumed Gaussian distribution,
#  then compute the probabilities for each of the points and then visualize
#  both the overall distribution and where each of the points falls in
#  terms of that distribution.
#
print('\nVisualizing Gaussian fit...')

#  Estimate my and sigma2
mu_m, sigma2_m = estimateGaussian(X, multi = True)
#  Visualize the fit
visualizeFit(X,  mu_m, sigma2_m)
plt.axis([0, 30, 0, 30])
plt.grid(True)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('consider covariance')
plt.show()

#  Estimate my and sigma2
mu, sigma2 = estimateGaussian(X, multi = False)
#  Visualize the fit
visualizeFit(X,  mu, sigma2)
plt.axis([0, 30, 0, 30])
plt.grid(True)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('no covariance')
#plt.show()

## ================== Part 3: Find Outliers ===================
#  Now you will find a good epsilon threshold using a cross-validation set
#  probabilities given the estimated Gaussian distribution
#
pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
print('\nBest epsilon found using cross-validation: %e' % epsilon)
print('you should see a value epsilon of about 8.99e-05')

print('Best F1 on Cross Validation Set:  %f' % F1)
print('you should see a Best F1 value of  0.875000')

#  Find the outliers in the training set and plot the
p = multivariateGaussian(X, mu, sigma2)
outliers = np.nonzero(p < epsilon)

#  Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', fillstyle ='none') #linewidth = 1, markersize=10)
plt.show()


## ================== Part 4: Multidimensional Outliers ===================
#  We will now use the code from the previous part and apply it to a
#  harder problem in which more features describe each datapoint and only
#  some features indicate whether a point is an outlier.
#
print('\nTrying with high dimenstional data...')
#  Loads the second dataset. You should now have the
#  variables X, Xval, yval in your environment
mat = scipy.io.loadmat('ex8data2.mat')
X = mat['X']
Xval = mat['Xval']
yval = mat['yval']

#  Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X, multi = False)
#  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: %e' % epsilon)
print('you should see a value epsilon of about 1.38e-18)')

print('Best F1 on Cross Validation Set:  %f' % F1)
print('you should see a Best F1 value of 0.615385)')

#  Training set
p = multivariateGaussian(X, mu, sigma2)
print('# Outliers found: %d' % np.sum(p < epsilon))
