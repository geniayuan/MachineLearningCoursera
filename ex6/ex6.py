## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm

from plotData import plotData
from visualizeBoundaryLinear import visualizeBoundaryLinear
from visualizeBoundary import visualizeBoundary
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params
## Initialization
#clear ; close all; clc

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.

print('Loading and Visualizing Data ...')

# Load from ex6data1:
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X = mat['X']
y = mat['y']

# Plot training data
plt.figure(figsize = (8,6))
plotData(X, y)
plt.show()

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.

# Load from ex6data1:
# You will have X, y in your environment
#load('ex6data1.mat');

print('\nTraining Linear SVM ......')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
myC = 1
linear_svm = svm.SVC(C=myC, kernel='linear', tol=1e-3)#max_iter=20
linear_svm.fit(X, y.flatten())

plt.figure(figsize = (8,6))
plotData(X, y)
visualizeBoundaryLinear(X, y, linear_svm)
plt.show()


## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ......')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma);

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' \
         'is %f, this value should be about 0.324652)' % (sigma, sim) )

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#
print('\nLoading and Visualizing Data ......')

# Load from ex6data2:
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data2.mat')
X = mat['X']
y = mat['y']

# Plot training data
#plt.figure(figsize = (8,6))
#plotData(X, y)
#plt.show()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ......');

# SVM Parameters
myC = 1
sigma = 0.1
gamma = 1.0/(2.0*sigma**2)

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
gaussian_svm = svm.SVC(C=myC, kernel='rbf', gamma = gamma)
gaussian_svm.fit(X, y.flatten())

plt.figure(figsize = (8,6))
plotData(X,y)
visualizeBoundary(X, y, gaussian_svm)
plt.show()


## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

print('Loading and Visualizing Data Set 3......')

# Load from ex6data3:
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data3.mat')
X = mat['X']
y = mat['y']
Xval = mat['Xval']
yval = mat['yval']

# Plot training data
#plt.figure(figsize = (8,6))
#plotData(X, y)
#plt.show()

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.

# Try different SVM Parameters here
myC, mysigma = dataset3Params(X, y, Xval, yval)
gamma = 1.0/(2.0*mysigma**2)

# Train the SVM use the best parameters
best_gaussian_model = svm.SVC(C=myC, kernel='rbf', gamma = gamma)
best_gaussian_model.fit(X, y.flatten())

plt.figure(figsize = (8,6))
plotData(X, y)
visualizeBoundary(X, y, best_gaussian_model);
plt.show()
