## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from displayData import displayData
from predict import predict
## Initialization
#clear ; close all; clc

## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex3data1.mat'); # training data stored in arrays X, y
X = mat['X'] # size 5000*400
y = mat['y'] # size 5000*1

m = X.shape[0]
n = X.shape[1]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]
#print(sel.shape)

displayData(sel);
plt.show()

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.
print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weightMat = scipy.io.loadmat('ex3weights.mat')
Theta1 = weightMat['Theta1']
Theta2 = weightMat['Theta2']

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
pred = predict(Theta1, Theta2, X)
pp = (pred==y)
print('Training Set Accuracy: %f', np.average(pp) * 100)

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
#rp = np.random.permutation(m)
wrongcol = np.nonzero(pred != y)[0]
wrongNum = len(wrongcol)
print("wrong predict number:", wrongNum)
rp = np.random.permutation(wrongNum)

for i in range(wrongNum):
    # Display
    print('Displaying Example Image')
    thisImage = X[wrongcol[rp[i]],:].reshape(1,n)
    thisPred = predict(Theta1, Theta2, thisImage)
    #print('Neural Network Prediction: %d (digit %d)', thisPred, thisPred%10)
    print('Neural Network Prediction: %d (true y %d)', thisPred, y[wrongcol[rp[i]]])
    displayData(thisImage)
    plt.show()

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
      break
