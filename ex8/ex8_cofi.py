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
from scipy import optimize

from cofiCostFunc import cofiCostFunc, cofiCostGrad
from checkCostFunction import checkCostFunction
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings
## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#
print('\nLoading movie ratings dataset...')

#  Load data
mat = scipy.io.loadmat('ex8_movies.mat')
Y = mat['Y']
R = mat['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i
col = np.nonzero(R[0,:])[0]
#  From the matrix, we can compute statistics like average rating.
print('\nAverage rating for movie 1 (Toy Story): %f / 5' % np.mean(Y[0, col]) )

#  We can "visualize" the ratings matrix by plotting it with imagesc
#imagesc(Y);
plt.imshow(Y, extent=(0, Y.shape[1], 0, Y.shape[0]), aspect='equal', origin='upper')
plt.colorbar()
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in
#  cofiCostFunc.m to return J.
print('\nLoading pre-trained weights dataset...')

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
param_mat = scipy.io.loadmat('ex8_movieParams.mat')
X = param_mat['X']
Theta = param_mat['Theta']
num_users = param_mat['num_users']
num_movies = param_mat['num_movies']
num_features = param_mat['num_features']

print('\nChecking cost function value w/o regularization... ')
#  Reduce the data set size so that this runs faster
num_users_t = 4
num_movies_t = 5
num_features_t = 3
X_t = X[0:num_movies_t, 0:num_features_t]
Theta_t = Theta[0:num_users_t, 0:num_features_t]
Y_t = Y[0:num_movies_t, 0:num_users_t]
R_t = R[0:num_movies_t, 0:num_users_t]

myparam = np.hstack((X_t.flatten(), Theta_t.flatten()))

#  Evaluate cost function
J = cofiCostFunc(myparam, Y_t, R_t, num_users_t, \
                    num_movies_t, num_features_t, mylambda=0)

print('Cost at loaded parameters: %f (this value should be about 22.22)' % J)

## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement
#  the collaborative filtering gradient function. Specifically, you should
#  complete the code in cofiCostFunc.m to return the grad argument.
#
print('\nChecking Gradients (without regularization) ...')

#  Check gradients by running checkNNGradients
checkCostFunction(myparam, Y_t, R_t, num_users_t, \
                        num_movies_t, num_features_t, mylambda=0)

#Grad = cofiCostGrad(myparam, Y_t, R_t, num_users_t, \
#                    num_movies_t, num_features_t, mylambda=0)

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#
print('\nChecking function value with Regularization... ')
#  Evaluate cost function
J = cofiCostFunc(myparam, Y_t, R_t, num_users_t, \
                        num_movies_t, num_features_t, mylambda=1.5)

print('Cost at loaded parameters (lambda = 1.5): %f'  \
            ' (this value should be about 31.34)' % J)

## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement
#  regularization for the gradient.
#
print('\nChecking Gradients (with regularization) ... ')

#  Check gradients by running checkNNGradients
checkCostFunction(myparam, Y_t, R_t, num_users_t, \
                        num_movies_t, num_features_t, mylambda=1.5)

#Grad = cofiCostGrad(myparam, Y_t, R_t, num_users_t, \
#                        num_movies_t, num_features_t, mylambda=1.5)

## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
movieList = loadMovieList()
movieNum = len(movieList)
#  Initialize my ratings
my_ratings = np.zeros((movieNum,1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4
# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2
# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\nNew user ratings:')
for i in range(movieNum):
    if my_ratings[i] > 0:
        print('Rated %d/5 for %s' % (my_ratings[i], movieList[i]) )

## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users
#
print('\nTraining collaborative filtering...')
#  Load data (already done)
# load('ex8_movies.mat');
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings>0, R))

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.hstack( (X.flatten(), Theta.flatten()) )

# Set options
myoption={'gtol': 1e-5, 'disp': True, 'maxiter':200}
# Set Regularization
mylambda = 10

thisResult = optimize.minimize( cofiCostFunc, \
                                initial_parameters, \
                                method = 'CG', \
                                jac = cofiCostGrad, \
                                args = (Ynorm, R, \
                                        num_users, num_movies, \
                                        num_features, mylambda), \
                                options = myoption)

# Unfold the returned theta back into U and W
X = thisResult.x[0:num_movies * num_features]
X = X.reshape(num_movies, num_features)

Theta = thisResult.x[num_movies * num_features:]
Theta = Theta.reshape(num_users, num_features)

print('Recommender system learning completed.')

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#
p = X.dot(Theta.T)
my_predictions = p[:,0] + Ymean.flatten()
# load movieList name, already done
# movieList = loadMovieList();

ix = np.argsort(my_predictions)
# descending order
ix[:] = ix[::-1]

print('\nTop recommendations for you:')
for i in range(20):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % \
                (my_predictions[j], movieList[j]) )

print('\nOriginal ratings provided:')
for i in range(movieNum):
    if my_ratings[i] > 0:
        print('Rated %d/5 for %s' % \
                    (my_ratings[i], movieList[i]) )
print('\n')
