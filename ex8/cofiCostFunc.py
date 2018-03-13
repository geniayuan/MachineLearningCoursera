#COFICOSTFUNC Collaborative filtering cost function
#   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
#   num_features, lambda) returns the cost and gradient for the
#   collaborative filtering problem.
#
import numpy as np

def cofiCostFunc(params, Y, R, \
                    num_users, num_movies, num_features, mylambda = 0):
    # Unfold the U and W matrices from params
    X = params[0:num_movies * num_features]
    X = X.reshape(num_movies, num_features)

    Theta = params[num_movies * num_features:]
    Theta = Theta.reshape(num_users, num_features)

    temp = (X.dot(Theta.T) - Y) * R
    # You need to return the following values correctly
    J = np.sum(temp**2)/2
    X_reg = mylambda/2 * np.sum(X**2)
    Theta_reg = mylambda/2 * np.sum(Theta**2)

    #print("current value: %15.5f" % (J + X_reg + Theta_reg) )
    return J + X_reg + Theta_reg


def cofiCostGrad(params, Y, R, \
                    num_users, num_movies, num_features, mylambda = 0):
    # Unfold the U and W matrices from params
    X = params[0:num_movies * num_features]
    X = X.reshape(num_movies, num_features)

    Theta = params[num_movies * num_features:]
    Theta = Theta.reshape(num_users, num_features)

    temp = (X.dot(Theta.T) - Y) * R
    # You need to return the following values correctly
    X_grad = temp.dot(Theta) + mylambda * X
    Theta_grad = temp.T.dot(X) + mylambda * Theta

    grad = np.hstack( (X_grad.flatten(), Theta_grad.flatten()) )

    return grad

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta
