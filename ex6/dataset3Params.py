#DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
#where you select the optimal (C, sigma) learning parameters to use for SVM
#with RBF kernel
#   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
#   sigma. You should complete this function to return the optimal C and
#   sigma based on a cross-validation set.
#
import numpy as np
from sklearn import svm

def dataset3Params(X, y, Xval, yval):
    testData = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    bestScore = 0

    for cvalue in testData:
        for sigmavalue in testData:
            gamma = 1.0/(2.0*sigmavalue**2)

            # We set the tolerance and max_passes lower here so that the code will run
            # faster. However, in practice, you will want to run the training to
            # convergence.
            gaussian_svm = svm.SVC(C=cvalue, kernel='rbf', gamma = gamma)
            gaussian_svm.fit(X, y.flatten())

            thisScore = gaussian_svm.score(Xval, yval.flatten())

            if thisScore > bestScore:
                bestScore = thisScore
                C = cvalue
                sigma = sigmavalue
                print("best:", thisScore, "C:", cvalue, "sigma:", sigmavalue)

    return C, sigma




    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example,
    #                   predictions = svmPredict(model, Xval);
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using
    #        mean(double(predictions ~= yval))
    #
