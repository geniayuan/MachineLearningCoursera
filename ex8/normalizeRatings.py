#NORMALIZERATINGS Preprocess data by subtracting mean rating for every
#movie (every row)
#   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
#   has a rating of 0 on average, and returns the mean rating in Ymean.
#
import numpy as np

def normalizeRatings(Y, R):
    #Ymean = np.sum(Y, axis = 1)/np.sum(R, axis=1)
    #Ymean = Ymean.reshape(Y.shape[0],1)
    #Ynorm = Y - Ymean
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.nonzero(R[i,:])[0]
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean
