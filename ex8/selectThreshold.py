#SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
#outliers
#   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
#   threshold to use for selecting outliers based on the results from a
#   validation set (pval) and the ground truth (yval).
#
import numpy as np

def selectThreshold(yval, pval):
    yval = yval.reshape(pval.shape)
    bestEpsilon = -1
    bestF1 = -1

    stepsize = ( np.max(pval) - np.min(pval) ) /1000.0

    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        index = np.nonzero(pval < epsilon)
        preditAnomalyNum = len( pval[index] )
        trueAnomalyNum = np.sum(yval == 1)
        #goodPredicNum = np.sum(pval == yval)
        tp = np.sum(yval[index] == 1)

        #print("\nepsilon:", epsilon)
        #print("preditAnomalyNum:", preditAnomalyNum)
        #print("trueAnomalyNum:", trueAnomalyNum)
        #print("tp:", tp)

        prec = 0
        if preditAnomalyNum > 0:
            prec = tp/preditAnomalyNum

        rec = 0
        if trueAnomalyNum > 0:
            rec = tp/trueAnomalyNum

        #fp = preditAnomalyNum - tp
        #fn = trueAnomalyNum - tp
        #prec = tp/(tp + fp)
        #rec = tp/(tp + fn)

        F1 = 0
        if (prec + rec) > 0:
            F1 = 2 * prec * rec / (prec + rec)

        if F1 > bestF1:
            #print("best F1:", F1)
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions
