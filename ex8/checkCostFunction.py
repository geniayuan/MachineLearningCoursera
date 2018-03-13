#CHECKCOSTFUNCTION Creates a collaborative filering problem
#to check your cost function and gradients
#   CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem
#   to check your cost function and gradients, it will output the
#   analytical gradients produced by your code and the numerical gradients
#   (computed using computeNumericalGradient). These two gradient
#   computations should result in very similar values.
import numpy as np
from cofiCostFunc import cofiCostFunc, cofiCostGrad

def checkCostFunction(params, Y, R, \
                        num_users, num_movies, num_features, mylambda = 0):

    Grad = cofiCostGrad(params, Y, R, num_users, \
                            num_movies, num_features, mylambda)

    e = 1e-4
    m = params.size
    rand_indices = np.random.permutation(m)
    #print(rand_indices)
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    for i in range(10):
        p = rand_indices[i]
        perturb = np.zeros(params.shape)
        perturb[p] = e

        loss1 = cofiCostFunc(params - perturb, Y, R, \
                                num_users, num_movies, num_features, mylambda)

        loss2 = cofiCostFunc(params + perturb, Y, R, \
                                num_users, num_movies, num_features, mylambda)

        # Compute Numerical Gradient
        numgrad = (loss2 - loss1) / (2*e)
        grad = Grad[p]

        diff = 0.0
        if numgrad+grad > e:
            diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

        print('%0.15f \t %0.15f \t %0.8e' % (numgrad, grad, diff) )
