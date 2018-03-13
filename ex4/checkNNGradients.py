#CHECKNNGRADIENTS Creates a small neural network to check the
#backpropagation gradients
#   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
#   backpropagation gradients, it will output the analytical gradients
#   produced by your backprop code and the numerical gradients (computed
#   using computeNumericalGradient). These two gradient computations should
#   result in very similar values.
#
import numpy as np

from flatAndReshape import flatTheta, reshapeTheta
from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnCostFunction, nnCostFunctionGrad
from computeNumericalGradient import computeNumericalGradient

def checkNNGradients(lambd=0):
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;

    # We generate some 'random' test data
    Theta1 = np.ones( (hidden_layer_size, input_layer_size+1) ) #5*4
    Theta2 = np.ones( (num_labels, hidden_layer_size+1) )#3*6
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1) #m=5*3
    y  = np.mod(range(1, m+1), num_labels)

    # Unroll parameters
    nn_params = flatTheta([Theta1,Theta2])

    # back propogate gradient
    grad = nnCostFunctionGrad(  nn_params, \
                                input_layer_size, \
                                hidden_layer_size,\
                                num_labels, \
                                X, y, lambd)
    # Short hand for cost function
    costFunc = lambda p: nnCostFunction(    p, \
                                            input_layer_size, \
                                            hidden_layer_size,\
                                            num_labels, \
                                            X, y, lambd)

    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print( np.column_stack((numgrad, grad)) )
    print('The above two columns you get should be very similar.\n' \
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad);

    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             'Relative Difference: %g', diff)
