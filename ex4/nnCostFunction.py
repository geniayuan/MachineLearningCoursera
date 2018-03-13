#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#   X, y, lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices.
#
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#
import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from flatAndReshape import flatTheta, reshapeTheta

def nnCostFunction( nn_params, \
                    input_layer_size, \
                    hidden_layer_size, \
                    num_labels, \
                    X, y, lambd):
    #print("nnCostFunction eval")
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    myThetaList = reshapeTheta(nn_params, input_layer_size, hidden_layer_size, num_labels)
    Theta1 = myThetaList[0] # 25*401
    Theta2 = myThetaList[1] # 10*26

    # Setup some useful variables
    m = X.shape[0]
    X = np.insert(X, 0, 1, axis=1) # 5000*401

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    # propogate forward
    h1 = sigmoid(X.dot(Theta1.T)) #5000*25
    h1 = np.insert(h1, 0, 1, axis = 1) #5000*26
    h2 = sigmoid(h1.dot(Theta2.T)) #5000*10

    myCost = 0
    for example in range(m):
        hrow = h2[example,:].reshape(num_labels,1)

        logic_y = np.zeros((num_labels,1))
        logic_y[y[example]-1] = 1

        p1 = - np.dot(logic_y.T, np.log(hrow))
        p2 = - np.dot(1-logic_y.T, np.log(1-hrow))
        myCost += float(p1+p2)

    #regularization terms
    Theta1temp = Theta1
    Theta1temp[:,0] = 0
    Theta2temp = Theta2
    Theta2temp[:,0] = 0
    ThetaSum = np.sum(Theta1temp*Theta1temp) + np.sum(Theta2temp*Theta2temp)
    reg = lambd/2 * ThetaSum
    print( "current value:", (myCost + reg)/m )

    return (myCost + reg)/m



def nnCostFunctionGrad( nn_params, \
                        input_layer_size, \
                        hidden_layer_size, \
                        num_labels, \
                        X, y, lambd):
    #print("nnCostFunctionGrad eval")
    thetaSize = nn_params.size
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    myThetaList = reshapeTheta(nn_params, input_layer_size, hidden_layer_size, num_labels)
    Theta1 = myThetaList[0] # 25*401
    Theta2 = myThetaList[1] # 10*26

    # Setup some useful variables
    m = X.shape[0]
    X = np.insert(X, 0, 1, axis=1) # 5000*401

    Delta2 = np.zeros((num_labels, hidden_layer_size+1))
    Delta1 = np.zeros((hidden_layer_size, input_layer_size+1))

    for example in range(m):
        a1 = X[example, :].reshape(1,input_layer_size+1) #1*401
        # Part 1: Feedforward the neural network and return the cost in the
        #         variable J. After implementing Part 1, you can verify that your
        #         cost function computation is correct by verifying the cost
        #         computed in ex4.m
        z2 = a1.dot(Theta1.T) #1*25
        a2 = sigmoid(z2)
        a2 = np.insert(a2, 0, 1, axis = 1) #1*26

        z3 = a2.dot(Theta2.T) #1*10
        a3 = sigmoid(z3) #10

        # Part 2: Implement the backpropagation algorithm to compute the gradients
        #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
        #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
        #         Theta2_grad, respectively. After implementing Part 2, you can check
        #         that your implementation is correct by running checkNNGradients
        #
        # Note: The vector y passed into the function is a vector of labels
        #       containing values from 1..K. You need to map this vector into a
        #       binary vector of 1's and 0's to be used with the neural network
        #       cost function.
        logic_y = np.zeros((1,num_labels)) #1*10
        logic_y[0, y[example]-1] = 1

        delta3 = a3 - logic_y #1*10
        Delta2 += delta3.T.dot(a2) # 10*26

        delta2 = delta3.dot(Theta2[:,1:]) * sigmoidGradient(z2) #1*25
        Delta1 += delta2.T.dot(a1) # 25*401
    #
    # Part 3: Implement regularization with the cost function and gradients.
    Theta1temp = Theta1
    Theta1temp[:,0] = 0
    Theta2temp = Theta2
    Theta2temp[:,0] = 0

    D1 = Delta1/m + lambd/m * Theta1temp
    D2 = Delta2/m + lambd/m * Theta2temp

    # Unroll gradients
    grad = flatTheta([D1,D2]) #size 10285

    return grad
