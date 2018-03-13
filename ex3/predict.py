#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)
import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    # Useful values
    m = X.shape[0] # 5000
    n = X.shape[1] # 400
    num_layers = Theta1.shape[0] #25 #25*401
    num_labels = Theta2.shape[0] #10 #10*26

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1 , axis = 1) #5000*401

    h1 = sigmoid(X.dot(Theta1.T)) #5000*25
    h1 = np.insert(h1, 0, 1, axis = 1) #5000*26

    h2 = sigmoid(h1.dot(Theta2.T)) #5000*10

    # the ith col index = i-1 represent y = i
    p = np.argmax(h2, axis=1) + 1
    # You need to return the following variables correctly
    return p.reshape(m,1)

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#
