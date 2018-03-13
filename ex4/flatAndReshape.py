import numpy as np

def flatTheta(myThetaList):
    flat = np.array([])
    for theta in myThetaList:
        flat = np.append(flat, theta.flatten())

    return flat


def reshapeTheta(myFlatTheta, input_layer_size, hidden_layer_size, num_labels):
    sz = myFlatTheta.size
    szWant = hidden_layer_size*(input_layer_size+1)\
                + num_labels * (hidden_layer_size+1)
    if(sz != szWant):
        print("myFlatTheta size is wrong")
        exit()

    Theta1 = myFlatTheta[0:hidden_layer_size*(input_layer_size+1)]
    Theta1 = Theta1.reshape(hidden_layer_size, input_layer_size+1)
    #print("Theta1 shape", Theta1.shape)

    Theta2 = myFlatTheta[hidden_layer_size*(input_layer_size+1):]
    Theta2 = Theta2.reshape(num_labels,hidden_layer_size+1)
    #print("Theta2 shape", Theta2.shape)

    myThetaList=[Theta1, Theta2]
    return myThetaList
