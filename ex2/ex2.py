## Machine Learning Online Class - Exercise 2: Logistic Regression
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#from matplotlib import cm
#from mpl_toolkits.mplot3d import axes3d, Axes3D

from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from costFunction import costFunction, costFunctionGradient
from predict import predict

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

#filename = input("please enter the file name: ")
filename = "ex2data1.txt"

data = np.loadtxt(filename, delimiter = ',')
m = data.shape[0] # number of training examples
n = data.shape[1] - 1

x = data[:,0:n]
y = data[:,n]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(x, y)
plt.ylabel('Exam 1 Score')
plt.xlabel('Exam 2 Score')

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

# Add intercept term to x and X_test
X = np.insert(x, 0, 1, axis=1)
y = y.reshape(m,1)

# Initialize fitting parameters
initial_theta = np.zeros((n+1,1));

#zz = np.linspace(-10,10,500)
#plt.figure(2,figsize = (8,6))
#plt.plot(zz, sigmoid(zz), 'b-', linewidth = 3)
#plt.grid(True)
#plt.show()


# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = costFunctionGradient(initial_theta, X, y)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):', grad)
print('Expected gradients (approx): -0.1000, -12.0092, -11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24],[0.2],[0.2]])
cost = costFunction(test_theta, X, y)
grad = costFunctionGradient(test_theta, X, y)

print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:', grad)
print('Expected gradients (approx): 0.043, 2.566, 2.647')


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
#options = optimset('GradObj', 'on', 'MaxIter', 400);

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost
#[theta, cost] = ...
#	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
# theta, cost = optimize.fmin_cg(costFunction, initial_theta, fprime=costFunctionGradient, args = (X,y))
theta = optimize.fmin(costFunction, x0=initial_theta, maxiter=400, args = (X,y))
cost = costFunction(theta, X, y)
# Print theta to screen
print('Cost at theta found by fminunc: %f', cost);
print('Expected cost (approx): 0.203')
print('theta:', theta)
print('Expected theta (approx): -25.161, 0.206, 0.201')

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.show()

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
testvalue = np.array([1,45,85])
prob = sigmoid(np.dot(testvalue, theta))
print('For a student with scores 45 and 85, we predict an admission probability of is: ', prob)
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)
p = p.reshape(m,1)
pp = (p==y)

print('Train Accuracy:', np.average(pp) * 100)
print('Expected accuracy (approx): 89.0\n')
