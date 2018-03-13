# Machine Learning Online Class - Exercise 1: Linear Regression

# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

from computeCost import computeCost
from gradientDescent import gradientDescent

# ==================== Part 1: Basic Function ====================
warmup = np.eye(5)
print(warmup)

# ======================= Part 2: Plotting =======================
#filename = input("please enter the file name: ")
filename = "ex1data1.txt"

data = np.loadtxt(filename, delimiter = ',')
m = data.shape[0] # number of training examples

x = data[:,0]
y = data[:,1]

# Plot the Data Points
print("...... plotting data ......")

plt.figure(1, figsize=(8,6))
plt.plot(x, y, 'rx', markersize=10,label='Training Data')
plt.grid(True) #Always plot.grid true!
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')

# =================== Part 3: Cost and Gradient descent ===================
print('...... Testing the cost function ......')
X = np.vstack(zip(np.ones(m), x))
y = y.reshape(m,1)

theta = np.zeros((2,1))
J = computeCost(X, y, theta)
print('With theta = [0 ; 0], Cost computed = ', J)
print('Expected cost value (approx) 32.07')

# further testing of the cost function
theta = np.array([[-1],[2]])
J = computeCost(X, y, theta)
print('With theta = [-1; 2], Cost computed = ', J)
print('Expected cost value (approx) 54.24')

# Some gradient descent settings
print('...... Running the Gradient Descent ......')
iterations = 1500;
alpha = 0.01;
theta = np.zeros((2,1))

theta, J_values = gradientDescent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:', theta)
print('Expected theta values (approx): [-3.6303; 1.1664]')

# Plot the linear fit
plt.figure(1)
plt.plot(X[:,1], np.dot(X,theta), 'b-', label='Hypothesis')

# Predict values for population sizes of 35,000 and 70,000
predict1 = float(np.dot(np.array([1, 3.5]),theta))
print('For population = 35,000, we predict a profit:', predict1*10000)
predict2 = float(np.dot(np.array([1, 7]),theta))
print('For population = 70,000, we predict a profit:', predict2*10000)

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('...... Visualizing J(theta_0, theta_1) ......')

theta0 = np.linspace(-10, 10, 100);
theta1 = np.linspace(-1, 4, 100);

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0), len(theta1)));

# Fill out J_vals
for i in range(len(theta0)):
    for j in range(len(theta1)):
        t = np.array([ [theta0[i]], [theta1[j]] ])
        J_vals[i,j] = computeCost(X, y, t)

theta0, theta1 = np.meshgrid(theta0, theta1)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = np.transpose(J_vals);

# Surface plot
fig = plt.figure(2, figsize=(8,6))
ax = fig.gca(projection='3d')
ax.plot_surface(theta0, theta1, J_vals, alpha=0.8, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')

# Contour plot
plt.figure(3, figsize=(8,6))
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0, theta1, J_vals, np.logspace(-2, 3, 20))
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)

plt.show()
