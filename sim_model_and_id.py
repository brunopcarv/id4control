#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Plot example

import numpy as np
import matplotlib.pyplot as plt

from model_predator_prey import PredatorPreyModel
from model_linear import LinearModel

from id_kernel_ridge_regression import KernelRidgeRegression, \
kernel_rbf_function_M, kernel_rbf_function_N, kernel_linear_function, \
kernel_poly_function, kernel_tanh_function
from id_linear_ridge_regression import LinearRidgeRegression


# Simulation parameters
dt = 0.01
# a, b, c, d = 2.8, 1.0, 0.99, 0.5
a, b, c, d = 1.0, 1.0, 1.0, 1.0
x1o = 1.5
x2o = 1.5
final_time_unit = 2000
half = int(final_time_unit*0.5)

# Model
predator_prey_model = PredatorPreyModel(a, b, c, d, x1o, x2o, dt)
x1, x2, time = predator_prey_model.run(final_time_unit)


# # Linear Model
# A = np.array([[0.998, 0.2],[0.0, 0.999]])
# B = np.array([1, 0]).T
# xo = np.array([2, 2]).T
# linear_model = LinearModel(A, B, xo, dt)
# x, time = linear_model.run(final_time_unit)
# x1 = x[:,0]
# x2 = x[:,1]


X = np.array([x1[:half-1], x2[:half-1]]).T
Y = np.array([x1[1:half], x2[1:half]]).T
m, n = np.shape(X)
print(m)
print(n)
M = np.zeros((m, m))



# RBF kernel id
lambda_reg = 0.00001
regression = KernelRidgeRegression(lambda_reg)
regression.training(X, Y, kernel_rbf_function_M)

Y_ridge = np.array([x1[:half], x2[:half]]).T
for k in range(half,final_time_unit):
	y, N = regression.predict(Y_ridge[-1,:], kernel_rbf_function_N)
	Y_ridge = np.append(Y_ridge, [y], axis=0)


# # Linear kernel id
# lambda_reg = 0.00001
# regression = KernelRidgeRegression(lambda_reg)
# regression.training(X, Y, kernel_linear_function)

# Y_ridge = np.array([x1[:half], x2[:half]]).T
# for k in range(half,final_time_unit):
# 	y, N = regression.predict(Y_ridge[-1,:], kernel_linear_function)
# 	Y_ridge = np.append(Y_ridge, [y], axis=0)


# #Polynomial Kernel id
# lambda_reg = 0.00001
# regression = KernelRegression(lambda_reg)
# regression.training(X, Y, kernel_poly_function)

# Y_ridge = np.array([x1[:half], x2[:half]]).T
# for k in range(half,final_time_unit):
# 	y, N = regression.predict(Y_ridge[-1,:], kernel_poly_function)
# 	Y_ridge = np.append(Y_ridge, [y], axis=0)


# # Hiperbolic tangent id
# TODO: devud hiperbolic tangent code
# lambda_reg = 0.0001
# regression = KernelRidgeRegression(lambda_reg)
# regression.training(X, Y, kernel_tanh_function)

# Y_ridge = np.array([x1[:half], x2[:half]]).T
# for k in range(half,final_time_unit):
# 	y, N = regression.predict(Y_ridge[-1,:], kernel_tanh_function)
# 	Y_ridge = np.append(Y_ridge, [y], axis=0)


# # Linear Ridge Regression id
# # TODO: debud the linear ridge regression code
# lambda_reg = 0.00001
# regression = LinearRidgeRegression(lambda_reg)
# regression.training(X, Y)

# Y_ridge = np.array([x1[:half], x2[:half]]).T
# for k in range(half,final_time_unit):
# 	y, N = regression.predict(Y_ridge[-1,:], kernel_linear_function)
# 	Y_ridge = np.append(Y_ridge, [y], axis=0)



# Plot
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, x1, time, Y_ridge[:,0])
axs[0].set_xlim(0,final_time_unit)
axs[0].set_xlabel('Time units (k)')
axs[0].set_ylabel('Prey: x1 (actual) and x1 (ridge)')
axs[0].grid(True)

axs[1].plot(time, x2, time, Y_ridge[:,1])
axs[1].set_xlim(0,final_time_unit)
axs[1].set_xlabel('Time units (k)')
axs[1].set_ylabel('Pred: x2 (actual) and x2 (ridge)')
axs[1].grid(True)
# cxy, f = axs[1].cohere(x1, x2, 5, 1. / dt)

fig.tight_layout()
plt.show()