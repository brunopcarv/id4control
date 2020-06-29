#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Kernelized Ridge Regression - System Identification

import numpy as np

# from sklearn.metrics.pairwise import rbf_kernel


# RBF kernel (Gaussian kernel)
def kernel_rbf_function_M(X, Y, var=1.0, gamma=1.0):
	import numexpr as ne
	X_norm = np.sum(X**2, axis=-1)
	return ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
	        'A' : X_norm[:,None],
	        'B' : X_norm[None,:],
	        'C' : np.dot(X, X.T),
	        'g' : gamma,
	        'v' : var
	})
def kernel_rbf_function_N(x, X, var=1.0, gamma=1.0):
	import numexpr as ne
	diff = X-x
	norm_squared = np.sum(diff**2, axis=-1)
	return ne.evaluate('v * exp(-g * A)', {
	        'A' : norm_squared,
	        'g' : gamma,
	        'v' : var
	})

# Laplace kernel TODO: update
def kernel_laplace_function_M(X, Y, var=1.0, gamma=1.0):
	import numexpr as ne
	X_norm = np.sum(X**2, axis=-1)
	return ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
	        'A' : X_norm[:,None],
	        'B' : X_norm[None,:],
	        'C' : np.dot(X, X.T),
	        'g' : gamma,
	        'v' : var
	})
def kernel_laplace_function_N(x, X, var=1.0, gamma=1.0):
	import numexpr as ne
	diff = X-x
	norm_squared = np.sum(diff**2, axis=-1)
	return ne.evaluate('v * exp(-g * A)', {
	        'A' : norm_squared,
	        'g' : gamma,
	        'v' : var
	})

# Linear kernel
def kernel_linear_function(x, y, a=1.0, b=0.0):
	import numexpr as ne
	return ne.evaluate('a * A + b', {
	        'A' : np.dot(x, y.T),
	        'b' : b,
	        'a' : a
	})

# Polynomial kernel
def kernel_poly_function(x, y, a=1.0, b=1.0, d=2):
	import numexpr as ne
	return ne.evaluate('(a * A + b)**d', {
	        'A' : np.dot(x, y.T),
	        'b' : b,
	        'a' : a,
	        'd' : d
	})

#Hiperbolic tangent kernel (conditionally positive definite)
def kernel_tanh_function(x, y, a=1.0, b=0.0):
	import numexpr as ne
	return ne.evaluate('tanh(a * A + b)', {
	        'A' : np.dot(x, y.T),
	        'b' : b,
	        'a' : a
	})

# Other possible kernels (the list is actually long):
# - Fourier kernel
# - Wavelet kernel
# - Spline kernel
# reference: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#:~:text=The%20Linear%20kernel%20is%20the,the%20same%20as%20standard%20PCA.



class KernelRidgeRegression():

	def __init__(self, lambda_reg):
		self.X = None
		self.Y = None
		self.M = None
		self.inverse = None
		self.lambda_reg = lambda_reg
		self.m = None
		self.n = None

	# def kernel_function(self, X, var=1.0, gamma=1.0):
	# 	return var * rbf_kernel(X, gamma = gamma)

	def training(self, X, Y, kernel_function):
		self.X = X
		self.Y = Y
		self.m, self.n = np.shape(X)
		self.M = kernel_function(X,X)
		self.inverse = np.linalg.inv(self.M + self.lambda_reg*np.identity(self.m, dtype = float))
		return self.M

	def predict(self, x, kernel_function):
		N = kernel_function(x, self.X)
		temp = np.dot(N.T, self.inverse)
		hat_y = np.dot(temp, self.Y)
		return hat_y.T