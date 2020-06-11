#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Linear Ridge Regression - System Identification

import numpy as np


class LinearRidgeRegression():

	def __init__(self, lambda_reg):
		self.X = None
		self.Y = None
		self.w = None
		self.lambda_reg = lambda_reg
		self.m = None
		self.n = None

	# def kernel_function(self, X, var=1.0, gamma=1.0):
	# 	return var * rbf_kernel(X, gamma = gamma)

	def training(self, X, Y):
		self.X = X
		self.Y = Y
		self.m, self.n = np.shape(X)

		inverse = np.linalg.inv(np.dot(self.X.T,self.X) + self.lambda_reg*np.identity(self.n, dtype = float))
		self.w = inverse.dot(self.X.T).dot(self.Y)
		print(np.shape(self.w))
		return self.w

	def predict(self, x, kernel_function):
		hat_y = self.w.T*x
		return hat_y.T