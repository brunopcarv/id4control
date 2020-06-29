#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Closed-loop system - Plant and Controller

import numpy as np

class ClosedLoopSystem():

	def __init__(self, plant, controller, xo, dt=0.01):
		self.plant = plant
		self.controller = controller
		self.x = xo
		self.xo = xo
		self.k = 0
		self.dt = dt # delta t: sample period

	def next_points(self):
		x_temp = self.x

		self.x = self.plant.dynamics(x_temp, self.controller.action(x_temp), self.dt)
		self.k += 1
		return self.x

	def	reset(self):
		self.x = self.xo
		self.k = 0

	def run(self, k):
		x_list = [self.x]

		for i in range(1, k):
			x_list = np.append(x_list, [self.x], axis=0)
			self.next_points()

		self.reset()
		return x_list, range(k)


if __name__ == "__main__":
	from model_plants import LinearPlant, LinearDCMotorPlant, \
	InvertedPendulum, PredatorPreyPlant

	# Scalar zero controller 
	class NoController():
		def __init__(self):
			pass
		def action(self, x):
			return 0

	controller = NoController()
	plant = PredatorPreyPlant(1.0, 1.0, 1.0, 1.0)

	xo = np.array([2.0, 1.0]).T
	closed_loop = ClosedLoopSystem(plant, controller, xo)

	final_time_unit = 2000
	half = int(final_time_unit*0.25)
	quarter = int(half*0.05)

	x, time = closed_loop.run(final_time_unit)
	x1 = x[:,0]
	x2 = x[:,1]

	X = np.array([x1[:half-1], x2[:half-1]]).T
	Y = np.array([x1[1:half], x2[1:half]]).T





	from id_kernel_ridge_regression import KernelRidgeRegression, \
	kernel_rbf_function_M, kernel_rbf_function_N, kernel_linear_function, \
	kernel_poly_function, kernel_tanh_function
	from id_linear_ridge_regression import LinearRidgeRegression

	# RBF kernel id
	lambda_reg = 0.00001
	regression = KernelRidgeRegression(lambda_reg)
	# regression.training(X[:quarter,:], Y[:quarter,:], kernel_rbf_function_M)
	regression.training(X[:half,:], Y[:half,:], kernel_rbf_function_M)


	Y_ridge = np.array([x1[:half], x2[:half]]).T
	for k in range(half,final_time_unit):
		y, N = regression.predict(Y_ridge[-1,:], kernel_rbf_function_N)
		Y_ridge = np.append(Y_ridge, [y], axis=0)


	# RBF kernel id
	random_ids = np.random.choice(half-1, size=quarter, replace=False)
	regression_random = KernelRidgeRegression(lambda_reg)
	regression_random.training(X[random_ids,:], Y[random_ids,:], kernel_rbf_function_M)

	Y_ridge_random = np.array([x1[:half], x2[:half]]).T
	for k in range(half,final_time_unit):
		y, N = regression_random.predict(Y_ridge_random[-1,:], kernel_rbf_function_N)
		Y_ridge_random = np.append(Y_ridge_random, [y], axis=0)




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

	# Similarity Sampling
	def similarity_sampling_increment(X, x_candidates, similarity_function):
		dets = list()
		# M = similarity_function(X, X)
		# dets.append(np.linalg.det(M))
		for x in x_candidates:
			X_extended = X
			X_extended = np.append(X_extended, [x], axis=0)
			M = similarity_function(X_extended, X_extended)
			dets.append(np.linalg.det(M))
		id_maxdet =  dets.index(max(dets))
		x_maxdet = x[id_maxdet]
		return x_maxdet

	def similarity_sampling(X, m_tilde, similarity_function):
		m, n = np.shape(X)
		X_current = X[0:10,:]
		X_sampled = np.zeros((m_tilde, n))
		for i in range(m_tilde):
			a = X_current[10+i,:]
			b = X_current[10+(i+1)*2]
			candidates = list([a,b])
			X_sampled[i,:] =  similarity_sampling_increment(
				X_current,
				candidates,
				similarity_function,
				)




	# Plot
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(2, 1)
	axs[0].plot(time, x1, time, Y_ridge[:,0], time, Y_ridge_random[:,0])
	axs[0].set_xlim(0,final_time_unit)
	axs[0].set_xlabel('Time units (k)')
	axs[0].set_ylabel('Prey: x1 (actual) and x1 (ridge)')
	axs[0].grid(True)

	axs[1].plot(time, x2, time, Y_ridge[:,1], time, Y_ridge_random[:,1])
	axs[1].set_xlim(0,final_time_unit)
	axs[1].set_xlabel('Time units (k)')
	axs[1].set_ylabel('Pred: x2 (actual) and x2 (ridge)')
	axs[1].grid(True)
	# cxy, f = axs[1].cohere(x1, x2, 5, 1. / dt)

	fig.tight_layout()
	plt.show()