#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Confidence Interval - Plots

import numpy as np

if __name__ == "__main__":

	from model_closedloop import ClosedLoopSystem
	from model_plants import LinearPlant, LinearDCMotorPlant, \
	InvertedPendulum, PredatorPreyPlant

	# Scalar zero controller
	class NoController():
		def __init__(self):
			pass
		def action(self, x):
			return 0

	controller = NoController()
	# plant = PredatorPreyPlant(2/3, 4/3, 1.0, 1.0)
	plant = LinearDCMotorPlant(0.20, 0.015, 0.2, 1.015, 0.2 ,0.5)

	xo = np.array([20.0, 10.0]).T
	closed_loop = ClosedLoopSystem(plant, controller, xo, dt=0.01)

	final_time_unit = 2000
	half = int(final_time_unit*0.50)
	quarter = int(half*0.50)

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
	regression.training(X[:quarter,:], Y[:quarter,:], kernel_rbf_function_M)


	Y_ridge = np.array([x1[:half], x2[:half]]).T
	for k in range(half,final_time_unit):
		y, N = regression.predict(Y_ridge[-1,:], kernel_rbf_function_N)
		Y_ridge = np.append(Y_ridge, [y], axis=0)


	# RBF kernel id random
	random_ids = np.random.choice(half-1, size=quarter, replace=False)
	regression_random = KernelRidgeRegression(lambda_reg)
	regression_random.training(X[random_ids,:], Y[random_ids,:], kernel_rbf_function_M)

	Y_ridge_random = np.array([x1[:half], x2[:half]]).T
	for k in range(half,final_time_unit):
		y, N = regression_random.predict(Y_ridge_random[-1,:], kernel_rbf_function_N)
		Y_ridge_random = np.append(Y_ridge_random, [y], axis=0)


	# Plot
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(2, 1)
	axs[0].plot(time, x1, "r", time, Y_ridge[:,0], time, Y_ridge_random[:,0])
	axs[0].set_xlim(0,final_time_unit)
	axs[0].set_xlabel('Time units (k)')
	axs[0].set_ylabel('Prey: x1 (actual) and x1 (ridge)')
	axs[0].grid(True)
	axs[0].legend(['Actual', 'Pred', 'Pred random sampling'])

	axs[1].plot(time, x2, "r", time, Y_ridge[:,1], time, Y_ridge_random[:,1])
	axs[1].set_xlim(0,final_time_unit)
	axs[1].set_xlabel('Time units (k)')
	axs[1].set_ylabel('Pred: x2 (actual) and x2 (ridge)')
	axs[1].grid(True)
	axs[1].legend(['Actual', 'Pred', 'Pred random sampling'])
	# cxy, f = axs[1].cohere(x1, x2, 5, 1. / dt)

	fig.tight_layout()
	plt.show()





	# Plot
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(2, 1)
	axs[0].plot(x[half:-1,0], x[half+1:,0], "r", Y_ridge[half:-1,0], Y_ridge[half+1:,0], ".", Y_ridge_random[half:-1,0], Y_ridge_random[half+1:,0], ".")
	axs[0].set_xlabel('x1(k)')
	axs[0].set_ylabel('x1(k+1)')
	axs[0].grid(True)
	axs[0].legend(['Actual', 'Pred', 'Pred random sampling'])

	axs[1].plot(x[half:-1,1], x[half+1:,1], "r", Y_ridge[half:-1,1], Y_ridge[half+1:,1], ".", Y_ridge_random[half:-1,1], Y_ridge_random[half+1:,1],".")
	axs[1].set_xlabel('x2(k)')
	axs[1].set_ylabel('x2(k+1)')
	axs[1].grid(True)
	axs[1].legend(['Actual', 'Pred', 'Pred random sampling'])
	# cxy, f = axs[1].cohere(x1, x2, 5, 1. / dt)

	plt.show()




	# Plot
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(2, 1)
	axs[0].plot(x[half:,0], x[half:,1], "r.", Y_ridge[half:,0], Y_ridge[half:,1], ".", Y_ridge_random[half:,0], Y_ridge_random[half:,1], ".")
	axs[0].set_xlabel('x1(k)')
	axs[0].set_ylabel('x1(k+1)')
	axs[0].grid(True)
	axs[0].legend(['Actual', 'Pred', 'Pred random sampling'])

	plt.show()


	# 3D plot
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure(figsize=plt.figaspect(0.5))

	axs1 = fig.add_subplot(1,2,1, projection='3d')
	axs1.plot(x[half:-1,0],x[half:-1,1],x[half+1:,0], "r")
	axs1.plot(Y_ridge[half:-1,0], Y_ridge[half:-1,1], Y_ridge[half+1:,0])
	axs1.plot(Y_ridge_random[half:-1,0], Y_ridge_random[half:-1,1], Y_ridge_random[half+1:,0])
	axs1.legend(['Actual', 'Pred', 'Pred random sampling'])
	axs1.set_xlabel('x1(k)')
	axs1.set_ylabel('x2(k)')
	axs1.set_zlabel('x1(k+1)')

	axs2 = fig.add_subplot(1,2,2, projection='3d')
	axs2.plot(x[half:-1,0],x[half:-1,1],x[half+1:,1], "r")
	axs2.plot(Y_ridge[half:-1,0], Y_ridge[half:-1,1], Y_ridge[half+1:,1])
	axs2.plot(Y_ridge_random[half:-1,0], Y_ridge_random[half:-1,1], Y_ridge_random[half+1:,1])
	axs2.legend(['Actual', 'Pred', 'Pred random sampling'])
	axs2.set_xlabel('x1(k)')
	axs2.set_ylabel('x2(k)')
	axs2.set_zlabel('x2(k+1)')

	# axs[1] = fig.add_subplot(111, projection='3d')
	# axs[1].plot(x[half:-1,0],x[half:-1,1],x[half+1:,1], Y_ridge[half:-1,0], Y_ridge[half:-1,1], Y_ridge[half+1:,1], Y_ridge_random[half:-1,0], Y_ridge_random[half:-1,1], Y_ridge_random[half+1:,1])

	plt.show()


