#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Confidence Interval - Plots

import logging

import numpy as np

from id_kernel_ridge_regression import KernelRidgeRegression, \
kernel_rbf_function_M, kernel_rbf_function_N, kernel_linear_function, \
kernel_poly_function, kernel_tanh_function
from id_linear_ridge_regression import LinearRidgeRegression

from model_closedloop import ClosedLoopSystem
from model_plants import LinearPlant, LinearDCMotorPlant, \
InvertedPendulum, PredatorPreyPlant

logger = logging.getLogger(__name__)


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(
	 description="Test",
	)

	parser.add_argument("--log-level",
	 default="INFO",
	 help="Logging level (eg. INFO, see Python logging docs)",
	)

	parser.add_argument("--kernel", 
	 choices=['linear', 'poly', 'rbf', 'tanh'],
	 default='linear',
	 help="Kernel function for kernel ridge regression",
	)

	parser.add_argument("--plant", 
	 choices=['linear', 'dcmotor', 'invertedpendulum', 'predatorprey'],
	 default='predatorprey',
	 help="Model plant to be simulated",
	)

	parser.add_argument("--dt", 
	 default=0.01,
	 type=np.float32,
	 help="Sampling time of simulation (seconds)",
	)

	parser.add_argument("--timelapses",
	 type=int,
	 default=2000,
	 help="Final time of model simulation",
	)

	parser.add_argument("--trainration",
	 default=0.5,
	 help="Dataset training ratio",
	)

	parser.add_argument("--samplingration",
	 default=0.5,
	 help="Ratio of the training dataset available for actual training",
	)

	try:
		import argcomplete
		argcomplete.autocomplete(parser)
	except:
		pass

	args = parser.parse_args()

	logging.basicConfig(
	 level=getattr(logging, args.log_level),
	 format="%(levelname)s %(message)s"
	)


	kernel = args.kernel
	dispatcher0={'linear':kernel_linear_function,
				'poly':kernel_poly_function,
				'rbf':kernel_rbf_function_M,
				'tanh':kernel_tanh_function}
	dispatcher1={'linear':kernel_linear_function,
				'poly':kernel_poly_function,
				'rbf':kernel_rbf_function_N,
				'tanh':kernel_tanh_function}
	try:
		kernel_function=[dispatcher0[kernel], dispatcher1[kernel]]
	except KeyError:
		raise ValueError('invalid input')
	logger.info("Kernel: %s", kernel)

	plant = args.plant
	dispatcher2={'linear':LinearPlant,
				'dcmotor':LinearDCMotorPlant,
				'invertedpendulum':InvertedPendulum,
				'predatorprey':PredatorPreyPlant}
	try:
		plant_model=dispatcher2[plant]
	except KeyError:
		raise ValueError('invalid input')
	logger.info("Plant: %s", plant)

	dt = args.dt
	logger.info("dt: %ss", dt)

	number_of_time_lapses = args.timelapses
	logger.info("Number of time lapses: %d", number_of_time_lapses)

	trainration = args.trainration
	samplingratio = args.trainration
	number_of_training_points = int(number_of_time_lapses*trainration)
	number_of_sampled_points = int(number_of_training_points*samplingratio)
	logger.info("Train ratio: %s, total points: %d", trainration, number_of_training_points)
	logger.info("Sampling ratio: %s, total points: %d", samplingratio, number_of_sampled_points)

	random_ids = np.random.choice(number_of_training_points-1, size=number_of_sampled_points, replace=False)
	logger.info("Total random points: %d", np.shape(random_ids)[0])

	logger.info("Final time: %ss", number_of_time_lapses*dt)


	# Scalar zero controller
	class NoController():
		def __init__(self):
			pass
		def action(self, x):
			return 0

	controller = NoController()
	plant = plant_model(1.0, 1.0, 1.0, 1.0)
	# plant = LinearDCMotorPlant(0.20, 0.015, 0.2, 1.015, 0.2 ,0.5)

	xo = np.array([2.0, 1.0]).T
	closed_loop = ClosedLoopSystem(plant, controller, xo, dt=dt)



	x, time = closed_loop.run(number_of_time_lapses)
	x1 = x[:,0]
	x2 = x[:,1]

	X_train = np.array([x1[:number_of_training_points-1], x2[:number_of_training_points-1]]).T
	Y_train = np.array([x1[1:number_of_training_points], x2[1:number_of_training_points]]).T


	# RBF kernel id
	lambda_reg = 0.00001
	regression = KernelRidgeRegression(lambda_reg)
	regression.training(X_train[:number_of_sampled_points,:], Y_train[:number_of_sampled_points,:], kernel_function[0])


	Y_ridge = np.array([x1[:number_of_training_points], x2[:number_of_training_points]]).T
	for k in range(number_of_training_points,number_of_time_lapses):
		y = regression.predict(Y_ridge[-1,:], kernel_function[1])
		Y_ridge = np.append(Y_ridge, [y], axis=0)


	# RBF kernel id random
	regression_random = KernelRidgeRegression(lambda_reg)
	regression_random.training(X_train[random_ids,:], Y_train[random_ids,:], kernel_function[0])

	Y_ridge_random = np.array([x1[:number_of_training_points], x2[:number_of_training_points]]).T
	for k in range(number_of_training_points,number_of_time_lapses):
		y = regression_random.predict(Y_ridge_random[-1,:], kernel_function[1])
		Y_ridge_random = np.append(Y_ridge_random, [y], axis=0)


	# Plot
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(2, 1)
	axs[0].plot(time, x1, "r", time, Y_ridge[:,0], time, Y_ridge_random[:,0])
	axs[0].set_xlim(0,number_of_time_lapses)
	axs[0].set_xlabel('Time units (k)')
	axs[0].set_ylabel('Prey: x1 (actual) and x1 (ridge)')
	axs[0].grid(True)
	axs[0].legend(['Actual', 'Pred', 'Pred random sampling'])

	axs[1].plot(time, x2, "r", time, Y_ridge[:,1], time, Y_ridge_random[:,1])
	axs[1].set_xlim(0,number_of_time_lapses)
	axs[1].set_xlabel('Time units (k)')
	axs[1].set_ylabel('Pred: x2 (actual) and x2 (ridge)')
	axs[1].grid(True)
	axs[1].legend(['Actual', 'Pred', 'Pred random sampling'])
	# cxy, f = axs[1].cohere(x1, x2, 5, 1. / dt)

	fig.tight_layout()
	plt.show()





	# # Plot
	if 0:
		import matplotlib.pyplot as plt
		fig, axs = plt.subplots(2, 1)
		axs[0].plot(x[number_of_training_points:-1,0], x[number_of_training_points+1:,0], "r", Y_ridge[number_of_training_points:-1,0], Y_ridge[number_of_training_points+1:,0], ".", Y_ridge_random[number_of_training_points:-1,0], Y_ridge_random[number_of_training_points+1:,0], ".")
		axs[0].set_xlabel('x1(k)')
		axs[0].set_ylabel('x1(k+1)')
		axs[0].grid(True)
		axs[0].legend(['Actual', 'Pred', 'Pred random sampling'])

		axs[1].plot(x[number_of_training_points:-1,1], x[number_of_training_points+1:,1], "r", Y_ridge[number_of_training_points:-1,1], Y_ridge[number_of_training_points+1:,1], ".", Y_ridge_random[number_of_training_points:-1,1], Y_ridge_random[number_of_training_points+1:,1],".")
		axs[1].set_xlabel('x2(k)')
		axs[1].set_ylabel('x2(k+1)')
		axs[1].grid(True)
		axs[1].legend(['Actual', 'Pred', 'Pred random sampling'])
		# cxy, f = axs[1].cohere(x1, x2, 5, 1. / dt)

		plt.show()




	# # Plot
	if 0:
		import matplotlib.pyplot as plt
		fig, axs = plt.subplots(2, 1)
		axs[0].plot(x[number_of_training_points:,0], x[number_of_training_points:,1], "r.", Y_ridge[number_of_training_points:,0], Y_ridge[number_of_training_points:,1], ".", Y_ridge_random[number_of_training_points:,0], Y_ridge_random[number_of_training_points:,1], ".")
		axs[0].set_xlabel('x1(k)')
		axs[0].set_ylabel('x1(k+1)')
		axs[0].grid(True)
		axs[0].legend(['Actual', 'Pred', 'Pred random sampling'])

		plt.show()


	# 3D plot
	if 0:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure(figsize=plt.figaspect(0.5))

		axs1 = fig.add_subplot(1,2,1, projection='3d')
		axs1.plot(x[number_of_training_points:-1,0],x[number_of_training_points:-1,1],x[number_of_training_points+1:,0], "r")
		axs1.plot(Y_ridge[number_of_training_points:-1,0], Y_ridge[number_of_training_points:-1,1], Y_ridge[number_of_training_points+1:,0])
		axs1.plot(Y_ridge_random[number_of_training_points:-1,0], Y_ridge_random[number_of_training_points:-1,1], Y_ridge_random[number_of_training_points+1:,0])
		axs1.legend(['Actual', 'Pred', 'Pred random sampling'])
		axs1.set_xlabel('x1(k)')
		axs1.set_ylabel('x2(k)')
		axs1.set_zlabel('x1(k+1)')

		axs2 = fig.add_subplot(1,2,2, projection='3d')
		axs2.plot(x[number_of_training_points:-1,0],x[number_of_training_points:-1,1],x[number_of_training_points+1:,1], "r")
		axs2.plot(Y_ridge[number_of_training_points:-1,0], Y_ridge[number_of_training_points:-1,1], Y_ridge[number_of_training_points+1:,1])
		axs2.plot(Y_ridge_random[number_of_training_points:-1,0], Y_ridge_random[number_of_training_points:-1,1], Y_ridge_random[number_of_training_points+1:,1])
		axs2.legend(['Actual', 'Pred', 'Pred random sampling'])
		axs2.set_xlabel('x1(k)')
		axs2.set_ylabel('x2(k)')
		axs2.set_zlabel('x2(k+1)')

		plt.show()


	x = y = np.arange(0.0, 3.0, 0.05)
	X, Y = np.meshgrid(x, y)
	z1 = np.zeros([np.shape(x)[0], np.shape(y)[0]])
	z2 = np.zeros([np.shape(x)[0], np.shape(y)[0]])
	z1_reg = np.zeros([np.shape(x)[0], np.shape(y)[0]])
	z2_reg = np.zeros([np.shape(x)[0], np.shape(y)[0]])
	z1_regrndsample = np.zeros([np.shape(x)[0], np.shape(y)[0]])
	z2_regrndsample = np.zeros([np.shape(x)[0], np.shape(y)[0]])

	for i in range(np.shape(x)[0]):
		for j in range(np.shape(y)[0]):
			z1[i,j], z2[i,j] = plant.dynamics(np.array([x[i], y[j]]), 0, dt)
			z1_reg[i,j], z2_reg[i,j] = regression.predict(np.array([x[i], y[j]]), kernel_function[1])
			z1_regrndsample[i,j], z2_regrndsample[i,j] = regression_random.predict(np.array([x[i], y[j]]), kernel_function[1])



	# 3D plot
	if 0:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure(figsize=plt.figaspect(0.5))
		axs1 = fig.add_subplot(1,2,1, projection='3d')
		axs1.plot_surface(X, Y, z1)
		axs1.plot_surface(X, Y, z1_reg)
		axs1.plot_surface(X, Y, z1_regrndsample)
		axs1.scatter3D(X_train[:number_of_sampled_points,0], X_train[:number_of_sampled_points,1], np.zeros(np.shape(X_train[:number_of_sampled_points,0])), c='r', marker='x')
		axs1.scatter3D(X_train[random_ids,0], X_train[random_ids,1], np.zeros(np.shape(X_train[random_ids,0])), c='g', marker='x')
		# axs1.legend(['Actual', 'Pred', 'Pred random sampling'])

		axs2 = fig.add_subplot(1,2,2, projection='3d')
		axs2.plot_surface(X, Y, z2)
		axs2.plot_surface(X, Y, z2_reg)
		axs2.plot_surface(X, Y, z2_regrndsample)
		axs2.scatter3D(X_train[:number_of_sampled_points,0], X_train[:number_of_sampled_points,1], np.zeros(np.shape(X_train[:number_of_sampled_points,0])), c='r', marker='x')
		# axs2.legend(['Actual', 'Pred', 'Pred random sampling'])
		plt.show()


	if 1:
		# # Matshow plot
		Z1 = z1.reshape(X.shape)
		Z1_reg = z1_reg.reshape(X.shape)
		Z1_regrndsample = z1_regrndsample.reshape(X.shape)

		import matplotlib
		import matplotlib.pyplot as plt
		import matplotlib.colors as colors
		fig = plt.figure(figsize=plt.figaspect(0.5))


		# field1 = (1+np.abs(Z1_reg-Z1)) ** 10
		# field2 = (1+np.abs(Z1_regrndsample -Z1)) ** 10 
		field1 = Z1_reg-Z1
		field2 = Z1_regrndsample-Z1
		combined_data = np.array([field1,field2])
	    #Get the min and max of all your data
		_min, _max = np.amin(combined_data), np.amax(combined_data)


		class MidpointNormalize(colors.Normalize):
			def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
				self.vcenter = vcenter
				colors.Normalize.__init__(self, vmin, vmax, clip)

			def __call__(self, value, clip=None):
				x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
				return np.ma.masked_array(np.interp(value, x, y))

		midnorm = MidpointNormalize(vmin=_min, vcenter=0, vmax=_max)

		axs1 = fig.add_subplot(1,2,1)
		# img1 = axs1.matshow(field1, aspect="auto", cmap=plt.cm.YlGn, norm=colors.LogNorm(vmin=_min, vmax=_max))
		img1 = axs1.matshow(field1, aspect="auto", cmap=plt.cm.PuOr, norm=midnorm, extent=[0.0, 3.0, 0.0, 3.0])
		fig.colorbar(img1)
		axs1.scatter(X_train[:number_of_sampled_points,0], X_train[:number_of_sampled_points,1], c='r', s=1)

		axs2 = fig.add_subplot(1,2,2)
		# img2 = axs2.matshow(field2, aspect="auto", cmap=plt.cm.YlGn, norm=colors.LogNorm(vmin=_min, vmax=_max))
		img2 = axs2.matshow(field2, aspect="auto", cmap=plt.cm.PuOr, norm=midnorm, extent=[0.0, 3.0, 0.0, 3.0])
		fig.colorbar(img2)
		axs2.scatter(X_train[random_ids,0], X_train[random_ids,1], c='r', s=1)

		plt.show()