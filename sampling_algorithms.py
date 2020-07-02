#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Sampling algorithms - Sample Selection

import numpy as np
import math

# Minimum Squared-Norm of Similarity Sampling (MSNSS)
class MinSquaredNormSimilaritySampling():

	def __init__(self, n_samples):
		self.M = np.empty((0,1)) #Similarity matrix (Gramian or Kernel matrix)
		self.samples = np.empty((0,1)) #Samples collection
		self.squared_norms = np.zeros((1)) #Vector that keeps track of squared norms
		self.theta = 0
		self.n_samples = n_samples
		self.max_square_norm = math.inf

	def kernel_rbf_function_N(self, x, X, var=1.0, gamma=1.0):
		import numexpr as ne
		diff = X-x
		norm_squared = np.sum(diff**2, axis=-1)
		return ne.evaluate('v * exp(-g * A)', {
		        'A' : norm_squared,
		        'g' : gamma,
		        'v' : var
		})

	def append_sample(self, sample, v, squared_norm):
		v_with_diag = np.append(v, [1.0])
		v_transpose = np.array([v_with_diag]).T

		self.M = np.append(self.M, [v], axis=0)
		self.M = np.append(self.M, v_transpose, axis=1)

		self.squared_norms += v*v
		self.squared_norms = np.append(self.squared_norms, squared_norm)

		self.samples = np.append(self.samples, [sample], axis=0)

	def remove_sample(self):
		idx_to_remove = np.argmax(self.squared_norms)
		# sample_to_remove = self.samples[idx_to_remove]

		v_to_remove = self.M[idx_to_remove]

		squared_norms_vector_to_remove = v_to_remove*v_to_remove

		self.M = np.delete(self.M, idx_to_remove, 0)
		self.M = np.delete(self.M, idx_to_remove, 1)

		self.squared_norms -= squared_norms_vector_to_remove
		self.squared_norms = np.delete(self.squared_norms, idx_to_remove, 0)

		self.samples = np.delete(self.samples, idx_to_remove, 0)



	def replace_sample(self, sample, v, squared_norm):
		self.append_sample(sample, v, squared_norm)
		self.remove_sample()


	def new_sample(self, sample):

		if self.theta == 0:
			self.samples = np.array([sample])
			self.M = np.array([[1.0]])

			self.theta += 1

		else:
			v = self.kernel_rbf_function_N(sample, self.samples)
			squared_norm = np.dot(v.T, v)

			if self.theta < self.n_samples:
				self.append_sample(sample, v, squared_norm)

				self.theta += 1

			else:
				if squared_norm < max(self.squared_norms + v*v):
					self.replace_sample(sample, v, squared_norm)

		self.max_square_norm = max(self.squared_norms)



if __name__  == "__main__":

	import argparse
	import logging

	logger = logging.getLogger(__name__)

	parser = argparse.ArgumentParser(
	 description="Test",
	)

	parser.add_argument("--log-level",
	 default="INFO",
	 help="Logging level (eg. INFO, see Python logging docs)",
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


	sampling_method = MinSquaredNormSimilaritySampling(10)

	def test_max_norm_of_M(M):
		norms = list()
		for i in range(np.shape(M)[0]):
			norms.append(np.dot(M[i], M[i]))
		return max(norms)-1

	for i in range(100):
		sample = np.random.rand(10)
		sampling_method.new_sample(sample)
		# logger.info("Sample: %s", sample)
		# logger.info("Similarity Matirx M: %s", sampling_method.M)
		# logger.info("Samples: %s", sampling_method.samples)
		# logger.info("Squared_norms : %s", sampling_method.squared_norms)
		# logger.info("Theta : %s", sampling_method.theta)
		logger.info("Max_squared_norm : %s", sampling_method.max_square_norm)
		logger.info("Max norm by test: %s", test_max_norm_of_M(sampling_method.M))
		logger.info("Det of M: %s", np.linalg.det(sampling_method.M))







