#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Sampling algorithms - Sample Selection

import logging

import numpy as np
import math
from termcolor import colored

logger = logging.getLogger(__name__)

# Minimum Squared-Norm of Similarity Sampling (MSNSS)
class MinSquaredNormSimilaritySampling():

	def __init__(self, n_samples, similarity_function):
		self.M = np.empty((0,1)) #Similarity matrix (Gramian or Kernel matrix)
		self.samples = np.empty((0,1)) #Samples collection
		self.squared_norms = np.zeros((1)) #Vector that keeps track of squared norms
		self.theta = 0
		self.n_samples = n_samples
		self.max_square_norm = math.inf
		self.replaced = False
		self.similarity_function = similarity_function
		self.dataset_ids = list()
		self.current_dataset_id = None

	def append_sample(self, sample, v, squared_norm):
		v_with_diag = np.append(v, [1.0])
		v_transpose = np.array([v_with_diag]).T

		self.M = np.append(self.M, [v], axis=0)
		self.M = np.append(self.M, v_transpose, axis=1)

		self.squared_norms += v*v
		self.squared_norms = np.append(self.squared_norms, squared_norm)

		self.samples = np.append(self.samples, [sample], axis=0)

		self.dataset_ids.append(self.current_dataset_id)

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

		del self.dataset_ids[idx_to_remove]


	def replace_sample(self, sample, v, squared_norm):
		self.append_sample(sample, v, squared_norm)
		self.remove_sample()


	def new_sample(self, sample, dataset_id):
		self.replaced = False
		self.current_dataset_id = dataset_id

		if self.theta == 0:
			self.samples = np.array([sample])
			self.M = np.array([[1.0]])

			logger.debug("Sample added")
			self.dataset_ids = [self.current_dataset_id ]
			self.theta += 1

		else:
			v = self.similarity_function(sample, self.samples)
			squared_norm = np.dot(v.T, v)

			if self.theta < self.n_samples:
				self.append_sample(sample, v, squared_norm)

				logger.debug("Sample added")
				self.theta += 1

			else:
				if squared_norm < max(self.squared_norms + v*v):
					self.replace_sample(sample, v, squared_norm)
					logger.debug("Sample replaced")
					self.replaced = True
				else:
					logger.debug("Sample discarded")

		self.max_square_norm = max(self.squared_norms)


if __name__  == "__main__":

	import argparse

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


	def kernel_rbf_function_N(x, X, var=1.0, gamma=1.0):
		import numexpr as ne
		diff = X-x
		norm_squared = np.sum(diff**2, axis=-1)
		return ne.evaluate('v * exp(-g * A)', {
		        'A' : norm_squared,
		        'g' : gamma,
		        'v' : var
		})


	sampling_method = MinSquaredNormSimilaritySampling(5, kernel_rbf_function_N)

	def test_max_norm_of_M(M):
		norms = list()
		for i in range(np.shape(M)[0]):
			norms.append(np.dot(M[i], M[i]))
		return max(norms)-1

	last_detM = 0
	old_sample =  np.random.rand(2)
	for i in range(10):
		if 0:
			logger.info("-------------- New sample ---------------")
			sample = old_sample
			sampling_method.new_sample(sample)
			if 1:
				logger.debug("Sample: %s", sample)
				logger.debug("Similarity Matirx M: %s", sampling_method.M)
				logger.debug("Samples: %s", sampling_method.samples)
				logger.debug("Squared_norms : %s", sampling_method.squared_norms)
				logger.debug("Theta : %s", sampling_method.theta)
				logger.debug("Datase_ids: %s", sampling_method.dataset_ids)

			logger.info("Max_squared_norm : %s", sampling_method.max_square_norm)
			logger.info("Max norm by test: %s", test_max_norm_of_M(sampling_method.M))
			logger.info("Det of M: %s", np.linalg.det(sampling_method.M))


		if 0:
			# sample = np.random.rand(2)
			sample = np.array([0.1, 0.2, 0.3])
			sampling_method.new_sample(sample, i)
			if sampling_method.replaced:
				detM = np.linalg.det(sampling_method.M)
				if last_detM < detM:
					logger.info(colored("-------------- Sample Replaced ---------------", 'green'))
				else:
					logger.info(colored("-------------- Sample Replaced ---------------", 'red'))

				logger.info("Max_squared_norm : %s", sampling_method.max_square_norm)
				logger.info("Max norm by test: %s", test_max_norm_of_M(sampling_method.M))
				logger.info("Det of M: %s", detM)
				logger.info("Datase_ids: %s", sampling_method.dataset_ids)
				last_detM = detM

		if 1:
			# sample = np.array([0.1+i, 0.2/(i+1), 0.3**i])
			sample = old_sample + (np.identity(2)@old_sample)/10
			sampling_method.new_sample(sample, i)

			logger.info(colored("-------------- New Sample ---------------", 'blue'))
			logger.debug("Sample: %s", sample)
			logger.debug("Similarity Matirx M: %s", sampling_method.M)
			logger.debug("Samples: %s", sampling_method.samples)
			logger.debug("Squared_norms : %s", sampling_method.squared_norms)
			logger.debug("Theta : %s", sampling_method.theta)
			logger.debug("Datase_ids: %s", sampling_method.dataset_ids)

			if sampling_method.replaced:
				detM = np.linalg.det(sampling_method.M)
				if last_detM < detM:
					logger.info(colored("-------------- Sample Replaced ---------------", 'green'))
				else:
					logger.info(colored("-------------- Sample Replaced ---------------", 'red'))

				logger.info("Max_squared_norm : %s", sampling_method.max_square_norm)
				logger.info("Max norm by test: %s", test_max_norm_of_M(sampling_method.M))
				logger.info("Det of M: %s", detM)
				logger.info("Datase_ids: %s", sampling_method.dataset_ids)
				last_detM = detM
			old_sample = sample