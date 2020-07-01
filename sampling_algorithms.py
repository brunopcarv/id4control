#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Sampling algorithms - Sample Selection

import numpy as np
import math

class MinimumSimilaritySquaredNorm()

	def __init__():
		self.M = np.empty((0,1)) #Similarity matrix (Gramian or Kernel matrix)
		self.samples = np.empty((0,1)) #Samples (decide later if np.array or list)
		self.max_acceptable_norm = math.inf
		self.full = False
		self.theta = 0

	def kernel_rbf_function_N(x, X, var=1.0, gamma=1.0):
		import numexpr as ne
		diff = X-x
		norm_squared = np.sum(diff**2, axis=-1)
		return ne.evaluate('v * exp(-g * A)', {
		        'A' : norm_squared,
		        'g' : gamma,
		        'v' : var
		})

	def new_sample(self, sample):
		if not self.full:

			if theta == 0:
				self.samples = np.array([sample])
				self.M = np.array([[1.0]])
			else:

			self.samples = np.append(self.samples, [sample], axis=0)
			v = kernel_rbf_function_N(sample, self.samples)
			self.M = np.append(self.M, [v], axis=0)
			v_with_diag = np.append(v, [1.0])
			v_transpose = v_with_diag.T
			self.M = np.append(self.M, [v_transpose], axis=1)



