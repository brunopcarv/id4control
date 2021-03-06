#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Linear Example - Model

import numpy as np
import math

class LinearModel():

	def __init__(self, A, B, xo, dt):
		self.A = A
		self.B = B
		self.x = xo
		self.xo = xo
		self.k = 0
		self.dt = dt # delta t: sample period
		self.u = None

	def forcing(self): #Control law
		return self.u


	def next_points(self):
		x_temp = self.x

		if 0:
			u = self.forcing()
			self.x = self.A.dot(x_temp) + self.B.dot(u)
		else:
			self.x = self.A.dot(x_temp)
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