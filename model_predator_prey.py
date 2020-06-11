#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Predator-Prey Example - Lotka-Volterra Model

import numpy as np
import math

class PredatorPreyModel():

# Parameters (growth, death, and mutual influences rates)
# a, b, c, d = 1, 1, 1, 1

# States
# x1 = prey population
# x2 = predator population

# Model
# x1(k+1) = a x1(k) - b x1(k) x2(k)
# x2(k+1) = - c x2(k) + d x1(k) x2(k)

	def __init__(self, a, b, c, d, x1o, x2o, dt):
		self.x1o = x1o
		self.x2o = x2o
		self.x1 = x1o
		self.x2 = x2o
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.k = 0
		self.dt = dt # delta t: sample period

	def forcing(self):
		return 2*math.sin(self.k*self.dt) + 2*math.sin(self.k*self.dt/10)


	def next_points(self):
		x1_temp = self.x1
		x2_temp = self.x2

		u = self.forcing()
		if 1:
			u = 0
		# self.x1 = x1_temp + self.a*x1_temp - self.a*x1_temp**2- self.b*x1_temp*x2_temp + u**2
		self.x1 = x1_temp + (self.a*x1_temp - self.b*x1_temp*x2_temp + u**2)*self.dt
		self.x2 = x2_temp + (-self.c*x2_temp + self.d*x1_temp*x2_temp)*self.dt
		self.k += 1
		return self.x1, self.x2

	def	reset(self):
		self.x1 = self.x1o
		self.x2 = self.x2o
		self.k = 0

	def run(self, k):
		x1_list = list()
		x2_list = list()

		for i in range(k):
			x1_list.append(self.x1)
			x2_list.append(self.x2)
			self.next_points()

		self.reset()
		return x1_list, x2_list, range(k)