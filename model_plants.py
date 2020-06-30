#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Collection of system plants

import numpy as np


### General linear plant (Discrete)
class LinearPlantDiscrete():
	def __init__(self, A, B, C, H=np.array([1, 1]).T):
		self.A = A
		self.B = B
		self.C = C
		self.D = D
		self.H = H

	def dynamics(self, x, u, eta): #with additive noise eta
		Eta = np.array([next(eta), next(eta)])
		return self.A.dot(x) + self.B.dot(u) + self.H.dot(Eta)

	def output(self, x, u):
		return self.C.dot(x) + self.D.dot(u)


### General linear plant (Continuous)
class LinearPlantContinuous():
	def __init__(self, A, B, C, H=np.array([1, 1]).T):
		self.A = A
		self.B = B
		self.C = C
		self.D = D
		self.H = H

	def dynamics(self, x, u, eta): #with additive noise eta
		Eta = np.array([next(eta), next(eta)])
		return x + (self.A.dot(x) + self.B.dot(u) + self.H.dot(Eta))*dt

	def output(self, x, u):
		return self.C.dot(x) + self.D.dot(u)


### DC Motor - Linear
class LinearDCMotorPlant():
	"""
	States:
	w = load angular rate
	i = current

	Input:
	v_app = applied voltage

	Output (measured quantity):
	w = load angular rate

	Model parameters:
	J = inertial load
	Kb = emf constant
	Kf = linear viscous friction constant
	Km = armature constant
	R = circuit resitance
	L = armature self-inductance

	References:
	- https://ww2.mathworks.cn/help/control/getstart/linear-lti-models.html
	- http://webfiles.portal.chalmers.se/et/MSc/BaldurssonStefanMSc.pdf
	"""

	def __init__(self, J, Kb, Kf, Km, R, L, H=np.array([1, 1]).T):
		self.A = np.array([[-R/L, -Kb/L],[Km/J, -Kf/J]])
		self.B = np.array([1/L, 0]).T
		self.C = np.array([0, 1])
		self.H = H

	def dynamics(self, x, u, eta, dt): #with additive noise eta
		Eta = np.array([next(eta), next(eta)])
		return x + (self.A.dot(x) + self.B.dot(u) + self.H.dot(Eta))*dt

	def output(self, x, u):
		return self.C.dot(x)


### Predator Prey - Nonlinear
class PredatorPreyPlant():
	"""
	Parameters (growth, death, and mutual influences rates)
	a, b, c, d = 1.0, 1.0, 1.0, 1.0

	States
	x1 = prey population
	x2 = predator population

	Dynamics
	x1(k+1) = a x1(k) - b x1(k) x2(k) + u**2
	x2(k+1) = - c x2(k) + d x1(k) x2(k)
	"""
	def __init__(self, a, b, c, d, h1=1, h2=1):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.h1 = h1
		self.h2 = h2

	def dynamics(self, x, u, eta, dt): #with additive noise eta
		x0 = x[0]+ (self.a*x[0] - self.b*x[0]*x[1] + u**2 + self.h1*next(eta))*dt
		x1 = x[1]+ (-self.c*x[1] + self.d*x[0]*x[1] + self.h2*next(eta))*dt
		return np.array([x0, x1]).T

	def output(self, x, u):
		return x


### Inverted Pnedulum - Nonlinear
class InvertedPendulum():
	"""
	State space: x.T = [y ydot theta thetadot]
	- y = cart position coordinate
	- theta = pendulum angle from vertical

	Parameters:
	- M = cart mass
	- m = pedulum mass
	- b = friction coefficient
	- I = mass moment of inertia of pendulum
	- l = length to pendulum center of mass

	Input:
	u = [F]: force applied to cart

	Reference: shorturl.at/cvGQ4
	"""

	GRATVITY = 9.807

	def __init__(self, M=0.5, m=0.2, b=0.1, I=0.006, l=0.3, H=np.array([1,1,1,1]).T):
		self.M = M
		self.m = m
		self.b = b
		self.I = I
		self.l = l
		self.dt = dt
		self.H = H

	def dynamics(self, x, u, eta, dt): #with additive noise eta
		den = self.I + self.m*self.l**2
		bigden = self.M + self.m - (self.m*self.l*np.cos(x[2]))**2/den
		num0 = -self.b*x[1]
		num1 = (self.m*self.l)**2*GRAVITY*np.sin(x[2])*np.cos(x[2])/den
		num2 = self.m*self.l*x[3]**2*np.sin(x[2])
		x0 = x[0] + (x[1] + self.H[0]*next(eta))*dt 
		x1 = x[1] + ((num0+num1+num2+u)/bigden + self.H[1]*next(eta))*dt
		x2 = x[2] + (x[3] + self.H[2]*next(eta))*dt
		x3 = x[3] + ((-self.m*self.l*(x1*np.cos(x[2])+GRAVITY*np.sin(x[2])))/den + self.H[3]*next(eta))*dt
		return np.array([x0, x1, x2, x3]).T

	def output(self, x, u):
		return x