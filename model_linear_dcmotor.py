#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
#  Linear DC Motor - Model

import numpy as np

from nmodel_linear import LinearModel

# References:
# - https://ww2.mathworks.cn/help/control/getstart/linear-lti-models.html
# - http://webfiles.portal.chalmers.se/et/MSc/BaldurssonStefanMSc.pdf

# States:
# w = load angular rate
# i = current

# Input:
# v_app = applied voltage

# Output (measured quantity):
# w = load angular rate

# Model parameters:
# J = inertial load
# Kb = emf constant
# Kf = linear viscous friction constant
# Km = armature constant
# R = circuit resitance
# L = armature self-inductance

class LinearDCMotorModel():

	def __init__(self, J, Kb, Kf, Km, R, L, xo, dt):
		self.A = np.array([[-R/L, -Kb/L],[Km/J, -Kf/J]])
		self.B = np.array([1/L, 0]).T
		self.model = LinearModel(self.A, self.B, xo, dt)