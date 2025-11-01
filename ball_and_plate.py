#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:25:53 2025

@author: andthem
"""

import casadi.casadi as cs
import opengen as og
from Solver import Problem, Solver

import numpy as np
import matplotlib.pyplot as plt

build = False
build = True

name = "ball_and_plate"
folder = "python_build"

mass_ball = 1
moment_inertia = 0.0005
gravity_acceleration = 9.8044
dt = 0.01
nx = 4
nu = 1
N = 15

substeps = 10

x0 = [0.1, -0.5, 0.0, 0.0]
steps = 2000

U = og.constraints.BallInf(None, 0.95)


def dynamics_ct(xk, uk, P=None):
	dx1 = xk[1]
	dx2 = (5 / 7) * (xk[0] * xk[3]**2 - gravity_acceleration * cs.sin(xk[2]))
	dx3 = xk[3]
	dx4 = (uk[0] - mass_ball * gravity_acceleration * xk[0] * cs.cos(xk[2])
		- 2 * mass_ball * xk[0] * xk[1] * xk[3]) \
		/ (mass_ball * xk[0]**2 + moment_inertia)
	return cs.vertcat(dx1, dx2, dx3, dx4)


def stage_cost(xk, uk, P=None):
	cost = 5 * xk[0]**2 + 0.01 * xk[1]**2 + 0.01 * xk[2]**2 + 0.05 * xk[3]**2 + 2.2 * uk[0]**2
	return cost


def final_cost(xN, P=None):
	cost = 100 * xN[0]**2 + 50 * xN[2]**2 + 20 * xN[1]**2 + 0.8 * xN[3]**2
	return cost


P = Problem(nx, nu)
P.N = N
P.dt = dt
P.stage_cost = stage_cost
P.final_cost = final_cost
P.dynamics_ct = dynamics_ct
P.input_constraints = U


S = Solver(problem=P, name=name)
S.substeps = substeps
S.initialize(build=build)
S.run(x0, steps)


ss = S.state_sequence
time = np.arange(0, dt * steps, dt)

plt.plot(time, [float(x[0]) for x in ss[:-1]], '-', label="position")
plt.plot(time, [float(x[2]) for x in ss[:-1]], '-', label="angle")
plt.grid()
plt.ylabel('states')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(0.7, 0.85), loc='upper left', borderaxespad=0.)
plt.show()
