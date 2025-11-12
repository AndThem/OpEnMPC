#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:51:17 2025

@author: andthem
"""

import casadi.casadi as cs
import opengen as og
from Solver import ProblemMPC, ProblemLMPC, ProblemLMPC0, Solver

import numpy as np
import matplotlib.pyplot as plt

build = False
build = True

name = "inverted_pendulum"
folder = "python_build"

# =============================================================================
# Define problem
# https://onlinelibrary.wiley.com/doi/epdf/10.1002/rnc.70083
# =============================================================================

# Parameters (no need to rebuild after changing these)
x0 = [0.0, cs.pi, 0.0, 0.0]
steps = 2000
substeps = 10

# Problem data (need to rebuild after changing these)
nx = 4
nu = 1
N = 20
dt = 0.05

# Cost
x_ref = [0.0] * nx
u_ref = [0.0] * nu

Q = [2.5, 50.0, 0.01, 0.01]
Qf = [3.0, 50.0, 0.02, 0.02]
R = [0.1]

# Constraints
U = og.constraints.BallInf(None, 15.0)

# Dynamics
mc, mp, l, g = 1.0, 0.2, 1.0, 9.81


def dynamics_ct(x, u, P: dict = None):
	s = cs.sin(x[1])
	c = cs.cos(x[1])

	dx1 = x[2]
	dx2 = x[3]

	dx3 = -mp * l * x[3]**2 * s + mp * g * s * c + u[0]
	dx3 = dx3 / (mc + mp * c**2)

	dx4 = -mp * l * x[3]**2 * s * c + (mc + mp) * g * s + u[0] * c
	dx4 = dx4 / (l * (mc + mp * c**2))

	return cs.vertcat(dx1, dx2, dx3, dx4)


def stage_cost(xk, uk, k: int = None, P: dict = None):
	cost = 0.0
	for i in range(nx):
		cost += Q[i] * (xk[i] - x_ref[i])**2
	for i in range(nu):
		cost += R[i] * (uk[i] - u_ref[i])**2
	return cost


def final_cost(xN, P: dict = None):
	cost = 0.0
	for i in range(nx):
		cost += Qf[i] * (xN[i] - x_ref[i])**2
	return cost


# =============================================================================
# Construct problem instance
# =============================================================================

# P = ProblemLMPC0(nx, nu)
# P = ProblemLMPC(nx, nu)
P = ProblemMPC(nx, nu)
P.N = N
P.dt = dt
P.stage_cost = stage_cost
P.final_cost = final_cost
P.dynamics_ct = dynamics_ct
P.input_constraints = U


# =============================================================================
# Construct solver
# =============================================================================

if isinstance(P, ProblemLMPC0):
	name += "_lmpc0"
elif isinstance(P, ProblemLMPC):
	name += "_lmpc"
S = Solver(problem=P, name=name, folder=folder)
S.substeps = substeps
S.initialize(build=build)
S.run(x0, steps)


# =============================================================================
# Plot
# =============================================================================

ss = S.state_sequence
uu = S.input_sequence
time = np.arange(0, dt * steps, dt)

plt.subplot(1, 3, 1)
plt.plot([time[0], time[-1]], [x_ref[0]] * 2, 'k--')
plt.plot(time, [float(x[0]) for x in ss[:-1]], '-')
plt.grid()
plt.title('Position')
plt.xlabel('Time')

plt.subplot(1, 3, 2)
plt.plot([time[0], time[-1]], [x_ref[1]] * 2, 'k--')
plt.plot(time, [float(x[1]) for x in ss[:-1]], '-')
plt.grid()
plt.title('Angle')
plt.xlabel('Time')

plt.subplot(1, 3, 3)
plt.plot([time[0], time[-1]], [u_ref[0]] * 2, 'k--')
plt.plot(time, [float(u[0]) for u in uu], '-', linewidth=2)
plt.title('Input')
plt.grid()

plt.show()
