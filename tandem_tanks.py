#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:25:53 2025

@author: andthem
"""

import casadi.casadi as cs
import opengen as og
from Solver import ProblemMPC, ProblemLMPC, ProblemLMPC0, Solver

import numpy as np
import matplotlib.pyplot as plt

build = False
build = True

name = "tandem_tanks"
folder = "python_build"

# =============================================================================
# Define problem
# =============================================================================

# Parameters (no need to rebuild after changing these)
x0 = [65.0, 52.0]
steps = 2000
substeps = 10


# Problem data (need to rebuild after changing these)
nx = 2
nu = 1
dt = 15
N = 15

# Cost
x_ref = [56.0, 50.0]
u_ref = [10.6]

Q = [1.0, 1.0]
Qf = [50.0, 50.0]
R = [0.5]

# Constraints
U = og.constraints.Rectangle([8.0] * N, [11.0] * N)

# Dynamics
a1, a2 = 10 * 1e-4, 10 * 1.5e-4
A1, A2 = 2.5, 0.1
rho, g = 998, 9.8044


def dynamics_ct(x, u, P: dict = None):
	h1 = x[0]
	h2 = x[1]
	dx1 = u[0] / (rho * A1) - (a1 / A1) * cs.sqrt(2 * g * (h1 - h2))
	dx2 = (a1 * cs.sqrt(2 * g * (h1 - h2)) - a2 * cs.sqrt(h2)) / A2
	return cs.vertcat(dx1, dx2)


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

plt.subplot(1, 2, 1)
plt.plot([time[0], time[-1]], [x_ref[0]] * 2, 'k--')
plt.plot([time[0], time[-1]], [x_ref[1]] * 2, 'k--')
plt.plot(time, [float(x[0]) for x in ss[:-1]], '-', label="h1", linewidth=2)
plt.plot(time, [float(x[1]) for x in ss[:-1]], '-', label="h2", linewidth=2)
plt.grid()
plt.title('States')
plt.xlabel('Time')
plt.legend(bbox_to_anchor=(0.7, 0.85), loc='upper left', borderaxespad=0.)

plt.subplot(1, 2, 2)
plt.plot([time[0], time[-1]], [u_ref[0]] * 2, 'k--')
plt.plot(time, [float(u[0]) for u in uu], '-', linewidth=2)
plt.title('Input')
plt.grid()

plt.show()
