#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:25:53 2025

@author: andthem
"""

import casadi.casadi as cs
import opengen as og
from Solver import ProblemMPC, ProblemLMPC, ProblemLMPC0, Solver, Simulation

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
Q = [10.0, 10.0]
R = [0.5]
Qf = [50.0, 50.0]
x_ref = [56.0, 50.0]
u_ref = [10.6]

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


# =============================================================================
# Construct problem instance
# https://alphaville.github.io/optimization-engine/docs/example_tanks_py
# =============================================================================

# P = ProblemLMPC0(nx, nu)
# P = ProblemLMPC(nx, nu)
P = ProblemMPC(nx, nu)
P.x_labels = ["h1", "h2"]
P.N = N
P.dt = dt
P.x_ref = x_ref
P.u_ref = u_ref
P.dynamics_ct = dynamics_ct
P.input_constraints = U

P.set_quadratic_stage_cost(Q=Q, R=R)
P.set_quadratic_final_cost(Qf=Qf)


# =============================================================================
# Construct solver and run simulation
# =============================================================================

S = Solver(problem=P, name=name, folder=folder)
S.initialize(build=build)

simulation = Simulation(S)
simulation.substeps = substeps
simulation.run(x0, steps)


# =============================================================================
# Plot
# =============================================================================

plt.subplot(1, 2, 1)
plt.title('States')
simulation.plot_sequence(states=[0, 1])
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Input')
simulation.plot_sequence(inputs=[0])

plt.show()
