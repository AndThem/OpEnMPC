#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 17:51:17 2025

@author: andthem
"""

import casadi.casadi as cs
import opengen as og
from Solver import ProblemMPC, ProblemLMPC, ProblemLMPC0, Solver, Simulation

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
Q = [2.5, 50.0, 0.01, 0.01]
R = [0.1]
Qf = [3.0, 50.0, 0.02, 0.02]
x_ref = [0.0] * nx
u_ref = [0.0] * nu

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


# =============================================================================
# Construct problem instance
# =============================================================================

# P = ProblemLMPC0(nx, nu)
# P = ProblemLMPC(nx, nu)
P = ProblemMPC(nx, nu)
P.x_labels = ["Position", "Angle", "Velocity", "Angular velocity"]
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

plt.subplot(1, 3, 1)
plt.title('Position')
simulation.plot_sequence(states=[0], show_labels=False)

plt.subplot(1, 3, 2)
plt.title('Angle')
simulation.plot_sequence(states=[1], show_labels=False)

plt.subplot(1, 3, 3)
plt.title('Input')
simulation.plot_sequence(inputs=[0], show_labels=False)

plt.show()
