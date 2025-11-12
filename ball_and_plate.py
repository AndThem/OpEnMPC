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

name = "ball_and_plate"
folder = "python_build"

# =============================================================================
# Define problem
# https://alphaville.github.io/optimization-engine/docs/example_bnp_py
# =============================================================================

# Parameters (no need to rebuild after changing these)
x0 = [0.1, -0.5, 0.0, 0.0]
substeps = 10
steps = 2000

# Problem data
nx = 4
nu = 1
N = 15
dt = 0.01

# Cost
Q = [5.0, 0.01, 0.01, 0.05]
R = [0.5]
Qf = [100.0, 20.0, 50.0, 0.8]
x_ref = [0.0] * nx
u_ref = [0.0] * nu

# Constraints
U = og.constraints.BallInf(None, 0.95)

# Dynamics
mass_ball = 1
moment_inertia = 0.0005
g = 9.8044


def dynamics_ct(x, u, P: dict = None):
	dx1 = x[1]
	dx2 = (5 / 7) * (x[0] * x[3]**2 - g * cs.sin(x[2]))
	dx3 = x[3]
	dx4 = (u[0] - mass_ball * g * x[0] * cs.cos(x[2])
		- 2 * mass_ball * x[0] * x[1] * x[3]) \
		/ (mass_ball * x[0]**2 + moment_inertia)
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
