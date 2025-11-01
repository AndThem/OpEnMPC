#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:38:39 2025

@author: andthem
"""

import casadi.casadi as cs
import opengen as og
import importlib


# =============================================================================
# Problem class
# =============================================================================

class Problem:

	# Attributes to be provided
	_attributes = ('N', 'dt', 'stage_cost', 'final_cost', 'dynamics_ct')
	# Optional attributes
	_attributes_opt = ('input_constraints', 'state_constraints')
	# Parameter names for optimization problems
	_param_keys = ("x0", )

	def __init__(self, nx, nu, np=None):
		self.nx = nx
		self.nu = nu
		self.np = nx if np is None else np
		self.__parameters = dict()

		for attr in self.__class__._attributes:
			setattr(self, "_" + attr, None)

		for attr in self.__class__._attributes_opt:
			setattr(self, "_" + attr, [])

	def __getitem__(self, key):
		return self.__parameters[key]

	def __setitem__(self, key, val):
		if key not in self._param_keys:
			msg = f"only keys in {self._param_keys} allowed (got {key} instead)"
			raise KeyError(msg)
		self.__parameters[key] = val

	def dynamics_dt(self, xk: cs.DM, uk: cs.DM, P: dict = None) -> cs.DM:
		# P: parameter dictionary
		return self.x_next(xk, uk, P, substeps=1)

	def cost(self, u_seq: list, P: dict) -> cs.DM:
		if P is None:
			P = dict()
		xk = P["x0"] if "x0" in P else self["x0"]
		tot = 0.0
		for k in range(self.N):
			uk = u_seq[k]
			tot += self.stage_cost(xk, uk, P)
			xk = self.dynamics_dt(xk, uk, P)
		tot += self.final_cost(xk, P)
		return tot

# 	def linearize(self):
# 		P = ProblemLMPC(self.x, self.u, self.p)
# 		for attr in self.all_attributes:
# 			val = getattr(self, '_' + attr)
# 			if val is not None:
# 				setattr(P, attr, val)
# 		return P

	def check(self):
		undef = [
			attr for attr in self.__class__._attributes
			if getattr(self, "_" + attr) is None]
		if undef:
			msg = f"Attributes '{', '.join(undef)}' have not been provided"
			raise NotImplementedError(msg)

	def x_next(self, xk: cs.DM, uk: cs.DM, P: dict = None, substeps=10) -> cs.DM:
		dt = self.dt / substeps
		for j in range(substeps):
			dx = self.dynamics_ct(xk, uk, P)
			xk = xk + dt * dx
		return xk

	def update_parameters(self, u_seq: list, substeps=10):
		x0 = self["x0"]
		u0 = u_seq[0]
		P = self.parameters
		self["x0"] = self.x_next(x0, u0, P, substeps=substeps)

	@classmethod
	def parameters2vector(cls, P: dict) -> list:
		# converts dictionary of paramters to a list
		vecp = []
		for param in cls._param_keys:
			val = P[param]
			vecp += cs.vertsplit(val)
		return vecp

	@classmethod
	def vector2parameters(cls, vecp: list) -> dict:
		# Converts a concatenated parameter vector [p_1 ... p_r]
		# into a dictionary of parameters
		return {"x0": vecp}

	def vector2inputs(self, vecu: cs.DM) -> list:
		# Converts a concatenated input vector [u_0 ... u_{N-1}]
		# into a list of inputs [[u_0], ... [u_{N-1}]]
		N = self.N
		nu = self.nu
		U = []
		for k in range(N):
			U.append(vecu[k * nu:(k + 1) * nu])
		return U

	@property
	def parameters(self):
		return self.__parameters

	@property
	def parameters_vector(self):
		return self.parameters2vector(self.__parameters)

	@property
	def dynamics_ct(self):
		return self._dynamics_ct

	@dynamics_ct.setter
	def dynamics_ct(self, func):
		self._dynamics_ct = func

	@property
	def N(self):
		return self._N

	@N.setter
	def N(self, N):
		self._N = N

	@property
	def dt(self):
		return self._dt

	@dt.setter
	def dt(self, dt):
		self._dt = dt

	@property
	def stage_cost(self):
		return self._stage_cost

	@stage_cost.setter
	def stage_cost(self, func):
		self._stage_cost = func

	@property
	def final_cost(self):
		return self._final_cost

	@final_cost.setter
	def final_cost(self, func):
		self._final_cost = func

	@property
	def state_constraints(self):
		return self._state_constraints

	@state_constraints.setter
	def state_constraints(self, obj):
		self._state_constraints = obj

	@property
	def input_constraints(self):
		return self._input_constraints

	@input_constraints.setter
	def input_constraints(self, obj):
		self._input_constraints = obj


# =============================================================================
# ProblemLMPC class
# =============================================================================

class ProblemLMPC(Problem):

	def __init__(self, x_var, u_var, p_var):
		super().__init__(x_var, u_var, p_var)

	def dynamics_dt(self, xk: cs.DM, uk: cs.DM, P: dict = None) -> cs.DM:
		# P: parameter dictionary
		x_ = P["x_"]
		u_ = P["u_"]
		A = self.Jxf(x_, u_)
		B = self.Juf(x_, u_)
		f_ = self.dynamics_ct(xk, uk, P)
		dx = xk - x_
		du = uk - u_
		return xk + self.dt * (f_ + A @ dx + B @ du)

	@Problem.dynamics_ct.setter
	def dynamics_ct(self, func):
		self._dynamics_ct = func
		x = cs.SX.sym("x", self.nx)
		u = cs.SX.sym("u", self.nu)
		self._JxF = cs.Function('JxF', [x, u], [cs.jacobian(func(x, u), x)])
		self._JuF = cs.Function('JuF', [x, u], [cs.jacobian(func(x, u), u)])

	@property
	def JxF(self):
		return self._JxF

	@property
	def JuF(self):
		return self._JuF


# =============================================================================
# Solver class
# =============================================================================

class Solver:

	def __init__(self, problem: Problem, name: str, folder="python_build"):
		self.name = name
		self.folder = folder
		self.problem = problem
		self.substeps = 1  # for updating the state after every MPC solution

	def check(self):
		self.problem.check()

	def initialize(self, build=True, tol=1e-6):
		self.check()
		self.solution = [0.0] * self.problem.nu * self.problem.N
		self.tol = tol
		if build:
			problem = self.make_problem()
			builder = self.make_builder(problem, tol=tol)
			builder.build()
		module = importlib.import_module(self.folder + f".{self.name}" * 2)
		self.solver = module.solver()

	def make_problem(self) -> og.builder.problem.Problem:
		# optimization variable (input sequence)
		vecu = cs.SX.sym("u_seq", self.problem.N * self.problem.nu)
		U = self.problem.vector2inputs(vecu)
		# optimization parameter vector (initial state, ...)
		vecp = cs.SX.sym("p", self.problem.np)
		P = self.problem.vector2parameters(vecp)
		# construct problem
		cost = self.problem.cost(U, P)
		input_constraints = self.problem.input_constraints
		problem = og.builder.Problem(vecu, vecp, cost) \
			.with_constraints(input_constraints)
		return problem

	def make_builder(self, problem: og.builder.problem.Problem, tol: float):
		mode = "release"
		build_config = og.config.BuildConfiguration() \
			.with_build_directory(self.folder) \
			.with_build_mode(mode) \
			.with_build_python_bindings()
		meta = og.config.OptimizerMeta() \
			.with_optimizer_name(self.name)
		solver_config = og.config.SolverConfiguration() \
			.with_tolerance(tol) \
			.with_initial_tolerance(tol)
		builder = og.builder.OpEnOptimizerBuilder(
			problem,
			meta, build_config, solver_config)
		return builder

	def do_one_step(self, debug=False):
		if debug:
			# TODO
			raise NotImplementedError
		nu = self.problem.nu
		vecp = self.problem.parameters_vector
		init = self.solution[nu:] + self.solution[-nu:]
		out = self.solver.run(p=vecp, initial_guess=init)
		self.solution = out.solution
		return self.problem.vector2inputs(self.solution)

	def run(self, x0: cs.DM | list, steps: int, debug=False):
		self.state_sequence = [x0]
		self.input_sequence = []
		self.problem["x0"] = x0 if type(x0) is cs.DM else cs.vertcat(*x0)
		for k in range(steps):
			# Print progress
			print('.', end='\n' if k % 50 == 49 else '')
			# Solve MPC problem
			u_seq = self.do_one_step(debug=debug)
			# Apply input u0
			self.problem.update_parameters(u_seq, substeps=self.substeps)
			# Update state x0 in solver
			x0 = self.problem["x0"]
			# Record data
			self.state_sequence.append(x0)
			self.input_sequence.append(u_seq[0])
