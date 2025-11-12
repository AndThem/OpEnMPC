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

class ProblemMPC:

	# Attributes to be provided
	_attributes = ('N', 'dt', 'stage_cost', 'final_cost', 'dynamics_ct')
	# Optional attributes
	_attributes_opt = ('input_constraints', 'state_constraints')
	# Parameter names for optimization problems
	_param_keys = ("x0", )

	def __init__(self, nx: int, nu: int):
		self.nx = nx  # number of states
		self.nu = nu  # number of inputs
		self.np = nx  # number of parameters (only x0)
		self.__parameters = dict()

		for attr in self.__class__._attributes:
			setattr(self, "_" + attr, None)

		for attr in self.__class__._attributes_opt:
			setattr(self, "_" + attr, [])

	def __getitem__(self, key: str):
		return self.__parameters[key]

	def __setitem__(self, key: str, val: cs.DM | list):
		if key not in self._param_keys:
			msg = f"only keys in {self._param_keys} allowed (got {key} instead)"
			raise KeyError(msg)
		self.__parameters[key] = val

	def initialize(self, x0: cs.DM):
		self["x0"] = x0 if type(x0) is cs.DM else cs.vertcat(*x0)

	def dynamics_dt(self, xk: cs.DM, uk: cs.DM, k: int = None, P: dict = None) -> cs.DM:
		# k: stage
		# P: parameter dictionary
		return self.x_next(xk=xk, uk=uk, P=P, substeps=1)

	def cost(self, u_seq: list, P: dict) -> cs.DM:
		if P is None:
			P = dict()
		xk = P["x0"] if "x0" in P else self["x0"]
		tot = 0.0
		for k in range(self.N):
			uk = u_seq[k]
			tot += self.stage_cost(xk=xk, uk=uk, k=k, P=P)
			xk = self.dynamics_dt(xk=xk, uk=uk, k=k, P=P)
		tot += self.final_cost(xN=xk, P=P)
		return tot

	def linearize(self):
		P = ProblemLMPC(nx=self.nx, nu=self.nu)
		for attr in self._attributes + self._attributes_opt:
			val = getattr(self, '_' + attr)
			if val is not None:
				setattr(P, attr, val)
		return P

	def check(self):
		undef = [
			attr for attr in self.__class__._attributes
			if getattr(self, "_" + attr) is None]
		if undef:
			msg = f"Attributes '{', '.join(undef)}' have not been provided"
			raise NotImplementedError(msg)

	def x_next(self, xk: cs.DM, uk: cs.DM, P: dict = None, substeps=1) -> cs.DM:
		dt = self.dt / substeps
		for j in range(substeps):
			dx = self.dynamics_ct(x=xk, u=uk, P=P)
			xk = xk + dt * dx
		return xk

	def update_parameters(self, u_seq: list, substeps=1):
		x0 = self["x0"]
		u0 = u_seq[0]
		P = self.parameters
		self["x0"] = self.x_next(x0, u0, P, substeps=substeps)

	@classmethod
	def parameters2vector(cls, P: dict) -> list:
		"""Converts dictionary of paramters to a list"""
		vecp = []
		for param in cls._param_keys:
			val = P[param]
			vecp = cls.extend(vecp, val)
		return vecp

	@staticmethod
	def extend(x: list, val: list | float) -> list:
		if type(val) is list:
			for val_i in val:
				x = ProblemMPC.extend(x, val_i)
			return x
		return x + cs.vertsplit(val)

	def vector2parameters(self, vecp: list) -> dict:
		"""Converts a concatenated parameter vector [p_1 ... p_r]
		into a dictionary of parameters"""
		return {"x0": vecp}

	def vector2inputs(self, vecu: cs.DM) -> list:
		"""Converts a concatenated input vector [u_0 ... u_{N-1}]
		into a list of inputs [[u_0], ... [u_{N-1}]]"""
		U = []
		for k in range(self.N):
			U.append(vecu[k * self.nu:(k + 1) * self.nu])
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
# ProblemLMPC0 class
# =============================================================================

class ProblemLMPC0(ProblemMPC):

	# Parameter names for optimization problems
	_param_keys = ("x0", "u0")

	def __init__(self, nx: int, nu: int):
		super().__init__(nx, nu)
		self.np = nx + nu

	def initialize(self, x0: cs.DM):
		super().initialize(x0=x0)
		u0 = [0.0] * self.nu
		self["u0"] = cs.vertcat(*u0)

	def dynamics_dt(self, xk: cs.DM, uk: cs.DM, k: int = None, P: dict = None) -> cs.DM:
		# k: stage
		# P: parameter dictionary
		x_ = P["x0"]
		u_ = P["u0"]
		A_ = self.JxF(x_, u_)
		B_ = self.JuF(x_, u_)
		f_ = self.dynamics_ct(x=x_, u=u_, P=P)
		dx = xk - x_
		du = uk - u_
		return xk + self.dt * (f_ + A_ @ dx + B_ @ du)

	def update_parameters(self, u_seq: list, substeps=1):
		x0 = self["x0"]
		u0 = u_seq[0]
		P = self.parameters
		x1 = self.x_next(xk=x0, uk=u0, P=P, substeps=substeps)
		self["x0"] = x1
		self["u0"] = u_seq[1]

	def vector2parameters(self, vecp: list) -> dict:
		"""Converts a concatenated parameter vector [p_1 ... p_r]
		into a dictionary of parameters
		vecp is a vector of the form
		[x0 = x1_prev, u0 = u1_prev]"""
		P = dict()
		P["x0"] = vecp[:self.nx]
		P["u0"] = vecp[self.nx:]
		return P


# =============================================================================
# ProblemLMPC class
# =============================================================================

class ProblemLMPC(ProblemMPC):

	# Parameter names for optimization problems
	_param_keys = ("x0", "x_seq", "u_seq")

	def __init__(self, nx: int, nu: int):
		super().__init__(nx, nu)

	def initialize(self, x0: cs.DM):
		super().initialize(x0=x0)
		u0 = [0.0] * self.nu
		u_seq = [cs.vertcat(*u0) for k in range(self.N - 1)]
		self["u_seq"] = u_seq
		self["x_seq"] = self.x_sequence(x0=self["x0"], u_seq=u_seq)

	def dynamics_dt(self, xk: cs.DM, uk: cs.DM, k: int, P: dict = None) -> cs.DM:
		# k: stage
		# P: parameter dictionary
		x_ = P["x_seq"][k]
		u_ = P["u_seq"][min(k, self.N - 2)]
		A_ = self.JxF(x_, u_)
		B_ = self.JuF(x_, u_)
		f_ = self.dynamics_ct(x=x_, u=u_, P=P)
		dx = xk - x_
		du = uk - u_
		return xk + self.dt * (f_ + A_ @ dx + B_ @ du)

	def x_sequence(self, x0: cs.DM, u_seq: list, substeps=1):
		P = self.parameters
		x_seq = [x0]
		xk = x0
		for uk in u_seq:
			xk = self.x_next(xk=xk, uk=uk, P=P, substeps=1)
			x_seq.append(xk)
		return x_seq

	def update_parameters(self, u_seq: list, substeps=1):
		x0 = self["x0"]
		u0 = u_seq[0]
		P = self.parameters
		xk = self.x_next(xk=x0, uk=u0, P=P, substeps=substeps)
		self["x0"] = xk
		self["u_seq"] = u_seq[1:]
		self["x_seq"] = self.x_sequence(x0=xk, u_seq=self["u_seq"], substeps=1)

	def vector2parameters(self, vecp: list) -> dict:
		"""Converts a concatenated parameter vector [p_1 ... p_r]
		into a dictionary of parameters
		vecp is a vector of the form
		[x0 = x1_prev,
		x0_ = x0, x1_ = x2_prev, ... x(N-1)_ = xN_prev,
		u0_ = u1_prev, ... u(N-2)_ = u(N-1)_prev]
		where xk_, uk_ are the points of the linearization at step k"""
		P = dict()
		N = self.N
		nx = self.nx
		nu = self.nu
		P["x0"] = vecp[:nx]
		P["x_seq"] = [P["x0"]]
		P["u_seq"] = []
		index_u = (N + 1) * nx
		for k in range(N - 1):
			P["x_seq"].append(vecp[(k + 1) * nx:(k + 2) * nx])
			P["u_seq"].append(vecp[index_u + k * nu:index_u + (k + 1) * nu])
		return P

	@ProblemMPC.N.setter
	def N(self, N: int):
		self._N = N
		# update number of parameters ([x0, x0_, x1_ ... xN_, u0_ ... u(N-2)_])
		self.np = self.nx * (N + 1) + self.nu * (N - 1)


# =============================================================================
# Solver class
# =============================================================================

class Solver:

	def __init__(self, problem: ProblemMPC, name: str, folder="python_build"):
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
		problem = og.builder.Problem(vecu, vecp, cost)
		input_constraints = self.problem.input_constraints
		if input_constraints:
			problem = problem.with_constraints(input_constraints)
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

	def do_one_step(self, debug=False) -> list:
		if debug:
			# TODO
			raise NotImplementedError
		nu = self.problem.nu
		vecp = self.problem.parameters_vector
		init = self.solution[nu:] + self.solution[-nu:]
		out = self.solver.run(p=vecp, initial_guess=init)
		self.solution = out.solution
		u_seq = self.problem.vector2inputs(self.solution)
		return u_seq

	def run(self, x0: cs.DM | list, steps: int, debug=False):
		self.state_sequence = [x0]
		self.input_sequence = []
		self.problem.initialize(x0=x0)
		for k in range(steps):
			# Print progress
			if k % 10 == 9:
				print('.', end='\n' if k % 500 == 499 else '')
			# Solve MPC problem
			u_seq = self.do_one_step(debug=debug)
			# Apply input u0
			self.problem.update_parameters(u_seq, substeps=self.substeps)
			# Update state x0 in solver
			x0 = self.problem["x0"]
			# Record data
			self.state_sequence.append(x0)
			self.input_sequence.append(u_seq[0])


# =============================================================================
# Simulation class
# =============================================================================

# class Simulation:

# 	def __init__(self, solver: Solver):
# 		self.solver = solver

# 	def initialize_plots(self):
# 		pass

# 	def update_plots(self):
# 		pass

# 	def run(self, x0: cs.DM | list, steps: int, debug=False):
# 		self.state_sequence = [x0]
# 		self.input_sequence = []
# 		self.problem.initialize(x0=x0)
# 		self.initialize_plots()
# 		for k in range(steps):
# 			# Print progress
# 			print('.', end='\n' if k % 50 == 49 else '')
# 			# Solve MPC problem
# 			vecu = self.solver.do_one_step(debug=debug)
# 			u_seq = self.problem.vector2inputs(vecu)
# 			# Apply input u0
# 			self.problem.update_parameters(u_seq, substeps=self.substeps)
# 			# Update state x0 in solver
# 			x0 = self.problem["x0"]
# 			# Record data
# 			self.state_sequence.append(x0)
# 			self.input_sequence.append(u_seq[0])
# 			self.update_plots()

# 	@property
# 	def problem(self):
# 		return self.solver.problem
