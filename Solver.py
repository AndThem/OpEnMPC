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

	def __init__(self, nx, nu, np=None):
		self.nx = nx
		self.nu = nu
		self.np = nx if np is None else np

		for attr in self.__class__._attributes:
			setattr(self, "_" + attr, None)

		for attr in self.__class__._attributes_opt:
			setattr(self, "_" + attr, [])

	def dynamics_dt(self, xk: list, uk: list, P: dict = None):
		# P: parameter dictionary
		dx = self.dynamics_ct(xk, uk, P)
		return [xk[i] + self.dt * dx[i] for i in range(self.nx)]

	def cost(self, u: list, P: dict):
		tot = 0
		xk = P["x0"]
		for k in range(self.N):
			tot += self.stage_cost(xk, u[k], P)
			xk = self.dynamics_dt(xk, u[k], P)
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
		undef = [attr for attr in self.__class__._attributes if getattr(self, "_" + attr) is None]
		if undef:
			msg = f"Attributes '{', '.join(undef)}' have not been provided"
			raise NotImplementedError(msg)

	@property
	def dynamics_ct(self):
		return self._dynamics_ct

	@dynamics_ct.setter
	def dynamics_ct(self, func):
		self._dynamics_ct = func  # cs.Function('f', [x, u], [f])

	@property
	def all_attributes(self):
		return tuple(list(self.__class__._attributes)
			+ list(self.__class__._attributes_opt))

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

# class ProblemLMPC(Problem):

# 	def __init__(self, x_var, u_var, p_var):
# 		super().__init__(x_var, u_var, p_var)

# 	def dynamics_dt(self, x, u, x_, u_):
# 		f_ = self.dynamics_ct(x_, u_)
# 		A = self.Jxf(x_, u_)
# 		B = self.Juf(x_, u_)
# 		dx = cs.vertcat(*x) - cs.vertcat(*x_)
# 		du = cs.vertcat(*u) - cs.vertcat(*u_)
# 		return x + self.dt * (f_ + A @ dx + B @ du)
# 		#
# 		# TODO: make this more reliable and consistent
# 		# return x + self.dt * (f_ + A @ (x - x_) + B @ (u - u_))

# 	@Problem.dynamics_ct.setter
# 	def dynamics_ct(self, func):
# 		Problem.dynamics_ct.fset(self, func)
# 		x = self.x
# 		u = self.u
# 		f = func(x, u)
# 		self.Jxf = cs.Function('Jxf', [x, u], [cs.jacobian(f, x)])
# 		self.Juf = cs.Function('Juf', [x, u], [cs.jacobian(f, u)])


# =============================================================================
# Solver class
# =============================================================================

class Solver:

	_param_keys = ("x0",)

	def __init__(self, problem, name, folder="python_build"):
		problem.check()
		self.name = name
		self.folder = folder
		self.problem = problem

	def check(self):
		pass

	def initialize(self, build=True, tol=1e-5):
		self.check()
		self.solution = [0.0] * self.problem.nu * self.problem.N
		self.tol = tol
		if build:
			problem = self.make_problem()
			builder = self.make_builder(problem, tol=tol)
			builder.build()
		module = importlib.import_module(self.folder + f".{self.name}" * 2)
		self.solver = module.solver()

	def make_problem(self):
		# optimization parameter vector (initial state)
		vecp = cs.SX.sym("p", self.problem.np)
		# optimization variable (input sequence)
		vecu = cs.SX.sym("u_seq", self.problem.N * self.problem.nu)
		P = self.vector2parameters(vecp)
		u = self.vector2inputs(vecu)
		cost = self.problem.cost(u, P)
		# Bound input constraints
		input_constraints = self.problem.input_constraints
		problem = og.builder.Problem(vecu, vecp, cost) \
			.with_constraints(input_constraints)
		return problem

	def make_builder(self, problem, tol):
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
		builder = og.builder.OpEnOptimizerBuilder(problem,
			meta, build_config, solver_config)
		return builder

	def solve(self, debug=False):
		vecp = self.parameters_vector
		if debug:
			# TODO
			raise NotImplementedError
		nu = self.problem.nu
		init = self.solution[nu:] + self.solution[-nu:]
		out = self.solver.run(p=vecp, initial_guess=init)
		self.solution = out.solution
		return self.vector2inputs(self.solution)

	def run(self, x0, steps, debug=False):
		self.x0 = x0 + []
		self.state_sequence = [x0 + []]
		self.input_sequence = []
		x = x0
		nu = self.problem.nu
		for k in range(steps):
			# Print progress
			print('.', end='\n' if k % 50 == 49 else '')
			# Solve MPC problem
			self.x0 = x
			self.solve(debug=debug)
			u = self.solution[:nu]
			P = self.parameters
			x_next = self.problem.dynamics_dt(x, u, P)
			self.state_sequence.append(x_next)
			self.input_sequence.append(u)
			x = x_next

	# ==  Converters  =========================================================

	def vector2inputs(self, vecu):
		# Converts a concatenated input vector [u_0 ... u_{N-1}]
		# into a list of inputs [[u_0], ... [u_{N-1}]]
		N = self.problem.N
		nu = self.problem.nu
		u = []
		for k in range(N):
			u.append(vecu[k * nu:(k + 1) * nu])
		return u

	def vector2parameters(self, vecp):
		# Converts a concatenated parameter vector [p_1 ... p_r]
		# into a dictionary of parameters
		return {"x0": vecp}

	@property
	def parameters_vector(self):
		P = self.parameters
		vecp = []
		for param in self.param_keys:
			val = P[param]
			if type(val) is not list:
				val = [val]
			vecp.extend(val)
		return vecp

	@property
	def parameters(self):
		return {param: self.__getattribute__(param) for param in self.param_keys}

	@property
	def param_keys(self):
		return self._param_keys

# x = cs.SX.sym("x", 3)
# u = cs.SX.sym("u", 1)
# p = cs.SX.sym("p", 3)


# def f(x, u):
# 	f1 = cs.sin(x[0] * u[0])
# 	f2 = x[0] + 2 * u[0] + x[2]**2
# 	f3 = x[1] - u[0]
# 	return cs.vertcat(f1, f2, f3)


# P = Problem(x, u, p)
# P.dt = 0.1
# P.N = 4
# P.dynamics_ct = f

# print(P.dynamics_ct([1, 2, 0], 1))
# # print(P.dynamics_dt([1, 2, 0], [1]))
# print(P.dynamics_dt([1, 2, 0], [1]))
# print(P.dynamics_dt([1, 2, 0], [1]))

# PL = P.linearize()
# print(PL.Jxf([1, 2, 0], [1]))
# # print(P.dynamics_dt([1, 2, 0], [1]))
# print(PL.dynamics_dt([1, 2, 0], [1], [1, 2, 0], [1]))
# print(PL.dynamics_dt([1, 2, 0], [1], [0, 0, 0], [0]))
