import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform as uni
from matplotlib import cm
from scipy.optimize import minimize
import cvxpy as cp
from TwoGroup import final_size_searcher_binary


def three_group_social_primal(p, R):
	"""
	maximize the social utility of three group
	"""
	p1, p2, p3 = p
	R1, R2, R3 = R
	s1 = cp.Variable()
	s2 = cp.Variable()
	s3 = cp.Variable()
	# phi1 = cp.Variable()
	# phi2 = cp.Variable()
	# phi3 = cp.Variable()
	constraints = [
		# s1 <= phi1,
		# s2 <= phi2,
		# s3 <= phi3,
		s1 >= 0,
		s2 >= 0,
		s3 >= 0,
		R1 * s1 + R2 * s2 + R3 * s3 <= 1,
		s1 + s2 + s3 <= 1
		# phi1 + phi2 + phi3 <= 1
	]
	obj = cp.Maximize(p1 * s1 + p2 * s2 + p3 * s3)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	# print("status:", prob.status)
	# print("optimal value", prob.value)
	# print("optimal var")
	print(round(float(s1.value), 5))
	print(round(float(s2.value), 5))
	print(round(float(s3.value), 5))
	print('sum S_i:', round(s1.value + s2.value + s3.value, 5))
	print('sum R_i*S_i:', round(R1 * s1.value + R2 * s2.value + R3 * s3.value, 5))
	return


def four_group_social_primal(p, R):
	"""
	maximize the social utility of four group
	"""
	p1, p2, p3, p4 = p
	R1, R2, R3, R4 = R
	s1 = cp.Variable()
	s2 = cp.Variable()
	s3 = cp.Variable()
	s4 = cp.Variable()
	constraints = [
		s1 >= 0,
		s2 >= 0,
		s3 >= 0,
		s4 >= 0,
		R1 * s1 + R2 * s2 + R3 * s3 + R4 * s4 <= 1,
		s1 + s2 + s3 + s4 <= 1
	]
	obj = cp.Maximize(p1 * s1 + p2 * s2 + p3 * s3 + p4 * s4)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	# print("status:", prob.status)
	print("primal optimal value:", round(float(prob.value), 5))
	# print("optimal var")
	print('S_1:\t\t', round(float(s1.value), 5))
	print('S_2:\t\t', round(float(s2.value), 5))
	print('S_3:\t\t', round(float(s3.value), 5))
	print('S_4:\t\t', round(float(s4.value), 5))
	print('sum R_i*S_:\t', round(R1 * s1.value + R2 * s2.value + R3 * s3.value + R4 * s4.value, 5))
	print('sum S_i:\t', round(s1.value + s2.value + s3.value + s4.value, 5))
	return


def four_group_social_dual(p, R):
	"""
	dual program of social utility of four groups
	"""
	p1, p2, p3, p4 = p
	R1, R2, R3, R4 = R
	a1 = cp.Variable()
	a2 = cp.Variable()
	a3 = cp.Variable()
	a4 = cp.Variable()
	l = cp.Variable()
	d = cp.Variable()
	constraints = [
		a1 >= 0,
		a2 >= 0,
		a3 >= 0,
		a4 >= 0,
		l >= 0,
		d >= 0,
		R1 * l - a1 + d >= p1,
		R2 * l - a2 + d >= p2,
		R3 * l - a3 + d >= p3,
		R4 * l - a4 + d >= p4,
	]
	obj = cp.Minimize(l + d)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	print("dual optimal value:", round(float(prob.value), 5))
	print('lambda:\t\t', round(float(l.value), 5))
	print('delta:\t\t', round(float(d.value), 5))
	print('a1:\t\t\t', round(float(a1.value), 5))
	print('a2:\t\t\t', round(float(a2.value), 5))
	print('a3:\t\t\t', round(float(a3.value), 5))
	print('a4:\t\t\t', round(float(a4.value), 5))

	print('c1:\t\t\t', round(R1 * l.value - a1.value + d.value - p1, 5))
	print('c2:\t\t\t', round(R2 * l.value - a2.value + d.value - p2, 5))
	print('c3:\t\t\t', round(R3 * l.value - a3.value + d.value - p3, 5))
	print('c4:\t\t\t', round(R4 * l.value - a4.value + d.value - p4, 5))
	return


def three_group():
	p = [1.01, 0.5 * 2, 0.6]
	R = [2.02, 0.95 * 2, 0.9]
	print([pi / Ri for pi, Ri in zip(p, R)])
	three_group_social_primal(p, R)
	return


def four_group():
	R0 = 0.6
	R = [4, 1.6, 1.1, 1]
	R = [r * R0 for r in R]
	ratios = [1, 1.2, 1.35, 1.4]
	p = [r * ratio for r, ratio in zip(R, ratios)]
	print('payments:\n', [round(pi, 5) for pi in p])
	print('R0:\n', [round(R0, 5) for R0 in R])
	print('ratios:\n', [round(r, 5) for r in ratios])
	four_group_social_primal(p, R)
	print()
	four_group_social_dual(p, R)
	return


def main():
	# three_group()
	four_group()
	return


if __name__ == '__main__':
	main()
