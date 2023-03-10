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
	print(round(float(s1.value), 5))
	print(round(float(s2.value), 5))
	print(round(float(s3.value), 5))
	print(round(float(s4.value), 5))
	print('sum S_i:', round(s1.value + s2.value + s3.value + s4.value, 5))
	print('sum R_i*S_i:', round(R1 * s1.value + R2 * s2.value + R3 * s3.value + R4 * s4.value, 5))
	return


def three_group():
	p = [1.01, 0.5 * 2, 0.6]
	R = [2.02, 0.95 * 2, 0.9]
	print([pi / Ri for pi, Ri in zip(p, R)])
	three_group_social_primal(p, R)
	return


def four_group():
	R = [2, 1.95, 1.1, 0.4]
	ratios = [1, 1, 1.05, 1.1]
	p = [r * ratio for r, ratio in zip(R, ratios)]
	print(ratios)
	four_group_social_primal(p, R)
	return


def main():
	# three_group()
	four_group()
	return


if __name__ == '__main__':
	main()
