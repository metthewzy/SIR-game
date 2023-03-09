import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform as uni
from matplotlib import cm
from scipy.optimize import minimize
import cvxpy as cp
from TwoGroup import final_size_searcher_binary


def three_group_social_dec(p, R):
	"""
	maximize the social utility of three group
	"""
	p1, p2, p3 = p
	R1, R2, R3 = R
	s1 = cp.Variable()
	s2 = cp.Variable()
	s3 = cp.Variable()
	phi1 = cp.Variable()
	phi2 = cp.Variable()
	phi3 = cp.Variable()
	constraints = [
		s1 <= phi1,
		s2 <= phi2,
		s3 <= phi3,
		s1 >= 0,
		s2 >= 0,
		s3 >= 0,
		R1 * s1 + R2 * s2 + R3 * s3 <= 1,
		phi1 + phi2 + phi3 <= 1,
		phi1 >= 0,
		phi2 >= 0,
		phi3 >= 0,
	]
	obj = cp.Maximize(p1 * s1 + p2 * s2 + p3 * s3)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	print("status:", prob.status)
	print("optimal value", prob.value)
	print("optimal var", s1.value, phi1.value, s2.value, phi2.value, s3.value, phi3.value)
	return


def main():
	p = [1, 0.9, 0.8]
	R = [2, 1.9, 1.8]
	three_group_social_dec(p, R)
	return


if __name__ == '__main__':
	main()
