import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint


def OBJ(S):
	"""
	objective function
	"""
	ret = sum(S)
	return ret


def f_i(n, beta_M, gamma, epsilon, phi_V, S, i):
	"""
	constraint function f_i()
	"""
	X = sum([beta_M[i][j] / gamma * (S[j] - phi_V[j])
			 for j in range(n)])
	ret = S[i] - (1 - epsilon) * phi_V[i] * np.exp(X)
	return ret


def min_program(n, beta, gamma, epsilon, phi_V):
	bounds = Bounds([(0, (1 - epsilon) * phi) for phi in phi_V])
	constraints = NonlinearConstraint()
	return


def main():
	return


if __name__ == '__main__':
	main()
