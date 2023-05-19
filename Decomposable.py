import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cvxpy as cp

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
			   '#17becf']


def OPT_program(n):
	S = cp.Variable(n, nonneg=True)
	alpha = cp.Variable(nonneg=True)
	theta = cp.Variable(nonneg=True)
	payments = list(np.random.rand(n))
	payments.sort(reverse=True)
	R = [r * 1.5 for r in np.random.rand(n)]
	while not 0 < len([r for r in R if r > 1]) < n:
		R = [r * 1.5 for r in np.random.rand(n)]
	R.sort(reverse=True)
	ratios = [p / r for p, r in zip(payments, R)]
	print('payments:')
	print([round(p, 5) for p in payments])
	print('R:')
	print([round(r, 5) for r in R])
	print('p/R ratio:')
	print([round(r, 5) for r in ratios])
	# Primal
	obj_P = cp.Maximize(cp.sum([payments[i] * S[i] for i in range(n)]))
	constraints_P = [cp.sum(S) <= 1,
					 cp.sum([R[i] * S[i] for i in range(n)]) <= 1]
	prob_P = cp.Problem(obj_P, constraints_P)
	prob_P.solve()
	print('\nPrimal:')
	print("status:", prob_P.status)
	print("optimal value", round(prob_P.value, 5))
	print("S=", [round(s.value, 5) for s in S])

	# Dual
	obj_D = cp.Minimize(alpha + theta)
	constraints_D = [alpha + R[i] * theta >= payments[i] for i in range(n)]
	prob_D = cp.Problem(obj_D, constraints_D)
	prob_D.solve()
	print('\nDual:')
	print("status:", prob_D.status)
	print("optimal value", round(prob_D.value, 5))
	print("alpha=", round(alpha.value, 5), '\ntheta=', round(theta.value, 5))

	# Plot
	width = [3 if S[i].value > 0.00001 else 1.5 for i in range(n)]
	fig = plt.figure()
	ax1 = fig.add_subplot()
	for i in range(n):
		# Groups with R_i<1 in dashed lines
		if R[i] > 1:
			ax1.plot([0, payments[i]], [payments[i] / R[i], 0], label=f'group {i}', linewidth=width[i])
		else:
			ax1.plot([0, payments[i]], [payments[i] / R[i], 0], label=f'group {i}', linewidth=width[i], linestyle=':')

	ax1.scatter(alpha.value, theta.value, s=40, c='r', zorder=n)
	ax1.legend()
	plt.show()
	return


def main():
	OPT_program(n=8)
	return


if __name__ == '__main__':
	main()
