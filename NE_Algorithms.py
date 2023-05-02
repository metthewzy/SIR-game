import numpy as np
import matplotlib.pyplot as plt
from ConvexProgram import two_group_utility_cvxpy
from matplotlib import cm
import cvxpy as cp

epsilon = 0.0001


def one_group_f(beta, gamma, phi, p, U):
	ret = U - (1 - epsilon) * p * np.exp(beta / gamma * phi * (U / p - 1))
	return ret


def two_group_f1(b1, b2, gamma, phi1, phi2, p1, p2, U):
	ret = U - (1 - epsilon) * p1 * np.exp(b1 / gamma * phi1 * (U / p1 - 1) + b2 / gamma * phi2 * (U / p2 - 1))
	return ret


def two_group_f2(b1, b2, gamma, phi1, phi2, p1, p2, U):
	ret = U - (1 - epsilon) * p2 * np.exp(b1 / gamma * phi1 * (U / p1 - 1) + b2 / gamma * phi2 * (U / p2 - 1))
	return ret


def two_group_f_plot(b=2 / 14, kappa=0.8, gamma=1 / 14, p1=1, p2=0.8,
					 # U=0.32765):
					 U=0.1):
	b11 = b
	b12 = b21 = kappa * b
	b22 = kappa * kappa * b
	# two_group_utility_cvxpy(b, gamma, epsilon, kappa, p2)
	phi1_range = np.arange(-1, 4, 0.01)
	f1 = []
	f2 = []
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		f1.append(two_group_f1(b11, b12, gamma, phi1, phi2, p1, p2, U))
		f2.append(two_group_f2(b21, b22, gamma, phi1, phi2, p1, p2, U))
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1_range, f1, label='f1')
	ax1.plot(phi1_range, f2, label='f2')
	ax1.axhline(0, color='grey', linestyle=':')
	ax1.set_xlabel(r'$\phi_1$')
	# ax1.set_xlim(0, 1)
	ax1.legend()
	plt.show()
	return


def two_group_f_3Dplot(b=2 / 14, kappa=0.2, gamma=1 / 14, p1=1, p2=0.9,
					   # U=0.32765):
					   U=0.2):
	b11 = b
	b12 = b21 = kappa * b
	b22 = kappa * kappa * b
	# two_group_utility_cvxpy(b, gamma, epsilon, kappa, p2)
	phi_range = np.arange(0, 1.01, 0.05)
	f1 = []
	f2 = []
	X = []
	Y = []
	for phi1 in phi_range:
		for phi2 in phi_range:
			X.append(phi1)
			Y.append(phi2)
			f1.append(two_group_f1(b11, b12, gamma, phi1, phi2, p1, p2, U))
			f2.append(two_group_f2(b21, b22, gamma, phi1, phi2, p1, p2, U))
	surface_max = [max(p1, p2) for p1, p2 in zip(f1, f2)]
	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')

	ax2 = fig.add_subplot(122, projection='3d')
	ax1.plot_trisurf(X, Y, f1, color='red')
	ax1.plot_trisurf(X, Y, f2, color='blue')
	ax2.plot_trisurf(X, Y, surface_max, cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	# ax2.set_xlabel(r'$\phi_1$')
	# ax2.set_ylabel(r'$\phi_2$')
	# ax1.legend()
	plt.show()
	return


def two_group_f1_phi1(b1, b2, gamma, phi1, phi2, p1, p2, U):
	"""
	Solve phi1 given phi2 in f1
	"""
	phi1 = (np.log(U / (1 - epsilon) / p1) -
			b2 / gamma * (U / p2 - 1) * phi2) \
		   / (b1 / gamma * (U / p1 - 1))
	return phi1


def two_group_f1_phi2(b1, b2, gamma, phi1, phi2, p1, p2, U):
	"""
	Solve phi2 given phi1 in f1
	"""
	phi2 = (np.log(U / (1 - epsilon) / p1) -
			b1 / gamma * (U / p1 - 1) * phi1) \
		   / (b2 / gamma * (U / p2 - 1))
	return phi2


def two_group_f2_phi2(b1, b2, gamma, phi1, phi2, p1, p2, U):
	"""
	Solve phi2 given phi1 in f2
	"""
	phi2 = (np.log(U / (1 - epsilon) / p2) -
			b1 / gamma * (U / p1 - 1) * phi1) \
		   / (b2 / gamma * (U / p2 - 1))
	return phi2


def two_group_f2_phi1(b1, b2, gamma, phi1, phi2, p1, p2, U):
	"""
	Solve phi1 given phi2 in f2
	"""
	phi1 = (np.log(U / (1 - epsilon) / p2) -
			b2 / gamma * (U / p2 - 1) * phi2) \
		   / (b1 / gamma * (U / p1 - 1))
	return phi1


def two_group_f1_binary(b1, b2, gamma, phi1, phi2, p1, p2, U):
	l = 0
	r = 1
	f_l = two_group_f1(b1, b2, gamma, phi1, l, p1, p2, U)
	f_r = two_group_f1(b1, b2, gamma, phi1, r, p1, p2, U)
	if f_l * f_r > 0:
		return -1
	for _ in range(40):
		m = (l + r) / 2
		f_m = two_group_f1(b1, b2, gamma, phi1, m, p1, p2, U)
		if f_m * f_r > 0:
			r = m
			f_r = f_m
		else:
			l = m
			f_l = f_m
	m = (l + r / 2)
	return m


def two_group_f2_binary(b1, b2, gamma, phi1, phi2, p1, p2, U):
	l = 0
	r = 1
	f_l = two_group_f2(b1, b2, gamma, l, phi2, p1, p2, U)
	f_r = two_group_f2(b1, b2, gamma, r, phi2, p1, p2, U)
	if f_l * f_r > 0:
		return -1
	for _ in range(40):
		m = (l + r) / 2
		f_m = two_group_f2(b1, b2, gamma, m, phi2, p1, p2, U)
		if f_m * f_r > 0:
			r = m
			f_r = f_m
		else:
			l = m
			f_l = f_m
	m = (l + r / 2)
	return m


def two_group_feasibility_binary(ax1, b=2 / 14, kappa=0.3, gamma=1 / 14, p1=1, p2=0.5,
								 U=0.754, c='r'):
	b11 = b
	b12 = b21 = kappa * b
	b22 = kappa * kappa * b
	phi_range = np.arange(0, 1.01, 0.01)

	phi1_f1 = []
	phi2_f1 = []
	for phi1 in phi_range:
		ret = two_group_f1_binary(b11, b12, gamma, phi1, 0, p1, p2, U)
		if ret >= 0:
			phi1_f1.append(phi1)
			phi2_f1.append(ret)

	phi1_f2 = []
	phi2_f2 = []
	for phi2 in phi_range:
		ret = two_group_f2_binary(b21, b22, gamma, 0, phi2, p1, p2, U)
		if ret >= 0:
			phi1_f2.append(ret)
			phi2_f2.append(phi2)

	# fig = plt.figure()
	# ax1 = fig.add_subplot()
	ax1.plot(phi1_f1, phi2_f1, label='f1', color=c)
	ax1.plot(phi1_f2, phi2_f2, label='f2', color=c, linestyle='dashed')
	return


def two_group_feasibility_linear(ax1, b=2 / 14, kappa=0.3, gamma=1 / 14, p1=1, p2=0.5,
								 U=0.754, alpha=1, c='r'):
	b11 = b
	b12 = b21 = kappa * b
	b22 = kappa * kappa * b
	phi_range = np.arange(0, 1.01, 0.01)

	phi1_f1 = []
	phi2_f1 = []
	for phi2 in [0, 1]:
		phi1 = two_group_f1_phi1(b11, b12, gamma, 0, phi2, p1, p2, U)
		phi1_f1.append(phi1)
		phi2_f1.append(phi2)
	for phi1 in [0, 1]:
		phi2 = two_group_f1_phi2(b11, b12, gamma, phi1, 0, p1, p2, U)
		phi1_f1.append(phi1)
		phi2_f1.append(phi2)

	phi1_f2 = []
	phi2_f2 = []
	for phi1 in [0, 1]:
		phi2 = two_group_f2_phi2(b21, b22, gamma, phi1, 0, p1, p2, U)
		phi1_f2.append(phi1)
		phi2_f2.append(phi2)
	for phi2 in [0, 1]:
		phi1 = two_group_f2_phi1(b21, b22, gamma, 0, phi2, p1, p2, U)
		phi1_f2.append(phi1)
		phi2_f2.append(phi2)

	# fig = plt.figure()
	# ax1 = fig.add_subplot()
	ax1.plot(phi1_f1, phi2_f1, label='f1', color=c, alpha=alpha)
	ax1.plot(phi1_f2, phi2_f2, label='f2', color=c, alpha=alpha, linestyle='dashed')
	return


def one_group(beta=2 / 14, gamma=1 / 14, p=2, U=1):
	phi_range = np.arange(0.01, 1, 0.01)
	f = []
	for phi in phi_range:
		f.append(one_group_f(beta, gamma, phi, p, U))
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi_range, f)
	ax1.axhline(0, color='grey', linestyle=':')
	plt.show()
	return


def two_group_feasibility_family_linear():
	Us = [0.2, 0.25, 0.3, 0.35, 0.4]
	# alphas = [1, 0.6, 0.4]
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

	for U, c in zip(Us, colors):
		fig = plt.figure()
		ax1 = fig.add_subplot()
		two_group_feasibility_linear(ax1, b=2 / 14, kappa=0.3, gamma=1 / 14, p1=1, p2=0.5, U=U, c=c)
		ax1.axhline(0, color='grey', linestyle=':')
		ax1.axhline(1, color='grey', linestyle=':')
		ax1.axvline(0, color='grey', linestyle=':')
		ax1.axvline(1, color='grey', linestyle=':')
		ax1.plot([0, 1], [1, 0], color='grey', linestyle=':')
		plt.show()
	return


def two_group_feasibility_family_binary():
	Us = [0.2, 0.25, 0.3, 0.35, 0.4]
	# alphas = [1, 0.6, 0.4]
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

	for U, c in zip(Us, colors):
		fig = plt.figure()
		ax1 = fig.add_subplot()
		two_group_feasibility_binary(ax1, b=2 / 14, kappa=0.3, gamma=1 / 14, p1=1, p2=0.5, U=U, c=c)
		ax1.axhline(0, color='grey', linestyle=':')
		ax1.axhline(1, color='grey', linestyle=':')
		ax1.axvline(0, color='grey', linestyle=':')
		ax1.axvline(1, color='grey', linestyle=':')
		ax1.plot([0, 1], [1, 0], color='grey', linestyle=':')
		plt.show()
	return


def three_group_cvxpy(U, p, b, kappa, gamma=1 / 14):
	"""
	minimization program for 3 group NE
	"""
	k1, k2, k3 = kappa
	p1, p2, p3 = p
	phi1 = cp.Variable()
	phi2 = cp.Variable()
	phi3 = cp.Variable()
	b11 = b * k1 * k1
	b12 = b * k1 * k2
	b13 = b * k1 * k3
	b21 = b * k2 * k1
	b22 = b * k2 * k2
	b23 = b * k2 * k3
	b31 = b * k3 * k1
	b32 = b * k3 * k2
	b33 = b * k3 * k3
	obj = cp.Minimize(phi1 + phi2 + phi3)
	constrainst = [phi1 >= 0,
				   phi2 >= 0,
				   phi3 >= 0,
				   phi1 + phi2 + phi3 >= 1,
				   U - (1 - epsilon) * p1 * cp.exp(b11 / gamma * phi1 * (U / p1 - 1) +
												   b12 / gamma * phi2 * (U / p2 - 1) +
												   b13 / gamma * phi3 * (U / p3 - 1)) >= 0,
				   U - (1 - epsilon) * p2 * cp.exp(b21 / gamma * phi1 * (U / p1 - 1) +
												   b22 / gamma * phi2 * (U / p2 - 1) +
												   b23 / gamma * phi3 * (U / p3 - 1)) >= 0,
				   U - (1 - epsilon) * p3 * cp.exp(b31 / gamma * phi1 * (U / p1 - 1) +
												   b32 / gamma * phi2 * (U / p2 - 1) +
												   b33 / gamma * phi3 * (U / p3 - 1)) >= 0
				   ]
	prob = cp.Problem(obj, constrainst)
	prob.solve()
	print('f1:')
	print(U - (1 - epsilon) * p1 * np.exp(b11 / gamma * phi1.value * (U / p1 - 1) +
										  b12 / gamma * phi2.value * (U / p2 - 1) +
										  b13 / gamma * phi3.value * (U / p3 - 1)
										  ))
	print('f2:')
	print(U - (1 - epsilon) * p2 * np.exp(b21 / gamma * phi1.value * (U / p1 - 1) +
										  b22 / gamma * phi2.value * (U / p2 - 1) +
										  b23 / gamma * phi3.value * (U / p3 - 1)
										  ))
	print('f3:')
	print(U - (1 - epsilon) * p3 * np.exp(b31 / gamma * phi1.value * (U / p1 - 1) +
										  b32 / gamma * phi2.value * (U / p2 - 1) +
										  b33 / gamma * phi3.value * (U / p3 - 1)
										  ))
	print("status:", prob.status)
	print("optimal value", prob.value)
	print("optimal var", phi1.value, phi2.value, phi3.value)
	return


def two_group_cvxpy(U, p, b, kappa, gamma=1 / 14):
	"""
	minimization program for 2 group NE
	"""
	k1, k2 = kappa
	p1, p2 = p
	phi1 = cp.Variable()
	phi2 = cp.Variable()
	b11 = b * k1 * k1
	b12 = b * k1 * k2
	b21 = b * k2 * k1
	b22 = b * k2 * k2

	obj = cp.Minimize(phi1 + phi2)
	constrainst = [phi1 >= 0,
				   phi2 >= 0,
				   # phi1 + phi2 <= 1,
				   U - (1 - epsilon) * p1 * cp.exp(b11 / gamma * phi1 * (U / p1 - 1) +
												   b12 / gamma * phi2 * (U / p2 - 1)) >= 0,
				   U - (1 - epsilon) * p2 * cp.exp(b21 / gamma * phi1 * (U / p1 - 1) +
												   b22 / gamma * phi2 * (U / p2 - 1)) >= 0
				   ]
	prob = cp.Problem(obj, constrainst)
	prob.solve()
	print('f1:')
	print(U - (1 - epsilon) * p1 * np.exp(b11 / gamma * phi1.value * (U / p1 - 1) +
										  b12 / gamma * phi2.value * (U / p2 - 1)))
	print('f2:')
	print(U - (1 - epsilon) * p2 * np.exp(b21 / gamma * phi1.value * (U / p1 - 1) +
										  b22 / gamma * phi2.value * (U / p2 - 1)))
	print("status:", prob.status)
	print("optimal value", prob.value)
	print("optimal var", phi1.value, phi2.value)
	return


def two_group_plot(U=0.48, p=[1, 0.6], b=3 / 14, kappa=[1, 0.3], gamma=1 / 14):
	p1, p2 = p
	k1, k2 = kappa
	b11 = b * k1 * k1
	b12 = b * k1 * k2
	b21 = b * k2 * k1
	b22 = b * k2 * k2
	phi_range = np.arange(0, 1.01, 0.025)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	f1 = []
	f2 = []
	for phi1 in phi_range:
		f1.append((np.log(U / (1 - epsilon) / p1) - b11 / gamma * (U / p1 - 1) * phi1) /
				  (b12 / gamma * (U / p2 - 1))
				  )
		f2.append((np.log(U / (1 - epsilon) / p2) - b21 / gamma * (U / p1 - 1) * phi1) /
				  (b22 / gamma * (U / p2 - 1))
				  )
	# ax1.scatter(phi1, phi2)
	# ax1.scatter(X, Y)
	ax1.plot([0, 1], [1, 0], c='red', linestyle=':',label='sum')
	ax1.plot(phi_range, f1, label='f1')
	ax1.plot(phi_range, f2, label='f2')
	# ax1.set_xlim(0, 1)
	# ax1.set_ylim(0, 1)
	ax1.legend()
	plt.show()
	return


def decomposable():
	# one_group()
	# two_group_f_plot()
	# two_group_f_3Dplot()
	# two_group_feasibility_family_linear()
	# two_group_feasibility_family_binary()
	# three_group_cvxpy(U=0.42, p=[1, 0.6, 0.5], b=3 / 14, kappa=[1, 0.3, 0.2])
	# two_group_cvxpy(U=0.48, p=[1, 0.6], b=3 / 14, kappa=[1, 0.3])
	two_group_plot()
	return


def main():
	decomposable()
	return


if __name__ == '__main__':
	main()
