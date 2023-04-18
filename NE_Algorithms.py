import numpy as np
import matplotlib.pyplot as plt
from ConvexProgram import two_group_utility_cvxpy

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


def two_group_f1_binary(b1, b2, gamma, phi1, phi2, p1, p2, U):
	l = 0
	r = 1
	f_l = two_group_f1(b1, b2, gamma, phi1, l, p1, p2, U)
	f_r = two_group_f1(b1, b2, gamma, phi1, r, p1, p2, U)
	if f_l * f_r > 0:
		return -1
	for _ in range(20):
		m = (l + r) / 2
		print(m)
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
	for _ in range(20):
		m = (l + r) / 2
		print(m)
		f_m = two_group_f2(b1, b2, gamma, m, phi2, p1, p2, U)
		if f_m * f_r > 0:
			r = m
			f_r = f_m
		else:
			l = m
			f_l = f_m
	m = (l + r / 2)
	return m


def two_group_binary(b=2 / 14, kappa=0.8, gamma=1 / 14, p1=1, p2=0.8,
					 U=0.32):
	b11 = b
	b12 = b21 = kappa * b
	b22 = kappa * kappa * b
	phi_range = np.arange(0.01, 1, 0.01)

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

	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1_f1, phi2_f1, label='f1')
	ax1.plot(phi1_f2, phi2_f2, label='f2')
	ax1.axhline(0, color='grey', linestyle=':')
	ax1.axhline(1, color='grey', linestyle=':')
	ax1.axvline(0, color='grey', linestyle=':')
	ax1.axvline(1, color='grey', linestyle=':')
	ax1.legend()
	plt.show()
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


def decomposable():
	# one_group()
	# two_group_f_plot()
	two_group_binary()
	return


def main():
	decomposable()
	return


if __name__ == '__main__':
	main()
