import numpy as np
import matplotlib.pyplot as plt

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


def two_group(b1=2 / 14, b2=0.8 / 14, gamma=1 / 14, p1=2, p2=1, U=0.2):
	phi1_range = np.arange(0.01, 1, 0.01)
	f1 = []
	f2 = []
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		f1.append(two_group_f1(b1, b2, gamma, phi1, phi2, p1, p2, U))
		f2.append(two_group_f2(b1, b2, gamma, phi1, phi2, p1, p2, U))
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1_range, f1, label='f1')
	ax1.plot(phi1_range, f2, label='f2')
	ax1.axhline(0, color='grey', linestyle=':')
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_xlim(0, 1)
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
	two_group()
	return


def main():
	decomposable()
	return


if __name__ == '__main__':
	main()
