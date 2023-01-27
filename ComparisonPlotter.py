import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from TwoGroup import final_size_searcher_binary

binary_iterations = 100
# OPT_iterations = 30
# NE_iterations = 50


def zero_searcher(f, left, right, beta, gamma, epsilon, phi):
	mid = (left + right) / 2
	for _ in range(binary_iterations):
		if f(mid, beta, gamma, epsilon, phi) < 0:
			left = mid
		else:
			right = mid
		mid = (left + right) / 2
	return mid


def g(S_bar, beta, gamma, epsilon, phi):
	ret = S_bar - (1 - epsilon) * np.exp(beta / gamma * phi * (S_bar - 1))
	return ret


def g_peak(beta, gamma, epsilon, phi):
	ret = gamma / beta / phi * np.log(gamma / ((1 - epsilon) * beta * phi)) + 1
	return ret


def comparison_figure(beta, beta_ratio, gamma, epsilon, payment_ratio):
	phi1_step = 0.01
	phi1_range = np.arange(phi1_step, 1, phi1_step)

	plt.rcParams.update({'font.size': 12})
	fig = plt.figure(figsize=(12, 8.5))
	plt.subplots_adjust(hspace=0.3)

	# separable
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	group_utility1 = []
	group_utility2 = []
	individual_utility1 = []
	individual_utility2 = []
	social_utility = []
	b1 = 2 * beta
	b2 = 2 * beta * beta_ratio

	for phi1 in phi1_range:
		phi2 = 1 - phi1

		right_lim = g_peak(b1, gamma, epsilon, phi1)
		S1 = zero_searcher(g, 0, right_lim, b1, gamma, epsilon, phi1) * phi1

		right_lim = g_peak(b2, gamma, epsilon, phi2)
		S2 = zero_searcher(g, 0, right_lim, b2, gamma, epsilon, phi2) * phi2

		group_utility1.append(S1 * payment_ratio)
		group_utility2.append(S2)
		individual_utility1.append(group_utility1[-1] / phi1)
		individual_utility2.append(group_utility2[-1] / phi2)
		social_utility.append(group_utility1[-1] + group_utility2[-1])

	ax1.plot(phi1_range, group_utility1, label='Group 1')
	ax1.plot(phi1_range, group_utility2, label='Group 2')
	ax1.plot(phi1_range, social_utility, label='Social')
	ax2.plot(phi1_range, individual_utility1, label='Group 1')
	ax2.plot(phi1_range, individual_utility2, label='Group 2')
	ax1.set_xlim(0, 1)
	ax2.set_xlim(0, 1)
	l1, l2 = ax1.get_ylim()
	ax1.set_ylim(0, l2)
	l1, l2 = ax2.get_ylim()
	ax2.set_ylim(0, l2)
	ax1.set_xlabel(r'$\phi_1$')
	ax2.set_xlabel(r'$\phi_1$')
	ax1.set_title('Separable Group Utility')
	ax2.set_title('Separable Individual Utility')
	ax1.legend()
	ax2.legend()

	# interacting
	ax1 = fig.add_subplot(223)
	ax2 = fig.add_subplot(224)
	group_utility1 = []
	group_utility2 = []
	individual_utility1 = []
	individual_utility2 = []
	social_utility = []

	for phi1 in phi1_range:
		phi2 = 1 - phi1
		S1, S2 = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon)
		group_utility1.append(S1 * payment_ratio)
		group_utility2.append(S2)
		individual_utility1.append(group_utility1[-1] / phi1)
		individual_utility2.append(group_utility2[-1] / phi2)
		social_utility.append(group_utility1[-1] + group_utility2[-1])

	ax1.plot(phi1_range, group_utility1, label='Group 1')
	ax1.plot(phi1_range, group_utility2, label='Group 2')
	ax1.plot(phi1_range, social_utility, label='Social')
	ax2.plot(phi1_range, individual_utility1, label='Group 1')
	ax2.plot(phi1_range, individual_utility2, label='Group 2')
	ax1.set_xlim(0, 1)
	ax2.set_xlim(0, 1)
	l1, l2 = ax1.get_ylim()
	ax1.set_ylim(0, l2)
	l1, l2 = ax2.get_ylim()
	ax2.set_ylim(0, l2)
	ax1.set_xlabel(r'$\phi_1$')
	ax2.set_xlabel(r'$\phi_1$')
	ax1.set_title('Interacting Group Utility')
	ax2.set_title('Interacting Individual Utility')
	ax1.legend()
	ax2.legend()
	fig.savefig('comparison.png', bbox_inches='tight')
	# plt.show()
	return


def main():
	comparison_figure(beta=2 / 14, beta_ratio=0.6, gamma=1 / 14, epsilon=0.0001, payment_ratio=1.7)
	return


if __name__ == '__main__':
	main()
