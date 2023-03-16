import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ConvexProgram import one_group_binary_search

phi_step = 0.001


def dS_by_dPhi(beta, gamma, epsilon, S, phi):
	ret = S * (1 / phi - beta / gamma) / (1 - beta / gamma * S)
	return ret


def one_group_derivative(beta, gamma, epsilon):
	phi_range = np.arange(phi_step, 1, phi_step)
	Ss = []
	dS = [1]
	dS2 = []
	for phi in phi_range:
		Ss.append(one_group_binary_search(beta, gamma, epsilon, phi))
		dS2.append(dS_by_dPhi(beta, gamma, epsilon, Ss[-1], phi))
	dS.extend([(Ss[i] - Ss[i - 1]) / phi_step for i in range(1, len(Ss))])
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi_range, Ss, label='S')
	ax1.plot(phi_range, dS, label='num')
	ax1.plot(phi_range, dS2, label='formula')
	ax1.axhline(0, color='grey', linestyle=':')
	ax1.axvline(gamma / beta, color='grey', linestyle=':')
	ax1.legend()
	plt.show()
	return


def two_group_social(b1, b2, gamma, epsilon, p2):
	phi1_range = np.arange(phi_step, 1, phi_step)
	U1s = []
	U2s = []
	social = []
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		U1s.append(one_group_binary_search(b1, gamma, epsilon, phi1))
		U2s.append(p2 * one_group_binary_search(b2, gamma, epsilon, phi2))
		social.append(U1s[-1] + U2s[-1])
	phi_peak = phi1_range[social.index(max(social))]
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1_range, U1s, label='group1')
	ax1.plot(phi1_range, U2s, label='group2')
	ax1.plot(phi1_range, social, label='social')
	ax1.axvline(phi_peak, linestyle=':', color='grey')

	ax1.axvline(gamma / b1, linestyle=':', color='blue')
	ax1.legend()
	plt.show()
	return


def three_group_social(b1, b2, b3, gamma, epsilon, p2, p3):
	U1s = []
	U2s = []
	U3s = []
	social = []
	phi1s = []
	phi2s = []
	phi3s = []
	for i in range(101):
		for j in range(101):
			k = 100 - i - j
			if k < 0:
				continue
			phi1 = i / 100
			phi2 = j / 100
			phi3 = k / 100
			phi1s.append(phi1)
			phi2s.append(phi2)
			phi3s.append(phi3)
			if i == 0:
				U1s.append(0)
			else:
				U1s.append(one_group_binary_search(b1, gamma, epsilon, phi1))
			if j == 0:
				U2s.append(0)
			else:
				U2s.append(p2 * one_group_binary_search(b2, gamma, epsilon, phi2))
			if k == 0:
				U3s.append(0)
			else:
				U3s.append(p3 * one_group_binary_search(b3, gamma, epsilon, phi3))
			social.append(U1s[-1] + U2s[-1] + U3s[-1])
	fig = plt.figure()
	ax1 = fig.add_subplot(projection='3d')
	ax1.plot_trisurf(phi1s, phi2s, social, cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	plt.show()
	return


def f(S, phi, beta, gamma, epsilon):
	ret = S - (1 - epsilon) * phi * np.exp(beta / gamma * (S - phi))
	return ret


def f_plotter(beta, gamma, epsilon):
	phi_range = np.linspace(0, 1, 100)
	S_range = np.linspace(0, 0, 100)
	f_values = []
	for phi in phi_range:
		f_values.append([])
		for S in S_range:
			f_values[-1].append(f(S, phi, beta, gamma, epsilon))

	X, Y = np.meshgrid(phi_range, S_range)
	fig = plt.figure()
	ax1 = fig.add_subplot(projection='3d')
	ax1.plot_surface(X, Y, np.array(f_values), cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\varphi$')
	ax1.set_ylabel(r'$\bar{S}$')
	plt.show()
	return


def f_plotter_trisurf(beta, gamma, epsilon):
	phi_range = np.linspace(0, 1, 100)
	S_range = np.linspace(0, 1, 100)
	f_values = []
	f_values2 = []
	Ss = []
	phis = []
	Ss2 = []
	phis2 = []
	for phi in phi_range:
		for S in S_range:
			f_value = f(S, phi, beta, gamma, epsilon)
			# if S <= phi:
			if f_value > 0:
				f_values.append(f_value)
				Ss.append(S)
				phis.append(phi)
			else:
				f_values2.append(f_value)
				Ss2.append(S)
				phis2.append(phi)

	f_values = np.array(f_values)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	# ax1 = fig.add_subplot(projection='3d')

	# ax1.plot_trisurf(phis, Ss, f_values, cmap=cm.coolwarm)
	# ax1.scatter(phis, Ss, f_values, color='blue')
	# ax1.scatter(phis2, Ss2, f_values2, color='red')

	ax1.scatter(phis2, Ss2, color='red')
	ax1.set_xlabel(r'$\phi$')
	ax1.set_ylabel(r'$S$')
	plt.show()
	return


def main():
	# f_plotter(beta=4 / 14, gamma=1 / 14, epsilon=0.0001)
	# f_plotter_trisurf(beta=3 / 14, gamma=1 / 14, epsilon=0.0001)

	# one_group_derivative(beta=2 / 14, gamma=1 / 14, epsilon=0.0001)
	# two_group_social(b1=5 / 14, b2=4 / 14, gamma=1 / 14, epsilon=0.0001, p2=0.8)
	three_group_social(b1=8 / 14, b2=7 / 14, b3=6 / 14, gamma=1 / 14, epsilon=0.0001, p2=1, p3=1)
	return


if __name__ == '__main__':
	main()
