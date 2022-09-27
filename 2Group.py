import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

binary_iterations = 50


def two_group_simulate(phi1, phi2, beta, beta_ratio, gamma, epsilon, T, num_steps=10000, plot=False):
	"""
	Simulate 2 group2 with interaction
	"""
	S1 = [phi1 * (1 - epsilon)]
	I1 = [phi1 * epsilon]
	R1 = [0]
	S2 = [phi2 * (1 - epsilon)]
	I2 = [phi2 * epsilon]
	R2 = [0]
	dt = T / num_steps
	t_range = np.arange(0, T + dt, dt)
	for t in t_range[1:]:
		dS1 = -(beta * I1[-1] + beta * beta_ratio * I2[-1]) * S1[-1] * dt
		dI1 = -dS1 - gamma * I1[-1] * dt
		dR1 = gamma * I1[-1] * dt
		dS2 = -(beta * beta_ratio * I1[-1] + beta * beta_ratio * beta_ratio * I2[-1]) * S2[-1] * dt
		dI2 = -dS2 - gamma * I2[-1] * dt
		dR2 = gamma * I2[-1] * dt
		S1.append(S1[-1] + dS1)
		I1.append(I1[-1] + dI1)
		R1.append(R1[-1] + dR1)
		S2.append(S2[-1] + dS2)
		I2.append(I2[-1] + dI2)
		R2.append(R2[-1] + dR2)
	if plot:
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		# ax1.plot(t_range[-round(num_steps / 10):], S1[-round(num_steps / 10):], label='S1')
		# ax1.plot(t_range[-round(num_steps / 10):], S2[-round(num_steps / 10):], label='S2')
		ax1.plot(t_range, S1, label='S1')
		ax1.plot(t_range, S2, label='S2')
		ax2.plot(t_range, [I1[i] / phi1 for i in range(len(I1))], label='I1')
		ax2.plot(t_range, [I2[i] / phi2 for i in range(len(I2))], label='I2')
		ax1.legend()
		ax2.legend()
		plt.show()
	return t_range, S1, S2


def utility_plotter(beta, beta_ratio, gamma, epsilon, T, payment_ratio):
	"""
	plot the group and individual utility of 2 groups interacting over phi
	"""
	phi1_step = 0.01
	phi1_range = np.arange(phi1_step, 1, phi1_step)
	# print(phi1_range, phi1_range[1:-1])
	group_utility1 = []
	group_utility2 = []
	individual_utility1 = []
	individual_utility2 = []

	# # phi1 = 0
	# phi1 = 0
	# phi2 = 1 - phi1
	# t_range, S1, S2 = two_group_simulate(phi1, phi2, beta, beta_ratio, gamma, epsilon, T, plot=False)
	# group_utility1.append(np.mean(S1) * T * payment_ratio)
	# group_utility2.append(np.mean(S2) * T)
	# individual_utility1.append(payment_ratio)
	# individual_utility2.append(group_utility2[-1] / phi2)

	for phi1 in phi1_range:
		phi2 = 1 - phi1
		t_range, S1, S2 = two_group_simulate(phi1, phi2, beta, beta_ratio, gamma, epsilon, T, plot=False)
		group_utility1.append(np.mean(S1) * T * payment_ratio)
		group_utility2.append(np.mean(S2) * T)
		individual_utility1.append(group_utility1[-1] / phi1)
		individual_utility2.append(group_utility2[-1] / phi2)

	# # phi1 = 1
	# phi1 = 1
	# phi2 = 1 - phi1
	# t_range, S1, S2 = two_group_simulate(phi1, phi2, beta, beta_ratio, gamma, epsilon, T, plot=False)
	# group_utility1.append(np.mean(S1) * T * payment_ratio)
	# group_utility2.append(np.mean(S2) * T)
	# individual_utility1.append(group_utility1[-1] / phi1)
	# individual_utility2.append(1)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(phi1_range, group_utility1, label='Group 1')
	ax1.plot(phi1_range, group_utility2, label='Group 2')
	ax1.plot(phi1_range, [group_utility1[i] + group_utility2[i] for i in range(len(group_utility1))], label='Social')
	ax2.plot(phi1_range, individual_utility1, label='Group 1')
	ax2.plot(phi1_range, individual_utility2, label='Group 2')
	ax1.set_xlabel('phi1')
	ax2.set_xlabel('phi1')
	ax1.set_title('Group utility')
	ax2.set_title('Individual utility')
	ax1.legend()
	ax2.legend()
	plt.show()
	return


def final_size_function_plotter(phi1, beta, beta_ratio, gamma, epsilon):
	"""
	Plot the final size functions. The zero points are the final sizes
	"""
	phi2 = 1 - phi1
	b11 = beta
	b12 = b21 = beta * beta_ratio
	b22 = beta * beta_ratio * beta_ratio

	def func_S1_final(S1, S2):
		S1_0 = phi1 * (1 - epsilon)
		return S1 - S1_0 * np.exp(b11 / gamma * (S1 - phi1) + b12 / gamma * (S2 - phi2))

	def func_S2_final(S1, S2):
		S2_0 = phi2 * (1 - epsilon)
		return S2 - S2_0 * np.exp(b21 / gamma * (S1 - phi1) + b22 / gamma * (S2 - phi2))

	S1_step = 0.01
	S1_range = np.arange(0, 1 + S1_step, S1_step)
	S2_step = 0.1
	S2_range = np.arange(0, phi2 + S2_step, S2_step)
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.axhline(0, c='grey', linestyle=':')
	ax1.set_xlabel('S1')
	for S2 in S2_range:
		S1_final = []
		positive = False
		for S1 in S1_range:
			ret = func_S1_final(S1, S2)
			S1_final.append(ret)
			if ret > 0:
				positive = True
			if positive and ret < 0:
				break

		ax1.plot(S1_range[:len(S1_final)], S1_final, label=f'S2={round(S2, 2)}')

	S2_step = 0.01
	S2_range = np.arange(0, 1 + S2_step, S2_step)
	S1_step = 0.1
	S1_range = np.arange(0, phi1 + S1_step, S1_step)
	ax2 = fig.add_subplot(122)
	ax2.axhline(0, c='grey', linestyle=':')
	ax2.set_xlabel('S2')
	for S1 in S1_range:
		S2_final = []
		positive = False
		for S2 in S2_range:
			ret = func_S2_final(S1, S2)
			S2_final.append(ret)
			if ret > 0:
				positive = True
			if positive and ret < 0:
				break

		ax2.plot(S2_range[:len(S2_final)], S2_final, label=f'S1={round(S1, 2)}')

	ax1.legend()
	ax2.legend()
	plt.show()
	return


def final_size_searcher_scipy(beta, beta_ratio, gamma, epsilon):
	"""
	search for the final sizes of 2 groups interacting using SciPy optimizer
	"""
	phi1_step = 0.01
	phi1_range = np.arange(0, 1 + phi1_step, phi1_step)
	S1_infs = []
	S2_infs = []
	S1_approx = []
	S2_approx = []
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		optimal = minimize(two_group_loss, [phi1 / 2, phi2 / 2],
						   args=(phi1, beta, beta_ratio, gamma, epsilon),
						   method='L-BFGS-B',
						   bounds=[(0, phi1), (0, phi2)])
		# print(optimal.fun)
		S1, S2 = optimal.x
		S1_infs.append(S1)
		S2_infs.append(S2)
		S1_approx.append((1 - epsilon) * phi1 *
						 (gamma - beta * (phi1 + beta_ratio * (beta_ratio + epsilon - beta_ratio * epsilon) * phi2)) /
						 (gamma + beta * (epsilon - 1) * (phi1 + beta_ratio ** 2 * phi2)))
		S2_approx.append((1 - epsilon) * phi2 *
						 (gamma - beta * (phi1 + (beta_ratio - 1) * epsilon * phi1 + beta_ratio ** 2 * phi2)) /
						 (gamma + beta * (epsilon - 1) * (phi1 + beta_ratio ** 2 * phi2)))
	# print([(S1, S2) for (S1, S2) in zip(S1_infs, S2_infs)])
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(S1_infs, S2_infs)
	# [ax1.scatter(S1_infs[i], S2_infs[i], color='blue', alpha=1 - 0.75 * phi1_range[i], s=1) for i in
	#  range(len(S1_infs))]
	# [ax1.scatter(S1_approx[i], S2_approx[i], color='red', alpha=1 - 0.75 * phi1_range[i], s=1) for i in
	#  range(len(S1_approx))]
	ax1.set_xlabel(r'$S_1(\infty)$')
	ax1.set_ylabel(r'$S_2(\infty)$')
	plt.show()
	return


def final_size_plotter(beta, beta_ratio, gamma, epsilon, payment_ratio=1):
	"""
	plot the final sizes of 2 groups interacting over various phi1 values
	"""
	phi1_step = 0.005
	phi1_range = np.arange(0, 1 + phi1_step, phi1_step)
	l = len(phi1_range)
	S1_final = []
	S2_final = []
	for phi1 in phi1_range:
		S1, S2 = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon)
		S1_final.append(S1)
		S2_final.append(S2)
	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	# ax3 = ax2.twinx()
	ax4 = fig.add_subplot(223)
	ax1.set_xlabel('S1')
	ax1.set_ylabel('S2')
	ax1.set_title('Scatter color becomes lighter as phi_1 increase in cycles\n'
				  f'beta={round(beta, 3)},  beta ratio={round(beta_ratio, 3)},  gamma={round(gamma, 3)}')
	ax1.plot(S1_final, S2_final, color='gray')
	# cs = [np.exp(i - l) / np.exp(l) for i in range(l)]
	# ax1.scatter(S1_final, S2_final, c=cs, cmap='binary_r')
	for i in range(20):
		ax1.scatter(S1_final[round(i * l / 20):round((i + 1) * l / 20)],
					S2_final[round(i * l / 20):round((i + 1) * l / 20)],
					c=range(round(i * l / 20), round((i + 1) * l / 20)),
					edgecolors='black', cmap='binary_r', s=20, zorder=2)

	ax2.plot(phi1_range, [S1 * payment_ratio for S1 in S1_final], label='S1')
	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_title(f'group utility\npayment ratio={round(payment_ratio, 5)}')
	# ax2.set_ylabel('S1')
	ax2.plot(phi1_range, S2_final, label='S2')
	ax2.plot(phi1_range, [S1_final[i] * payment_ratio + S2_final[i] for i in range(len(S1_final))], label='social')
	# ax3.set_xlabel(r'$\phi_1$')
	# ax3.set_ylabel('S2')
	ax2.legend()

	ax4.plot(phi1_range[1:-1], [payment_ratio * S1_final[i] / phi1_range[i] for i in range(1, l - 1)], label='S1')
	# ax5 = ax4.twinx()
	ax4.plot(phi1_range[1:-1], [S2_final[i] / (1 - phi1_range[i]) for i in range(1, l - 1)], label='S2')
	ax4.set_xlabel(r'$\phi_1$')
	# ax4.set_ylabel('S1')
	# ax5.set_ylabel('S2')
	ax4.set_title('individual utility')
	ax4.legend()
	# print(S1_final[1] / phi1_range[1], S2_final[1] / (1 - phi1_range[1]))
	# print(1 / (S1_final[1] / phi1_range[1]) * (S2_final[1] / (1 - phi1_range[1])))
	plt.show()
	return


def final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon):
	"""
	binary search the final sizes of 2 groups interacting
	"""
	phi2 = 1 - phi1
	S2_l = 0
	S2_r = phi2 * (1 - epsilon)
	for _ in range(binary_iterations):
		S2_m = (S2_l + S2_r) / 2
		S1 = S1_final_searcher(S2_m, beta, beta_ratio, gamma, epsilon, phi1)
		f = f2([S1, S2_m], phi1, beta, beta_ratio, gamma, epsilon)
		if f > 0:
			S2_r = S2_m
		else:
			S2_l = S2_m
	S2 = S2_m
	S1 = S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1)
	return S1, S2


def S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1):
	S_trace = []
	f_trace = []
	S1_l = 0
	S1_r = phi1 * (1 - epsilon)
	for _ in range(binary_iterations):
		S1_m = (S1_l + S1_r) / 2
		f = f1([S1_m, S2], phi1, beta, beta_ratio, gamma, epsilon)
		S_trace.append(S1_m)
		f_trace.append(f)
		if f > 0:
			S1_r = S1_m
		else:
			S1_l = S1_m
	# print(S_trace)
	# print(f_trace)
	# fig = plt.figure()
	# ax1 = fig.add_subplot()
	# ax1.plot(S_trace, f_trace)
	# plt.show()
	return S1_m


def two_group_loss(point, phi1, beta, beta_ratio, gamma, epsilon):
	loss = f1(point, phi1, beta, beta_ratio, gamma, epsilon) ** 2
	loss += f2(point, phi1, beta, beta_ratio, gamma, epsilon) ** 2
	return loss


def f1(point, phi1, beta, beta_ratio, gamma, epsilon):
	[S1, S2] = point
	phi2 = 1 - phi1
	b11 = beta
	b12 = beta * beta_ratio
	# b22 = beta * beta_ratio * beta_ratio
	S1_0 = phi1 * (1 - epsilon)
	# S2_0 = phi2 * (1 - epsilon)
	ret = S1 - S1_0 * np.exp(b11 / gamma * (S1 - phi1) + b12 / gamma * (S2 - phi2))
	return ret


def f2(point, phi1, beta, beta_ratio, gamma, epsilon):
	[S1, S2] = point
	phi2 = 1 - phi1
	# b11 = beta
	b21 = beta * beta_ratio
	b22 = beta * beta_ratio * beta_ratio
	# S1_0 = phi1 * (1 - epsilon)
	S2_0 = phi2 * (1 - epsilon)
	ret = S2 - S2_0 * np.exp(b21 / gamma * (S1 - phi1) + b22 / gamma * (S2 - phi2))
	return ret


def f1_plotter(beta, beta_ratio, gamma, epsilon):
	fig = plt.figure()
	ax1 = fig.add_subplot()
	phi1 = phi2 = 0.5
	S2_range = np.arange(0, phi2, 0.1)
	for S2 in S2_range:
		S1_range = np.arange(0, phi1, 0.001)
		fs = [f1([S1, S2], phi1, beta, beta_ratio, gamma, epsilon) for S1 in S1_range]
		ax1.plot(S1_range, fs, label=f'S2={round(S2, 2)}')

	ax1.set_xlabel('S1')
	ax1.set_ylabel('f')
	ax1.legend()
	plt.show()
	return


def final_size_approximation_comparison(beta, beta_ratio, gamma, epsilon):
	"""
	compare the approximated final sizes
	"""
	phi1_step = 0.005
	phi1_range = np.arange(0, 1 + phi1_step, phi1_step)
	l = len(phi1_range)
	S1_final = []
	S2_final = []
	S1_final_approx = []
	S2_final_approx = []
	for phi1 in phi1_range:
		S1, S2 = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon)
		S1_final.append(S1)
		S2_final.append(S2)
		S1_approx, S2_approx = final_size_approximation(phi1, beta, beta_ratio, gamma, epsilon)
		S1_final_approx.append(S1_approx)
		S2_final_approx.append(S2_approx)

	# fig = plt.figure()
	# ax1 = fig.add_subplot(121)
	# ax2 = fig.add_subplot(122)
	# ax1.plot(phi1_range, S1_final, label='actual')
	# ax1.plot(phi1_range, S1_final_approx, label='approx')
	# ax1.set_title('S1')
	# ax2.plot(phi1_range, S2_final, label='actual')
	# ax2.plot(phi1_range, S2_final_approx, label='approx')
	# ax2.set_title('S2')
	# ax1.legend()
	# ax2.legend()

	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.scatter(S1_final, S2_final, label='actual')
	ax1.scatter(S1_final_approx, S2_final_approx, label='approx')
	for i in range(l):
		print(i % 10)
		if i % 10 == 0:
			plt.show()
			fig = plt.figure()
			ax1 = fig.add_subplot()
		ax1.scatter(S1_final[i], S2_final[i], color='blue')
		ax1.scatter(S1_final_approx[i], S2_final_approx[i], color='orange')
	ax1.legend()
	ax1.set_xlabel('S1')
	ax1.set_ylabel('S2')

	plt.show()
	return


def final_size_approximation_comparison2(beta, beta_ratio, gamma, epsilon):
	"""
	compare the approximated final sizes
	"""
	phi1_step = 0.005
	phi1_range = np.arange(0, 1 + phi1_step, phi1_step)
	l = len(phi1_range)
	S1_final = []
	S2_final = []
	S1_final_approx = []
	S2_final_approx = []
	S1_final_approx2 = []
	S2_final_approx2 = []
	for phi1 in phi1_range:
		S1, S2 = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon)
		S1_final.append(S1)
		S2_final.append(S2)

		S1_approx, S2_approx = final_size_approximation(phi1, beta, beta_ratio, gamma, epsilon)
		S1_final_approx.append(S1_approx)
		S2_final_approx.append(S2_approx)
		S1_approx2, S2_approx2 = final_size_approximation2(phi1, beta, beta_ratio, gamma, epsilon)
		S1_final_approx2.append(S1_approx2)
		S2_final_approx2.append(S2_approx2)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	ax1.plot(phi1_range, S1_final, label='actual')
	ax2.plot(phi1_range, S2_final, label='actual')
	ax1.plot(phi1_range, S1_final_approx, label='approx')
	ax2.plot(phi1_range, S2_final_approx, label='approx')
	ax1.plot(phi1_range, S1_final_approx2, label='approx1')
	ax2.plot(phi1_range, S2_final_approx2, label='approx2')
	ax1.set_title('S1')
	ax2.set_title('S2')
	ax1.legend()
	ax2.legend()
	plt.show()
	return


def final_size_approximation(phi1, beta, beta_ratio, gamma, epsilon):
	"""
	approximate the final sizes with e^x=1+x
	"""
	phi2 = 1 - phi1
	b11 = beta
	b12 = b21 = beta * beta_ratio
	b22 = beta * beta_ratio * beta_ratio
	D = np.exp(((b11 + b21) * phi1 + (b12 + b22) * phi2) / gamma) * gamma ** 2 + \
		np.exp((b21 * phi1 + b22 * phi2) / gamma) * b11 * gamma * (epsilon - 1) * phi1 + \
		np.exp((b11 * phi1 + b12 * phi2) / gamma) * b22 * gamma * (epsilon - 1) * phi2 - \
		(b12 * b21 - b11 * b22) * (epsilon - 1) ** 2 * phi1 * phi2
	S1 = - gamma * (epsilon - 1) * phi1 * \
		 (np.exp((b21 * phi1 + b22 * phi2) / gamma) * gamma - (b12 - b22) * (epsilon - 1) * phi2) / D
	S2 = - gamma * (epsilon - 1) * phi2 * \
		 (np.exp((b11 * phi1 + b12 * phi2) / gamma) * gamma - (b11 - b21) * (epsilon - 1) * phi1) / D

	return S1, S2


def final_size_approximation2(phi1, beta, beta_ratio, gamma, epsilon):
	"""
	approximate the final sizes with e^x=1+2x
	"""
	phi2 = 1 - phi1
	b11 = beta
	b12 = b21 = beta * beta_ratio
	b22 = beta * beta_ratio * beta_ratio
	c1 = (1 - epsilon) * phi1 / np.exp(b11 * phi1 / gamma + b12 * phi2 / gamma)
	c2 = (1 - epsilon) * phi2 / np.exp(b21 * phi1 / gamma + b22 * phi2 / gamma)
	# c3 = c2 / (1 - c2 * b22 / gamma)
	# c4 = (c2 * b21 / gamma) / (1 - c2 * b22 / gamma)
	S1 = (c1 * gamma * (2 * c2 * (b12 - b22) + gamma)) / \
		 (gamma * (-2 * c2 * b22 + gamma) - 2 * c1 * (2 * c2 * b12 * b21 - 2 * c2 * b11 * b22 + b11 * gamma))
	S2 = (c2 * gamma * (2 * c1 * (b21 - b11) + gamma)) / \
		 (gamma * (-2 * c2 * b22 + gamma) - 2 * c1 * (2 * c2 * b12 * b21 - 2 * c2 * b11 * b22 + b11 * gamma))
	return S1, S2


def tmp(beta, beta_ratio, gamma, epsilon):
	phi1_step = 0.01
	phi1_range = np.arange(phi1_step, 1, phi1_step)
	b11 = beta
	b12 = b21 = beta * beta_ratio
	b22 = beta * beta_ratio * beta_ratio
	term1 = []
	term2 = []
	term3 = []
	term32 = []
	y = beta / gamma
	k = beta_ratio
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		c1 = (1 - epsilon) * phi1 / np.exp(b11 * phi1 / gamma + b12 * phi2 / gamma)
		c2 = (1 - epsilon) * phi2 / np.exp(b21 * phi1 / gamma + b22 * phi2 / gamma)
		term1.append(1 - 2 * c1 * b11 / gamma - 4 * c1 * c2 * b12 * b21 / gamma / (gamma - 2 * c2 * b22))
		term2.append(2 * c1 * b11 / gamma)
		term3.append(4 * c1 * c2 * b12 * b21 / gamma / (gamma - 2 * c2 * b22))

		term32.append((4 * (k ** 2) * (y ** 2) + 2 * 1 * (k ** 2) * y)
					  / np.exp((phi1 + (k ** 2) + phi2 * (k ** 2)) * y))

	# term32.append(y * 2 * phi2 * (k ** 2)
	# 			  / np.exp(k * phi1 + (k ** 2) * phi2)
	# 			  * (1 + 8 * phi1 * y / np.exp(k * y)) - 1)

	fig = plt.figure()
	ax1 = fig.add_subplot()
	# ax1.plot(phi1_range, term1, label='overall')
	# ax1.plot(phi1_range, term2, label='term2')
	ax1.plot(phi1_range, term3, label='term3')
	# ax1.plot(phi1_range, term32, label='term32')
	ax1.axhline(0, c='grey')
	# ax1.axhline(0.5, color='gray')
	# ax1.axvline(gamma / (1 - beta_ratio) / beta, color='gray')
	ax1.legend()
	plt.show()
	return


def tmp2(beta_ratio, gamma, epsilon):
	beta_low = 0.01 / 14
	beta_high = 2 / 14
	beta_step = (beta_high - beta_low) / 40
	beta_range = np.arange(beta_low, beta_high + beta_step, beta_step)
	phi1_step = 0.01
	phi1_range = np.arange(phi1_step, 1, phi1_step)
	peaks1 = []
	peaks2 = []
	max_phi = []
	for beta in beta_range:
		b11 = beta
		b12 = b21 = beta * beta_ratio
		b22 = beta * beta_ratio * beta_ratio
		term1 = []
		term2 = []
		# term3 = []

		for phi1 in phi1_range:
			phi2 = 1 - phi1
			c1 = (1 - epsilon) * phi1 / np.exp(b11 * phi1 / gamma + b12 * phi2 / gamma)
			c2 = (1 - epsilon) * phi2 / np.exp(b21 * phi1 / gamma + b22 * phi2 / gamma)
			term1.append(1 - 2 * c1 * b11 / gamma - 4 * c1 * c2 * b12 * b21 / gamma / (gamma - 2 * c2 * b22))
			term2.append(2 * c1 * b11 / gamma)
		# term3.append(4 * c1 * c2 * b12 * b21 / gamma / (gamma - 2 * c2 * b22))
		peaks1.append(min(term1))
		peaks2.append(max(term2))
		max_phi.append(phi1_range[term2.index(max(term2))])
	print(max_phi)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(beta_range, peaks1, label='term1 min')
	ax1.plot(beta_range, peaks2, label='term2 peak')
	ax1.plot(beta_range, max_phi, label='phi')
	# ax1.plot(phi1_range, term2, label='term2')
	# ax1.plot(phi1_range, term3, label='term3')
	ax1.axhline(0.5, color='gray')
	ax1.legend()
	plt.show()
	return


def main():
	# two_group_simulate(0.1, 0.9, 1, 0.5, 1/14, 0.0001, 1000, 10000, True)
	# utility_plotter(1, 0.9, 1 / 14, 0.0001, 100, 1.025)
	# final_size_function_plotter(0.5, 0.5, 0.5, 1 / 14, 0.0001)
	# final_size_searcher_scipy(2, 0.5, 1 / 14, 0.0001)
	# final_size_plotter(beta=0.5, beta_ratio=0.7, gamma=1 / 14, epsilon=0.0001, payment_ratio=1)
	# final_size_approximation_comparison(beta=0.5, beta_ratio=0.5, gamma=1 / 14, epsilon=0.0001)
	# final_size_approximation_comparison2(beta=3, beta_ratio=0.7, gamma=1 / 14, epsilon=0.0001)
	# f1_plotter(0.5, 0.5, 1 / 14, 0.0001)
	tmp(beta=100 / 14, beta_ratio=0.1, gamma=1 / 14, epsilon=0.0001)
	# tmp2(beta_ratio=0.5, gamma=1 / 14, epsilon=0.0001)
	return


if __name__ == '__main__':
	main()
