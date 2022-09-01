import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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


def NE_searcher(beta1, beta2, gamma, epsilon, T):
	# try phi1=1
	return


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


def final_size_plotter(phi1, beta, beta_ratio, gamma, epsilon):
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


def final_size_searcher(beta, beta_ratio, gamma, epsilon):
	"""
	search for the final sizes of 2 groups interacting
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
	[ax1.scatter(S1_infs[i], S2_infs[i], color='blue', alpha=1 - 0.75 * phi1_range[i], s=1) for i in
	 range(len(S1_infs))]
	[ax1.scatter(S1_approx[i], S2_approx[i], color='red', alpha=1 - 0.75 * phi1_range[i], s=1) for i in
	 range(len(S1_approx))]
	ax1.set_xlabel(r'$S_1(\infty)$')
	ax1.set_ylabel(r'$S_2(\infty)$')
	plt.show()
	return


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


def main():
	# two_group_simulate(0.1, 0.9, 1, 0.5, 1/14, 0.0001, 1000, 10000, True)
	# utility_plotter(1, 0.9, 1 / 14, 0.0001, 100, 1.025)
	# final_size_plotter(0.5, 0.5, 0.9, 1 / 14, 0.0001)
	final_size_searcher(0.5, 0.5, 1 / 14, 0.0001)
	return


if __name__ == '__main__':
	main()
