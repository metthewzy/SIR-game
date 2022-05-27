import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import uniform as uni
import time

S_0 = 1
I_0 = 0.0001
GAMMA = 1 / 14
beta_0 = 1
separate_betas = True
show_figure = True


def tests():
	return


def simulate(beta, gamma, S0, I0, t_vac, showPlot):
	"""
	1 group SIR simulation
	"""
	S = [S0]
	I = [I0 * S0]
	dt = 0.01
	t_range = np.arange(0, t_vac + dt, dt)
	for t in t_range[1:]:
		dS = (- beta * S[-1] * I[-1]) * dt
		dI = (beta * S[-1] - gamma) * I[-1] * dt
		S.append(S[-1] + dS)
		I.append(I[-1] + dI)

	if showPlot:
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.plot(t_range, S, label='S')
		ax.plot(t_range, I, label='I')
		ax.legend()
		plt.show()
		plt.close(fig)
	return S, I, t_range


def simulate_interaction(beta1, beta2, gamma, S1_0, S2_0, t_vac, showPlot):
	"""
	2 groups simulation with interactions in between
	"""
	S1 = [S1_0]
	I1 = [I_0 * S1_0]
	S2 = [S2_0]
	I2 = [I_0 * S2_0]
	dt = 0.01
	t_range = np.arange(0, t_vac + dt, dt)
	for t in t_range[1:]:
		# dS1 = (- S1[-1] * (beta1 * I1[-1] + beta2 * I2[-1])) * dt
		# dI1 = (S1[-1] * (beta1 * I1[-1] + beta2 * I2[-1]) - gamma * I1[-1]) * dt
		dS1 = (- beta1 * S1[-1] * (I1[-1] + I2[-1])) * dt
		dI1 = (beta1 * S1[-1] * (I1[-1] + I2[-1]) - gamma * I1[-1]) * dt

		# dS2 = (- S2[-1] * (beta1 * I1[-1] + beta2 * I2[-1])) * dt
		# dI2 = (S2[-1] * (beta1 * I1[-1] + beta2 * I2[-1]) - gamma * I2[-1]) * dt
		dS2 = (- beta2 * S2[-1] * (I1[-1] + I2[-1])) * dt
		dI2 = (beta2 * S2[-1] * (I1[-1] + I2[-1]) - gamma * I2[-1]) * dt

		S1.append(S1[-1] + dS1)
		I1.append(I1[-1] + dI1)
		S2.append(S2[-1] + dS2)
		I2.append(I2[-1] + dI2)

	if showPlot:
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.plot(t_range, S1, label='S1')
		ax.plot(t_range, I1, label='I1')
		ax.plot(t_range, S2, label='S2')
		ax.plot(t_range, I2, label='I2')
		ax.legend()
		plt.show()
		plt.close(fig)
	return S1, I1, S2, I2, t_range


def simulate_interaction_V2(beta11, beta12, beta21, beta22, gamma, S1_0, S2_0, t_vac, showPlot):
	"""
	2 groups simulation with interactions in between. 4 betas for each S and I combination
	"""
	S1 = [S1_0]
	I1 = [I_0 * S1_0]
	S2 = [S2_0]
	I2 = [I_0 * S2_0]
	dt = 0.01
	t_range = np.arange(0, t_vac + dt, dt)
	for t in t_range[1:]:
		dS1 = - (beta11 * S1[-1] * I1[-1] + beta12 * S1[-1] * I2[-1]) * dt
		dI1 = (beta11 * S1[-1] * I1[-1] + beta12 * S1[-1] * I2[-1] - gamma * I1[-1]) * dt

		dS2 = - (beta21 * S2[-1] * I1[-1] + beta22 * S2[-1] * I2[-1]) * dt
		dI2 = (beta21 * S2[-1] * I1[-1] + beta22 * S2[-1] * I2[-1] - gamma * I2[-1]) * dt

		S1.append(S1[-1] + dS1)
		I1.append(I1[-1] + dI1)
		S2.append(S2[-1] + dS2)
		I2.append(I2[-1] + dI2)

	if showPlot:
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.plot(t_range, S1, label='S1')
		ax.plot(t_range, I1, label='I1')
		ax.plot(t_range, S2, label='S2')
		ax.plot(t_range, I2, label='I2')
		ax.legend()
		plt.show()
		plt.close(fig)
	return S1, I1, S2, I2, t_range


def dU_by_dt(income, beta, S_t, I_t, S0, t, t_vac):
	if S0 == 0:
		return income
	value = income - beta * S_t * I_t / S0 * (t_vac - t) * income
	return value


def utility_plotter(income_ratio, beta_ratio, t_vac, gamma):
	"""
	2 group game w/o interaction. S:susceptible, M:mask. plotting
	the player's expected utility in each group, and the social
	utility with varying size of group S
	"""
	U_S = []
	U_M = []
	SS_list = []
	SM_list = []
	socialU = []
	# susceptible group daily payment. mask group daily payment assumed to be 1
	GDP1 = income_ratio
	GDP2 = 1
	beta_S = beta_0
	beta_M = beta_0 * beta_ratio
	step_size = 0.005
	# S0_S_range = np.arange(0, 1 + step_size, step_size)
	S0_S_range = np.arange(0 + step_size, 1, step_size)
	for S0_S in S0_S_range:
		SS, IS, t_range = simulate(beta_S, gamma, S0_S, I_0, t_vac, False)
		# susceptible group utility
		SS_list.append(GDP1 * np.mean(SS) * t_vac)
		# player's expected utility in susceptible group
		# U_S.append(np.mean(
		# 	[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)
		U_S.append(GDP1 * t_vac if S0_S == 0 else SS_list[-1] / S0_S)

		S0_M = 1 - S0_S
		SM, IM, t_range = simulate(beta_M, gamma, S0_M, I_0, t_vac, False)
		# mask group utility
		SM_list.append(GDP2 * np.mean(SM) * t_vac)
		# player's expected utility in mask group
		# U_M.append(np.mean(
		# 	[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)
		U_M.append(GDP2 * t_vac if S0_M == 0 else SM_list[-1] / S0_M)

		socialU.append(SS_list[-1] + SM_list[-1])

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	# ax1.plot(S0_S_range, U_S, label='U(Player)_S')
	# ax1.plot(S0_S_range, U_M, label='U(Player)_M')
	ax1.plot(S0_S_range, SS_list, label='Group Utility s')
	ax1.plot(S0_S_range, SM_list, label='Group Utility m')
	ax1.plot(S0_S_range, socialU, label='social')
	max_social = max(socialU)
	maxIndex = socialU.index(max_social)
	ax1.axhline(max_social, linestyle=':', label=f'OPT={round(max_social, 4)}', color='red')
	ax1.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')
	ax1.set_xlabel('susceptible size')
	ax1.set_ylabel('utility')
	ax1.legend()

	# dGroup utility by d phi(S)
	ax3 = fig.add_subplot(223)
	dGroup_utility_S = [SS_list[i] - SS_list[i - 1] for i in range(1, len(SS_list))]
	ax3.plot(S0_S_range[1:], dGroup_utility_S, label='dUG(S)')

	dGroup_utility_M = [SM_list[i] - SM_list[i - 1] for i in range(1, len(SM_list))]
	ax3.plot(S0_S_range[1:], dGroup_utility_M, label='dUG(M)')

	sum_dUG = [dGroup_utility_S[i] + dGroup_utility_M[i] for i in range(len(dGroup_utility_S))]
	ax3.plot(S0_S_range[1:], sum_dUG, label='sum dUG')
	# ax3.axhline(sum_dUG[maxIndex], linestyle=':', label=f'OPT={round(sum_dUG[maxIndex], 4)}', color='red')
	ax3.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')
	ax3.legend()

	# search for the NE point
	NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, gamma)
	NE_S0_S = NE_S0_S_range[-1]
	POA = max_social / NE_utility
	print('POA=', POA)
	# NE_S0_S_range, NE_U_S, NE_U_M = map(list, zip(*sorted(zip(NE_S0_S_range, NE_U_S, NE_U_M))))
	ax2 = fig.add_subplot(222)
	# ax2.plot(NE_S0_S_range, NE_U_S, label='U(Player)_S')
	# ax2.plot(NE_S0_S_range, NE_U_M, label='U(Player)_M')
	ax2.plot(S0_S_range, U_S, label='U(Player)_S')
	ax2.plot(S0_S_range, U_M, label='U(Player)_M')
	ax2.axhline(NE_utility, label=f'NE={round(NE_utility, 4)}', linestyle=':', color='red')
	ax2.axvline(NE_S0_S, label=f'NE@{round(NE_S0_S, 4)}', linestyle=':', color='red')
	ax2.set_xlabel('susceptible size')
	ax2.set_ylabel('utility')
	ax2.legend()
	fig.suptitle(f'POA={round(max_social / NE_utility, 4)}')
	plt.show()
	plt.close(fig)
	return


def NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, gamma):
	"""
	return the Nash Equilibrium at the last location in the list
	"""
	# player utility for group S and M
	U_S = []
	U_M = []

	# trying S0_S=0
	S0_S_range = [0]
	S0_S = S0_S_range[-1]
	SS, IS, t_range = simulate(beta_S, gamma, S0_S, I_0, t_vac, False)
	U_S.append(np.mean(
		[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	S0_M = 1 - S0_S
	SM, IM, t_range = simulate(beta_M, gamma, S0_M, I_0, t_vac, False)
	U_M.append(np.mean(
		[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	if U_M[-1] > U_S[-1]:
		return S0_S_range, U_S, U_M, U_M[-1]

	# trying S0_S=1
	S0_S_range.append(1)
	S0_S = S0_S_range[-1]
	SS, IS, t_range = simulate(beta_S, gamma, S0_S, I_0, t_vac, False)
	U_S.append(np.mean(
		[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	S0_M = 1 - S0_S
	SM, IM, t_range = simulate(beta_M, gamma, S0_M, I_0, t_vac, False)
	U_M.append(np.mean(
		[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	if U_S[-1] > U_M[-1]:
		return S0_S_range, U_S, U_M, U_S[-1]

	left, right, mid = 0, 1, 0.5
	for _ in range(40):
		S0_S_range.append(mid)
		S0_S = S0_S_range[-1]
		SS, IS, t_range = simulate(beta_S, gamma, S0_S, I_0, t_vac, False)
		U_S.append(np.mean(
			[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

		S0_M = 1 - S0_S
		SM, IM, t_range = simulate(beta_M, gamma, S0_M, I_0, t_vac, False)
		U_M.append(np.mean(
			[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

		if U_S[-1] > U_M[-1]:
			# S0_S too small
			left = mid
		else:
			# S0_S too large
			right = mid
		mid = (left + right) / 2
	return S0_S_range, U_S, U_M, (U_S[-1] + U_M[-1]) / 2


def POA_grid():
	"""
	heatmap for POA over different daily income ratios and beta ratios
	"""
	# income_ratios = range(1, 11)
	income_ratios = [1, 2, 4, 6, 8, 10]
	beta_ratios = [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10]
	t_vac = 100
	POAs = []
	max_POA = 0
	for GDP1 in income_ratios:
		POAs.append([])
		GDP2 = 1
		beta_S = beta_0
		for beta_ratio in beta_ratios:
			print(GDP1, beta_ratio)
			U_S = []
			U_M = []
			SS_list = []
			SM_list = []
			socialU = []
			beta_M = beta_S * beta_ratio
			step_size = 0.01
			S0_S_range = np.arange(0, 1 + step_size, step_size)
			for S0_S in S0_S_range:
				SS, IS, t_range = simulate(beta_S, GAMMA, S0_S, I_0, t_vac, False)
				# susceptible group utility
				SS_list.append(GDP1 * np.mean(SS) * t_vac)
				# player's expected utility in susceptible group
				U_S.append(np.mean(
					[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in
					 range(len(t_range))]) * t_vac)

				S0_M = 1 - S0_S
				SM, IM, t_range = simulate(beta_M, GAMMA, S0_M, I_0, t_vac, False)
				# mask group utility
				SM_list.append(GDP2 * np.mean(SM) * t_vac)
				# player's expected utility in mask group
				U_M.append(np.mean(
					[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in
					 range(len(t_range))]) * t_vac)

				socialU.append(SS_list[-1] + SM_list[-1])

			max_social = max(socialU)
			maxIndex = socialU.index(max_social)

			NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, GAMMA)
			NE_S0_S = NE_S0_S_range[-1]
			POA = max_social / NE_utility
			POAs[-1].append(POA)
			max_POA = max(max_POA, POA)
	ax = sns.heatmap(POAs, annot=True, fmt=".3")
	ax.set_xticklabels([round(i, 2) for i in beta_ratios])
	ax.set_yticklabels(income_ratios)
	ax.set_xlabel('beta ratio')
	ax.set_ylabel('income ratio')
	print('max POA=', max_POA)
	plt.show()

	return


def tmp():
	i = j = [3]
	i += j
	print(i, j)
	return


def utility_plotter_interaction(income_ratio, beta_ratio, t_vac, gamma):
	"""
	2 group game with interaction. S:susceptible, M:mask. plotting the player's
	expected utility in each group,	and the social utility with varying size of
	group S
	"""
	U_S = []
	U_M = []
	SS_list = []
	SM_list = []
	socialU = []
	# susceptible group daily payment. mask group daily payment assumed to be 1
	GDP1 = income_ratio
	GDP2 = 1
	beta_S = beta_0
	beta_M = beta_0 * beta_ratio

	beta_SS = beta_0
	beta_SM = beta_0 * beta_ratio
	beta_MS = beta_0 * beta_ratio ** 2
	beta_MM = beta_0 * beta_ratio ** 3

	step_size = 0.005
	# S0_S_range = np.arange(0, 1 + step_size, step_size)
	S0_S_range = np.arange(0 + step_size, 1, step_size)
	for S0_S in S0_S_range:
		S0_M = 1 - S0_S
		if not separate_betas:
			SS, IS, SM, IM, t_range = simulate_interaction(beta_S, beta_M, gamma, S0_S, S0_M, t_vac,
			                                               True if S0_S == 0.8 and show_figure else False)
		else:
			SS, IS, SM, IM, t_range = simulate_interaction_V2(beta_SS, beta_SM, beta_MS, beta_MM, gamma, S0_S, S0_M,
			                                                  t_vac, True if S0_S == 0.8 and show_figure else False)

		# susceptible group utility
		SS_list.append(GDP1 * np.mean(SS) * t_vac)
		# player's expected utility in susceptible group
		U_S.append(GDP1 * t_vac if S0_S == 0 else SS_list[-1] / S0_S)

		# mask group utility
		SM_list.append(GDP2 * np.mean(SM) * t_vac)
		# player's expected utility in mask group
		U_M.append(GDP2 * t_vac if S0_M == 0 else SM_list[-1] / S0_M)

		socialU.append(SS_list[-1] + SM_list[-1])

	fig = plt.figure()
	ax1 = fig.add_subplot(221)
	# ax1.plot(S0_S_range, U_S, label='U(Player)_S')
	# ax1.plot(S0_S_range, U_M, label='U(Player)_M')
	ax1.plot(S0_S_range, SS_list, label='Group Utility s')
	ax1.plot(S0_S_range, SM_list, label='Group Utility m')
	ax1.plot(S0_S_range, socialU, label='social')
	max_social = max(socialU)
	maxIndex = socialU.index(max_social)
	ax1.axhline(max_social, linestyle=':', label=f'OPT={round(max_social, 4)}', color='red')
	ax1.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')
	ax1.set_xlabel('susceptible size')
	ax1.set_ylabel('utility')
	ax1.legend()

	# dGroup utility by d phi(S)
	ax3 = fig.add_subplot(223)
	dGroup_utility_S = [SS_list[i] - SS_list[i - 1] for i in range(1, len(SS_list))]
	ax3.plot(S0_S_range[1:], dGroup_utility_S, label='dUG(S)')

	dGroup_utility_M = [SM_list[i] - SM_list[i - 1] for i in range(1, len(SM_list))]
	ax3.plot(S0_S_range[1:], dGroup_utility_M, label='dUG(M)')

	sum_dUG = [dGroup_utility_S[i] + dGroup_utility_M[i] for i in range(len(dGroup_utility_S))]
	ax3.plot(S0_S_range[1:], sum_dUG, label='sum dUG')
	# ax3.axhline(sum_dUG[maxIndex], linestyle=':', label=f'OPT={round(sum_dUG[maxIndex], 4)}', color='red')
	ax3.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')
	ax3.legend()

	# # search for the NE point
	# NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M)
	# NE_S0_S = NE_S0_S_range[-1]
	# POA = max_social / NE_utility
	# print('POA=', POA)
	# NE_S0_S_range, NE_U_S, NE_U_M = map(list, zip(*sorted(zip(NE_S0_S_range, NE_U_S, NE_U_M))))

	ax2 = fig.add_subplot(222)
	# ax2.plot(NE_S0_S_range, NE_U_S, label='U(Player)_S')
	# ax2.plot(NE_S0_S_range, NE_U_M, label='U(Player)_M')
	ax2.plot(S0_S_range, U_S, label='U(Player)_S')
	ax2.plot(S0_S_range, U_M, label='U(Player)_M')

	# ax2.axhline(NE_utility, label=f'NE={round(NE_utility, 4)}', linestyle=':', color='red')
	# ax2.axvline(NE_S0_S, label=f'NE@{round(NE_S0_S, 4)}', linestyle=':', color='red')

	ax2.set_xlabel('susceptible size')
	ax2.set_ylabel('utility')
	ax2.legend()

	# fig.suptitle(f'POA={round(max_social / NE_utility, 4)}')

	plt.show()
	plt.close(fig)
	return


def POA_monte_carlo(runs):
	"""
	Monte Carlo to generate max POA. 2 group game without interaction
	"""
	np.random.seed()
	beta_range = (0.05, 1)
	gamma_range = (1 / 20, 1 / 5)
	income_range = (1, 50)
	beta_ratio_range = (0.01, 1)

	t_vac = 100
	max_POA = 0
	max_paras = []
	GDP2 = 1

	t1 = time.perf_counter()
	for _ in range(runs):
		beta_S = uni(beta_range[0], beta_range[1])
		gamma = uni(gamma_range[0], gamma_range[1])
		GDP1 = uni(income_range[0], income_range[1])
		beta_M = beta_S * uni(beta_ratio_range[0], beta_ratio_range[1])

		U_S = []
		U_M = []
		SS_list = []
		SM_list = []
		socialU = []
		step_size = 0.01
		S0_S_range = np.arange(0, 1 + step_size, step_size)
		for S0_S in S0_S_range:
			SS, IS, t_range = simulate(beta_S, gamma, S0_S, I_0, t_vac, False)
			# susceptible group utility
			SS_list.append(GDP1 * np.mean(SS) * t_vac)
			# player's expected utility in susceptible group
			U_S.append(np.mean(
				[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in
				 range(len(t_range))]) * t_vac)

			S0_M = 1 - S0_S
			SM, IM, t_range = simulate(beta_M, gamma, S0_M, I_0, t_vac, False)
			# mask group utility
			SM_list.append(GDP2 * np.mean(SM) * t_vac)
			# player's expected utility in mask group
			U_M.append(np.mean(
				[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in
				 range(len(t_range))]) * t_vac)

			socialU.append(SS_list[-1] + SM_list[-1])

		max_social = max(socialU)
		maxIndex = socialU.index(max_social)

		NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, gamma)
		NE_S0_S = NE_S0_S_range[-1]
		POA = max_social / NE_utility

		if POA > max_POA:
			max_POA = POA
			max_paras = [beta_S, beta_M, gamma, GDP1]
			print('POA=', POA)
			print('beta_S=', beta_S)
			print('beta_M=', beta_M)
			print('gamma=', gamma)
			print('GDP1=', GDP1)
			print()

	t2 = time.perf_counter()
	print(f'{round((t2 - t1) / 60, 3)} minutes for {runs} runs')
	return


def main():
	# tmp()
	tests()

	# utility_plotter(income_ratio=6, beta_ratio=0.5, t_vac=100, gamma=GAMMA)
	# utility_plotter_interaction(income_ratio=2.5, beta_ratio=0.5, t_vac=100)
	# POA_grid()
	# POA_monte_carlo(runs=20)

	return


if __name__ == '__main__':
	main()
