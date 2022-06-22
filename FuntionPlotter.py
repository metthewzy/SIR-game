import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import uniform as uni
import time
import concurrent.futures
import traceback
import sys
from scipy.optimize import minimize

S_0 = 1
I_0 = 0.0001
GAMMA = 1 / 14
BETA_0 = 1
separate_betas = True
show_figure = True

BETA_RANGE = (0.05, 1)
GAMMA_RANGE = (1 / 20, 1 / 5)
INCOME_RANGE = (1, 50)
BETA_RATIO_RANGE = (0.01, 1)

NUM_THREADS = 18


def dummy_worker(dummy_id):
	for _ in range(1000000000):
		_ += 0
	return dummy_id


def tests():
	"""
	trying multithreading
	"""
	with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
		results = [executor.submit(dummy_worker, i) for i in range(10)]

		try:
			for f in concurrent.futures.as_completed(results):
				dummy_id = f.result()
				print(dummy_id, "completed")
		except:
			traceback.print_exception(*sys.exc_info())
	return


def simulate(beta, gamma, S0, I0, t_vac, showPlot, num_steps=10000):
	"""
	1 group SIR simulation
	"""
	S = [S0]
	I = [I0 * S0]
	# dt = 0.01
	dt = t_vac / num_steps
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
	2 groups simulation with interactions in between. 4 betas for each S-I combination
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


def utility_plotter(beta, income_ratio, beta_ratio, gamma, t_vac):
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
	beta_S = beta
	beta_M = beta * beta_ratio
	step_size = 0.01
	S0_S_range = np.arange(0, 1 + step_size, step_size)
	# S0_S_range = np.arange(0 + step_size, 1, step_size)
	for S0_S in S0_S_range:
		SS, IS, t_range = simulate(beta_S, gamma, S0_S, I_0, t_vac, False)
		# susceptible group utility
		SS_list.append(GDP1 * np.mean(SS) * t_vac)
		# player's expected utility in susceptible group
		U_S.append(np.mean(
			[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)
		# U_S.append(GDP1 * t_vac if S0_S == 0 else SS_list[-1] / S0_S)

		S0_M = 1 - S0_S
		SM, IM, t_range = simulate(beta_M, gamma, S0_M, I_0, t_vac, False)
		# mask group utility
		SM_list.append(GDP2 * np.mean(SM) * t_vac)
		# player's expected utility in mask group
		U_M.append(np.mean(
			[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)
		# U_M.append(GDP2 * t_vac if S0_M == 0 else SM_list[-1] / S0_M)

		socialU.append(SS_list[-1] + SM_list[-1])

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	# ax1.plot(S0_S_range, U_S, label='U(Player)_S')
	# ax1.plot(S0_S_range, U_M, label='U(Player)_M')
	ax1.plot(S0_S_range, SS_list, label='UG(S)')
	ax1.plot(S0_S_range, SM_list, label='UG(M)')
	ax1.plot(S0_S_range, socialU, label='social')
	max_social = max(socialU)
	maxIndex = socialU.index(max_social)
	ax1.axhline(max_social, linestyle=':', label=f'OPT={round(max_social, 4)}', color='red')
	ax1.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')
	ax1.set_xlabel('susceptible size')
	ax1.set_ylabel('utility')

	# # dGroup utility by d phi(S)
	# ax3 = fig.add_subplot(223)
	# dGroup_utility_S = [SS_list[i] - SS_list[i - 1] for i in range(1, len(SS_list))]
	# ax3.plot(S0_S_range[1:], dGroup_utility_S, label='dUG(S)')
	#
	# dGroup_utility_M = [SM_list[i] - SM_list[i - 1] for i in range(1, len(SM_list))]
	# ax3.plot(S0_S_range[1:], dGroup_utility_M, label='dUG(M)')
	#
	# sum_dUG = [dGroup_utility_S[i] + dGroup_utility_M[i] for i in range(len(dGroup_utility_S))]
	# ax3.plot(S0_S_range[1:], sum_dUG, label='sum dUG')
	# # ax3.axhline(sum_dUG[maxIndex], linestyle=':', label=f'OPT={round(sum_dUG[maxIndex], 4)}', color='red')
	# ax3.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')
	# ax3.legend()

	# search for the NE point
	NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, gamma)
	NE_S0_S = NE_S0_S_range[-1]
	POA = max_social / NE_utility
	print('POA=', POA)
	# NE_S0_S_range, NE_U_S, NE_U_M = map(list, zip(*sorted(zip(NE_S0_S_range, NE_U_S, NE_U_M))))
	ax2 = fig.add_subplot(122)
	# ax2.plot(NE_S0_S_range, NE_U_S, label='U(Player)_S')
	# ax2.plot(NE_S0_S_range, NE_U_M, label='U(Player)_M')
	ax2.plot(S0_S_range, U_S, label='U(S)')
	ax2.plot(S0_S_range, U_M, label='U(M)')
	ax2.axhline(NE_utility, label=f'NE={round(NE_utility, 4)}', linestyle=':', color='grey')
	ax2.axvline(NE_S0_S, label=f'NE@{round(NE_S0_S, 4)}', linestyle=':', color='grey')

	ax2.set_xlabel('susceptible size')
	ax2.set_ylabel('utility')

	ax1.axvline(NE_S0_S, label=f'NE@{round(NE_S0_S, 4)}', linestyle=':', color='grey')
	ax2.axvline(S0_S_range[maxIndex], linestyle=':', color='red', label=f'OPT@{round(S0_S_range[maxIndex], 4)}')

	ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
	ax2.legend(loc='upper left', bbox_to_anchor=(0, -0.15))
	# fig.suptitle(f'POA={round(max_social / NE_utility, 4)}')
	fig.suptitle('POA={:.6f}'.format(POA))
	plt.tight_layout()
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


def POA_heatmap():
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
		beta_S = BETA_0
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
	ax.set_title('POA @NE')
	print('max POA=', max_POA)
	plt.show()

	return


def POA_heatmap_V2(beta, t_vac, income_ratio):
	"""
	Heatmap of POA w.r.t. beta ratio and gamma
	"""
	# income_ratios = [1, 2, 4, 6, 8, 10]
	beta_ratios = [1 / 4, 1 / 5, 1 / 7, 1 / 9, 1 / 10, 1 / 11, 1 / 12]
	# gammas = [1 / 3, 1 / 4, 1 / 5, 1 / 7, 1 / 9, 1 / 10]
	gamma_step = (1 / 4 - 1 / 6) / 5
	gammas = np.arange(1 / 6, 1 / 4 + gamma_step, gamma_step)
	POAs = []
	# max_POA = 0
	for gamma in gammas:
		POAs.append([])
		GDP1 = income_ratio
		GDP2 = 1
		beta_S = beta
		for beta_ratio in beta_ratios:
			print(gamma, beta_ratio)
			U_S = []
			U_M = []
			SS_list = []
			SM_list = []
			socialU = []
			beta_M = beta_S * beta_ratio
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
			POAs[-1].append(POA)
	# max_POA = max(max_POA, POA)
	ax = sns.heatmap(POAs, annot=True, fmt=".4f")
	ax.set_xticklabels([round(i, 4) for i in beta_ratios])
	ax.set_yticklabels([round(i, 4) for i in gammas])
	ax.set_xlabel('beta ratio')
	ax.set_ylabel('gamma')
	ax.set_title('POA')
	# print('max POA=', max_POA)
	plt.show()
	return


def tmp():
	beta_S = 0.9794676182860252
	beta_ratio = 0.148881166
	income_ratio = 8.47300431687476
	gamma = 0.17000166641559938
	max_POA = 1
	with open('POA.txt', 'w') as f:
		f.write(f'MAX POA=\n{max_POA}\n\n')
		f.write(f'beta_S=\n{beta_S}\n\n')
		f.write(f'beta_ratio=\n{beta_ratio}\n\n')
		f.write(f'income_ratio=\n{income_ratio}\n\n')
		f.write(f'gamma=\n{gamma}\n')
	return


def utility_plotter_interaction(income_ratio, beta_ratio, gamma, t_vac):
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
	beta_S = BETA_0
	beta_M = BETA_0 * beta_ratio

	beta_SS = BETA_0
	beta_SM = BETA_0 * beta_ratio
	beta_MS = BETA_0 * beta_ratio ** 2
	beta_MM = BETA_0 * beta_ratio ** 3

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


def POA_Monte_Carlo(runs):
	"""
	Monte Carlo to generate max POA. 2 group game without interaction
	"""
	np.random.seed()

	t_vac = 100
	max_POA = 0
	max_paras = []

	t1 = time.perf_counter()
	with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
		num_threads = 0
		results = [executor.submit(POA_Monte_Carlo_calculator, t_vac) for _ in range(runs)]

		try:
			for f in concurrent.futures.as_completed(results):
				POA, [beta_S, beta_M, gamma, GDP1] = f.result()
				num_threads += 1
				if num_threads % round(runs / 20) == 0:
					print(f'{num_threads} / {runs} completed')
				if POA > max_POA:
					max_POA = POA
					max_paras = [beta_S, beta_M, gamma, GDP1]
					print('MAX POA updated at run', num_threads)
		except:
			traceback.print_exception(*sys.exc_info())

	t2 = time.perf_counter()
	print(f'\n{round((t2 - t1) / 60, 3)} minutes for {runs} runs')
	[beta_S, beta_M, gamma, GDP1] = max_paras
	print('MAX POA=', max_POA)
	print('beta_S=', beta_S)
	print('beta_M=', beta_M)
	print('gamma=', gamma)
	print('GDP1=', GDP1)
	return


def POA_Monte_Carlo_calculator(t_vac):
	"""
	Calculate POA with Monte Carlo
	"""
	beta_S = uni(BETA_RANGE[0], BETA_RANGE[1])
	gamma = uni(GAMMA_RANGE[0], GAMMA_RANGE[1])
	GDP1 = uni(INCOME_RANGE[0], INCOME_RANGE[1])
	GDP2 = 1
	beta_M = beta_S * uni(BETA_RATIO_RANGE[0], BETA_RATIO_RANGE[1])
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
	return POA, [beta_S, beta_M, gamma, GDP1]


def OPT_heatmap(beta, t_vac, gamma):
	"""
	Heatmap of social OPT susceptible size w.r.t. beta ratio and income ratio
	"""
	income_ratios = [1, 2, 4, 6, 8, 10]
	beta_ratios = [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10]
	OPT_S_sizes = []
	# max_POA = 0
	for GDP1 in income_ratios:
		OPT_S_sizes.append([])
		GDP2 = 1
		beta_S = beta
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

			# NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, gamma)
			# NE_S0_S = NE_S0_S_range[-1]
			# POA = max_social / NE_utility
			OPT_S_sizes[-1].append(S0_S_range[maxIndex])
	# max_POA = max(max_POA, POA)
	ax = sns.heatmap(OPT_S_sizes, annot=True, fmt=".3f")
	ax.set_xticklabels([round(i, 2) for i in beta_ratios])
	ax.set_yticklabels(income_ratios)
	ax.set_xlabel('beta ratio')
	ax.set_ylabel('income ratio')
	ax.set_title('Susceptible size @OPT')
	# print('max POA=', max_POA)
	plt.show()
	return


def OPT_heatmap_V2(beta, t_vac, income_ratio):
	"""
	Heatmap of social OPT susceptible size w.r.t. beta ratio and gamma
	"""
	# income_ratios = [1, 2, 4, 6, 8, 10]
	beta_ratios = [1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10]
	gammas = [1 / 3, 1 / 7, 1 / 10, 1 / 14, 1 / 21]
	OPT_S_sizes = []
	# max_POA = 0
	for gamma in gammas:
		OPT_S_sizes.append([])
		GDP1 = income_ratio
		GDP2 = 1
		beta_S = beta
		for beta_ratio in beta_ratios:
			print(gamma, beta_ratio)
			U_S = []
			U_M = []
			SS_list = []
			SM_list = []
			socialU = []
			beta_M = beta_S * beta_ratio
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

			# NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M, gamma)
			# NE_S0_S = NE_S0_S_range[-1]
			# POA = max_social / NE_utility
			OPT_S_sizes[-1].append(S0_S_range[maxIndex])
	# max_POA = max(max_POA, POA)
	ax = sns.heatmap(OPT_S_sizes, annot=True, fmt=".3f")
	ax.set_xticklabels([round(i, 2) for i in beta_ratios])
	ax.set_yticklabels([round(i, 2) for i in gammas])
	ax.set_xlabel('beta ratio')
	ax.set_ylabel('gamma')
	ax.set_title('Susceptible size @OPT')
	# print('max POA=', max_POA)
	plt.show()
	return


def POA_MT_optimizer(beta_S, beta_ratio, income_ratio, gamma, t_vac, runs):
	"""
	Search for a max POA with multiprocessing
	"""
	np.random.seed()

	beta_S_orig, beta_ratio_orig, income_ratio_orig, gamma_orig = beta_S, beta_ratio, income_ratio, gamma

	# beta_S = 0.9794676182860252
	# beta_ratio = 0.148881166
	# income_ratio = 8.47300431687476
	# gamma = 0.17000166641559938
	# t_vac = 100
	initial_paras = [beta_S, beta_ratio, income_ratio, gamma]
	max_POA = -POA_calculator(initial_paras, t_vac)
	max_paras = [beta_S, beta_ratio, income_ratio, gamma]
	print('initial max POA=', max_POA)

	t1 = time.perf_counter()
	with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
		num_threads = 0
		results = [executor.submit(POA_optimizer, initial_paras, t_vac) for _ in range(runs)]

		try:
			for f in concurrent.futures.as_completed(results):
				POA, paras = f.result()
				num_threads += 1
				if num_threads % max(round(runs / 20), 1) == 0:
					print(f'{num_threads} / {runs} completed')
				if POA > max_POA:
					max_POA = POA
					max_paras = paras
					print('MAX POA updated at run', num_threads)
					with open('POA.txt', 'w') as fl:
						[beta_S, beta_ratio, income_ratio, gamma] = max_paras
						fl.write(f'MAX POA=\n{max_POA}\n\n')
						fl.write(f'beta_S=\n{beta_S}\n\n')
						fl.write(f'beta_ratio=\n{beta_ratio}\n\n')
						fl.write(f'income_ratio=\n{income_ratio}\n\n')
						fl.write(f'gamma=\n{gamma}\n')
		except:
			traceback.print_exception(*sys.exc_info())

	t2 = time.perf_counter()
	print(f'\n{round((t2 - t1) / 60, 3)} minutes for {runs} runs')
	[beta_S, beta_ratio, income_ratio, gamma] = max_paras
	print('MAX POA=', max_POA)
	print('beta_S=', beta_S)
	print(beta_S / beta_S_orig)
	print('beta_ratio=', beta_ratio)
	print(beta_ratio / beta_ratio_orig)
	print('income_ratio=', income_ratio)
	print(income_ratio / income_ratio_orig)
	print('gamma=', gamma)
	print(gamma / gamma_orig)

	return


def POA_optimizer(paras, t_vac):
	"""
	Maximize for a max POA around a given parameter point
	"""
	low, high = 0.4, 3
	beta_S, beta_ratio, income_ratio, gamma = paras
	beta_S_range = (beta_S * low, beta_S * high)
	beta_ratio_range = (beta_ratio * low, beta_ratio * high)
	income_ratio_range = (income_ratio * low, income_ratio * high)
	gamma_range = (gamma * low, gamma * high)
	optimal = minimize(POA_calculator,
					   [uni(beta_S_range[0], beta_S_range[1]),
						uni(beta_ratio_range[0], beta_ratio_range[1]),
						uni(income_ratio_range[0], income_ratio_range[1]),
						uni(gamma_range[0], gamma_range[1])],
					   args=t_vac,
					   method='L-BFGS-B',
					   bounds=[beta_S_range,
							   beta_ratio_range,
							   income_ratio_range,
							   gamma_range])
	POA = -POA_calculator(optimal.x, t_vac)
	return POA, optimal.x


def POA_calculator(paras, t_vac):
	"""
	Compute POA with given parameters
	Note that return is -POA
	"""
	beta_S, beta_ratio, income_ratio, gamma = paras

	GDP1 = income_ratio
	GDP2 = 1
	U_S = []
	U_M = []
	SS_list = []
	SM_list = []
	socialU = []
	beta_M = beta_S * beta_ratio
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
	return -POA


def main():
	# tmp()
	# tests()

	# utility_plotter(beta=BETA_0, income_ratio=6, beta_ratio=0.5, t_vac=100, gamma=GAMMA)
	# utility_plotter_interaction(income_ratio=2.5, beta_ratio=0.5, t_vac=100)
	# POA_heatmap()
	# POA_Monte_Carlo(runs=20000)

	# # max POA found
	# utility_plotter(beta=1.1753611419432302,
	# 				income_ratio=10.12476515963674,
	# 				beta_ratio=0.11910493280000001,
	# 				gamma=0.20400199969871927,
	# 				t_vac=100)

	# OPT_heatmap(beta=1, t_vac=100, gamma=1 / 14)

	# OPT_heatmap_V2(beta=1, t_vac=100, income_ratio=8.47300431687476)
	# POA_heatmap_V2(beta=1, t_vac=100, income_ratio=8.47300431687476)

	# utility_plotter(beta=1.1753611419432302,
	# 				income_ratio=5,
	# 				beta_ratio=0.7,
	# 				gamma=0.20400199969871927,
	# 				t_vac=100)

	POA_MT_optimizer(beta_S=0.9794676182860252,
	                 beta_ratio=0.148881166,
	                 income_ratio=8.47300431687476,
	                 gamma=0.17000166641559938,
	                 t_vac=100,
	                 runs=50)

	return


if __name__ == '__main__':
	main()
