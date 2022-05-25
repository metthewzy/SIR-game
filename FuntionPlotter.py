import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

S_0 = 1
I_0 = 0.0001
gamma = 1 / 14
beta_0 = 1


def I_t(S_t):
	return I_0 + gamma / beta * np.log(S_t / S_0) - (S_t - S_0)


def tests():
	print('I_0=', I_t(S_0))
	print('S_peak=', gamma / beta)
	print('I_peak=', I_t(gamma / beta))
	S_range = np.arange(S_0, 0, -0.00001)
	# print(S_range)
	Is = I_t(S_range)
	# print(Is)
	for i in range(1, len(S_range)):
		S_left = S_range[i]
		S_right = S_range[i - 1]
		if Is[i] < I_0:
			break
	print(S_left, S_right)

	while True:
		S_middle = (S_left + S_right) / 2
		I_middle = I_t(S_middle)
		if abs(I_middle - I_0) / I_0 < 0.0001:
			break
		if I_middle < I_0:
			S_left = S_middle
		else:
			S_right = S_middle
	S_end = S_middle
	I_end = I_t(S_end)
	print(f'S_end={S_end} I_end={I_end}')
	S_step = (S_0 - S_end) / 100000
	S_range = np.arange(S_0, S_end - S_step, - S_step)
	# print(S_range)
	dts = del_t(S_range)
	print('t_end=', np.mean(dts) * (S_end - S_0))
	# plot_dt(S_end)
	# plot_ds(S_end)
	return


def del_t(S):
	value = - 1 / (beta * S * (I_0 + gamma / beta * np.log(S / S_0) - (S - S_0)))
	value = - 1 / (beta * S * I_0 + gamma * S * np.log(S / S_0) - beta * S ** 2 + beta * S)
	return value


def del_s_by_del_t(S):
	value = - (beta * S * (I_0 + gamma / beta * np.log(S / S_0) - (S - S_0)))
	return value


# 1 group SIR simulation
def simulate(beta, S0, I0, t_vac, showPlot):
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


# 2 groups simulation with interactions in between
def simulate_interaction(beta1, S1_0, beta2, S2_0, I_0, t_vac, showPlot):
	S1 = [S1_0]
	I1 = [I_0 * S1_0]
	S2 = [S2_0]
	I2 = [I_0 * S2_0]
	dt = 0.01
	t_range = np.arange(0, t_vac + dt, dt)
	for t in t_range[1:]:
		# dS1 = (- S1[-1] * (beta1 * I1[-1] + beta2 * I2[-1])) * dt
		# dI1 = (S1[-1] * (beta1 * I1[-1] + beta2 * I2[-1]) + gamma * I1[-1]) * dt
		dS1 = (- beta1 * S1[-1] * (I1[-1] + I2[-1])) * dt
		dI1 = (beta1 * S1[-1] * (I1[-1] + I2[-1]) + gamma * I1[-1]) * dt

		# dS2 = (- S2[-1] * (beta1 * I1[-1] + beta2 * I2[-1])) * dt
		# dI2 = (S2[-1] * (beta1 * I1[-1] + beta2 * I2[-1]) + gamma * I2[-1]) * dt
		dS2 = (- beta2 * S2[-1] * (I1[-1] + I2[-1])) * dt
		dI2 = (beta2 * S2[-1] * (I1[-1] + I2[-1]) + gamma * I2[-1]) * dt

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


def simulate_scaled(beta, S0, I0, t_vac, dt, showPlot):
	S = [S0]
	I = [I0 * S0]
	# dt = 0.01
	t_range = np.arange(0, t_vac + dt, dt)
	for t in t_range[1:]:
		dS = - min((beta * S[-1] * I[-1]) * dt, S[-1])
		dI = -dS - gamma * I[-1] * dt
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


def simulate_release(beta, S0, I0, H0, t_release, t_vac, dt, showPlot):
	S = [S0]
	I = [I0 * S0]
	H = [H0]
	R = [0]
	released = False
	t_range = np.arange(0, t_vac + dt, dt)
	for t in t_range[1:]:
		dS = - min((beta * S[-1] * I[-1]) * dt, S[-1])
		dI = -dS - gamma * I[-1] * dt
		R.append(R[-1] + gamma * I[-1] * dt)
		S.append(S[-1] + dS)
		I.append(I[-1] + dI)
		H.append(H[-1])
		if not released and t >= t_release:
			released = True
			S[-1] += H[-1]
			H[-1] = 0

	if showPlot:
		fig = plt.figure()
		ax = fig.add_subplot()
		ax.plot(t_range, S, label='S')
		ax.plot(t_range, I, label='I')
		ax.plot(t_range, H, label='H')
		ax.legend()
		plt.show()
		plt.close(fig)
	return S, I, H, R, t_range


def dU_by_dt(income, beta, S_t, I_t, S0, t, t_vac):
	if S0 == 0:
		return income
	value = income - beta * S_t * I_t / S0 * (t_vac - t) * income
	return value


# 2 group game w/o interaction. S:susceptible, M:mask. plotting the player's expected utility in each group,
# and the social utility with varying size of group S
def utility_plotter(income_ratio, beta_ratio, t_vac):
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
		SS, IS, t_range = simulate(beta_S, S0_S, I_0, t_vac, False)
		# susceptible group utility
		SS_list.append(GDP1 * np.mean(SS) * t_vac)
		# player's expected utility in susceptible group
		# U_S.append(np.mean(
		# 	[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)
		U_S.append(GDP1 * t_vac if S0_S == 0 else SS_list[-1] / S0_S)

		S0_M = 1 - S0_S
		SM, IM, t_range = simulate(beta_M, S0_M, I_0, t_vac, False)
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
	NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M)
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


# return the Nash Equilibrium at the last location in the list
def NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M):
	# player utility for group S and M
	U_S = []
	U_M = []

	# trying S0_S=0
	S0_S_range = [0]
	S0_S = S0_S_range[-1]
	SS, IS, t_range = simulate(beta_S, S0_S, I_0, t_vac, False)
	U_S.append(np.mean(
		[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	S0_M = 1 - S0_S
	SM, IM, t_range = simulate(beta_M, S0_M, I_0, t_vac, False)
	U_M.append(np.mean(
		[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	if U_M[-1] > U_S[-1]:
		return S0_S_range, U_S, U_M, U_M[-1]

	# trying S0_S=1
	S0_S_range.append(1)
	S0_S = S0_S_range[-1]
	SS, IS, t_range = simulate(beta_S, S0_S, I_0, t_vac, False)
	U_S.append(np.mean(
		[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	S0_M = 1 - S0_S
	SM, IM, t_range = simulate(beta_M, S0_M, I_0, t_vac, False)
	U_M.append(np.mean(
		[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

	if U_S[-1] > U_M[-1]:
		return S0_S_range, U_S, U_M, U_S[-1]

	left, right, mid = 0, 1, 0.5
	for _ in range(40):
		S0_S_range.append(mid)
		S0_S = S0_S_range[-1]
		SS, IS, t_range = simulate(beta_S, S0_S, I_0, t_vac, False)
		U_S.append(np.mean(
			[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in range(len(t_range))]) * t_vac)

		S0_M = 1 - S0_S
		SM, IM, t_range = simulate(beta_M, S0_M, I_0, t_vac, False)
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


def plot_dt(S_end):
	S_range = np.arange(S_0, S_end, (S_end - S_0) / 1000)
	dt = []
	for s in S_range:
		dt.append(del_t(s))
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(S_range, dt, label='dt')
	plt.show()
	return


def plot_ds(S_end):
	S_range = np.arange(S_0, S_end, (S_end - S_0) / 1000)
	dS = []
	for s in S_range:
		dS.append(del_s_by_del_t(s))
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(S_range, dS, label='dt')
	plt.show()
	return


def curvePlotter():
	t_vac = 60
	SSList = []
	ISList = []
	SMList = []
	IMList = []
	socialU = []
	S0_S_range = np.arange(0 + 0.1, 1 + 0.1, 0.1)
	for S0_S in S0_S_range:
		SS, IS, t_range = simulate(beta_0, S0_S, I_0, t_vac, False)
		SSList.append(SS)
		ISList.append(IS)
		SM, IM, t_range = simulate(beta_0 / 2, 1 - S0_S, I_0, t_vac, False)
		SMList.append(SM)
		IMList.append(IM)

	fig = plt.figure()
	ax = fig.add_subplot()
	for i in range(len(S0_S_range)):
		# / S0_S_range[i]
		ax.plot(t_range, ISList[i] / S0_S_range[i], label=f'S0={round(S0_S_range[i], 2)}')
	ax.legend()
	plt.show()
	plt.close(fig)

	# fig = plt.figure()
	# fig.suptitle('dS/dt/S0')
	# ax = fig.add_subplot()
	# for i in range(len(S0_S_range)):
	# 	curve = []
	# 	S0 = S0_S_range[i]
	# 	for j in range(len(t_range)):
	# 		curve.append(beta_0 * SSList[i][j] * ISList[i][j] / S0)
	# 	ax.plot(t_range, curve, label=f'S0={round(S0, 2)}')
	# ax.legend()
	# plt.show()
	# plt.close(fig)

	return


def scalingBeta():
	t_vac = 60
	SSList = []
	ISList = []
	SMList = []
	IMList = []
	socialU = []
	S0_S_range = np.arange(0 + 0.1, 1 + 0.1, 0.1)
	for S0_S in S0_S_range:
		SS, IS, t_range = simulate(beta_0 * S0_S, 1, I_0, t_vac, False)
		SSList.append(SS)
		ISList.append(IS)
	# SM, IM, t_range = simulate(beta_0 / 2, 1 - S0_S, I_0, t_vac, False)
	# SMList.append(SM)
	# IMList.append(IM)

	# fig = plt.figure()
	# ax = fig.add_subplot()
	# for i in range(len(S0_S_range)):
	# 	S0 = S0_S_range[i]
	# 	# / S0_S_range[i]
	# 	ax.plot(t_range, [beta_0 * S0 * SSList[i][j] * ISList[i][j] for j in
	# 					  range(len(t_range))], label=f'S0={round(S0_S_range[i], 2)}')
	# ax.legend()
	# ax.set_title('beta S I')
	# plt.show()
	# plt.close(fig)

	fig = plt.figure()
	ax = fig.add_subplot()
	for i in range(len(S0_S_range)):
		S0 = S0_S_range[i]
		# / S0_S_range[i]
		ax.plot(t_range, ISList[i], label=f'S0={round(S0_S_range[i], 2)}')
	ax.legend()
	ax.set_ylabel('I')
	ax.set_xlabel('t')
	plt.show()
	plt.close(fig)

	return


def plotI():
	S_range = np.arange(0, 1.01, 0.01)
	I = []
	beta_range = np.arange(1, 5, 0.5)
	for beta in beta_range:
		I_list = []
		for S in S_range:
			I_list.append(0.0001 + gamma / beta * np.log(S) - (S - 1))
		I.append(I_list)

	fig = plt.figure()
	ax = fig.add_subplot()
	for j in range(len(beta_range)):
		beta = beta_range[j]
		ax.plot(S_range, I[j], label=f'beta={round(beta, 2)}')
	ax.legend()
	xl, xu = ax.get_xlim()
	ax.set_xlim(xu, xl)
	ax.set_xlabel('S')
	ax.set_ylabel('I')
	plt.show()
	plt.close(fig)

	return


def compare_scaled():
	t_vac = 60

	dts = [0.01, 0.1, 0.5, 1]
	fig = plt.figure()
	axes = fig.subplots(len(dts), 1)
	# ax2 = fig.add_subplot(312)
	# ax3 = fig.add_subplot(313)
	for i in range(len(dts)):
		dt = dts[i]
		ax = axes[i]
		ax.set_title(f'dt={round(dt, 4)}')
		S, I, t_range = simulate_scaled(beta_0, 1, I_0, t_vac, dt, False)
		ax.plot(t_range, S, label='S')
		ax.plot(t_range, I, label='I')
		ax.legend()

	# dt = 1
	# S1, I1, t_range1 = simulate_scaled(beta_0, 1, I_0, t_vac, dt, False)
	#
	# dt = 4
	# S4, I4, t_range4 = simulate_scaled(beta_0, 1, I_0, t_vac, dt, False)
	#
	# dt = 0.01
	# S, I, t_range = simulate_scaled(beta_0, 1, I_0, t_vac, dt, False)

	# print(S1)
	# print(S4)
	# print(S)

	# ax1.plot(t_range1, S1, label='S 1 day')
	# ax1.plot(t_range1, I1, label='I 1 day')
	# ax2.plot(t_range4, S4, label='S 4 day')
	# ax2.plot(t_range4, I4, label='I 4 day')
	# ax3.plot(t_range, S, label='S')
	# ax3.plot(t_range, I, label='I')
	# ax1.legend()
	# ax2.legend()
	# ax3.legend()
	fig.subplots_adjust(hspace=1)
	plt.show()
	plt.close(fig)
	return


def release_integral():
	t_vac = 200
	t_rcvr = 30
	dt = 0.01
	H0 = 0.5
	S_int = []
	H_int = []
	I_int = []
	R_int = []
	S_curves = []
	I_curves = []
	H_curves = []
	R_curves = []
	cum_I = []

	for t_release in range(t_vac):
		S, I, H, R, t_range = simulate_release(1, 1, I_0, H0, t_release, t_vac, dt, False)
		S_int.append(sum(S) * dt)
		H_int.append(sum(H) * dt)
		I_int.append(sum(I) * dt)
		S_curves.append(S)
		I_curves.append(I)
		H_curves.append(H)
		R = [R[i] for i in range(len(R)) if t_range[i] <= t_vac - t_rcvr]
		# print(len(dR))
		R_curves.append(R)
		R_int.append(sum(R) * dt)
		cum_I.append(1.5 - S[-1])

	fig = plt.figure()
	axes = fig.subplots(4, 2)
	axes[0][0].plot(range(t_vac), S_int, label='int S')
	axes[0][0].plot(range(t_vac), R_int, label='int R')
	axes[0][0].legend()
	axes[0][1].plot(range(t_vac), H_int)
	axes[1][0].plot(range(t_vac), I_int)
	axes[0][0].set_title('S integral')
	axes[1][0].set_title('I integral')
	axes[0][1].set_title('H integral')
	axes[1][1].set_title('Social')
	axes[1][1].plot(range(t_vac),
					[S_int[i] + 0.97 * R_int[i] - H_int[i] / 20 - cum_I[i] * 90 for i in range(t_vac)])
	axes[2][0].set_title('S')
	for i in range(t_vac - 20, t_vac):
		axes[2][0].plot(t_range, S_curves[i])
	axes[2][1].set_title('I')
	for i in range(t_vac - 20, t_vac):
		axes[2][1].plot(t_range, I_curves[i])
	axes[3][0].set_title('R (releasing on day 0 to 30)')
	for i in range(0, 30, 5):
		axes[3][0].plot(t_range[:len(R_curves[i])], R_curves[i])
	fig.subplots_adjust(hspace=0.6)
	plt.show()
	plt.close(fig)
	return


def tmp():
	a = [1, 2, 3, 4, 5]
	b = [i for i in a if i < 4]
	print(b)
	return


# heatmap for POA over different daily income ratios and beta ratios
def POA_grid():
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
				SS, IS, t_range = simulate(beta_S, S0_S, I_0, t_vac, False)
				# susceptible group utility
				SS_list.append(GDP1 * np.mean(SS) * t_vac)
				# player's expected utility in susceptible group
				U_S.append(np.mean(
					[dU_by_dt(GDP1, beta_S, SS[i], IS[i], S0_S, t_range[i], t_vac) for i in
					 range(len(t_range))]) * t_vac)

				S0_M = 1 - S0_S
				SM, IM, t_range = simulate(beta_M, S0_M, I_0, t_vac, False)
				# mask group utility
				SM_list.append(GDP2 * np.mean(SM) * t_vac)
				# player's expected utility in mask group
				U_M.append(np.mean(
					[dU_by_dt(GDP2, beta_M, SM[i], IM[i], S0_M, t_range[i], t_vac) for i in
					 range(len(t_range))]) * t_vac)

				socialU.append(SS_list[-1] + SM_list[-1])

			max_social = max(socialU)
			maxIndex = socialU.index(max_social)

			NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M)
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


# 2 group game with interaction. S:susceptible, M:mask. plotting the player's expected utility in each group,
# and the social utility with varying size of group S
def utility_plotter_interaction(income_ratio, beta_ratio, t_vac):
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
		S0_M = 1 - S0_S
		SS, IS, SM, IM, t_range = simulate_interaction(beta_S, S0_S, beta_M, S0_M, I_0, t_vac, False)

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

	# search for the NE point
	NE_S0_S_range, NE_U_S, NE_U_M, NE_utility = NE_searcher(t_vac, GDP1, GDP2, beta_S, beta_M)
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


def main():
	# tests()

	# utility_plotter(income_ratio=6, beta_ratio=0.5, t_vac=100)
	utility_plotter_interaction(income_ratio=2, beta_ratio=0.5, t_vac=100)
	# POA_grid()

	# curvePlotter()
	# scalingBeta()
	# plotI()
	# compare_scaled()
	# release_integral()
	# tmp()
	return


if __name__ == '__main__':
	main()
