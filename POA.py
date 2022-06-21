import numpy as np
import matplotlib.pyplot as plt
from FuntionPlotter import simulate


def POA_plot():
	gamma = 1 / 14
	S_0 = 1
	I_0 = 0.0001
	step = 0.025
	beta_range = np.arange(0.1, 1 + step, step)
	areas = []
	for beta in beta_range:
		C1 = S_0 / (gamma - beta * S_0)
		C2 = (beta * (I_0 + S_0) - gamma) * S_0 / (gamma - beta * S_0)
		t_peak = - C1 / C2 * np.log((gamma / beta) / (gamma / beta + C2) * (S_0 + C2) / S_0)
		areas.append(t_peak * gamma / beta)
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(beta_range, areas)
	ax.axhline(0, linestyle=':', color='r')
	ax.set_ylabel('rectangle area')
	ax.set_xlabel('beta')
	plt.show()
	return


def t_peak_area_comparison():
	S0 = 0.24
	I0_global = 0.0001
	t_vac = 100
	gamma = 0.20400199969871927
	beta = 1.1753611419432302
	I0 = I0_global * S0
	S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(t_range, S, label='S')
	ax1.plot(t_range, I, label='I')

	S_peak = gamma / beta
	# S_peak_idx = [i for i in range(len(S)) if S[i] >= S_peak][-1]
	# t_peak = t_range[S_peak_idx]
	# ax1.vlines(x=t_peak, ymin=0, ymax=S_peak, label='t_peak', color='red', linestyle=':')

	# t_first_half = t_first(S0, S0 / 2, beta, gamma, S0, I0)
	# t_second_half = t_second(S0 / 2, S_peak, beta, gamma, S0, I0)

	t_first_half = t_first(S0, S0 / 3 * 2, beta, gamma, S0, I0)
	t_second_half = t_second_V2(S0 / 3 * 2, S_peak, beta, gamma, S0, I0)

	t_peak_est = t_first_half + t_second_half

	ax1.vlines(x=t_peak_est, ymin=0, ymax=S_peak, label='t_peak_est', color='blue', linestyle=':')

	# ax1.vlines(x=t_first_half, ymin=0, ymax=S0 / 2, color='grey')
	# ax1.hlines(y=S0 / 2, xmin=0, xmax=t_first_half, color='grey')

	ax1.vlines(x=t_first_half, ymin=0, ymax=S0 / 3 * 2, color='grey')
	ax1.hlines(y=S0 / 3 * 2, xmin=0, xmax=t_first_half, color='grey')

	ax1.hlines(y=S_peak, xmin=t_first_half, xmax=t_vac, label='S_peak', color='grey', linestyle=':')

	# ax1.set_xlim(0, 20)
	ax1.set_title(f'beta={round(beta, 3)}')
	ax1.legend()
	S_area = np.mean(S) * t_vac
	print(S_area)
	# rectangle_area = S0 / 2 * t_first_half + S_peak * t_second_half
	rectangle_area = S0 / 3 * 2 * t_first_half + S_peak * t_second_half
	print(rectangle_area)
	print(f'{round(rectangle_area / S_area * 100, 2)}% of the actual area')

	plt.show()
	return


def t_first(S1, S2, beta, gamma, S0, I0):
	C1 = S0 / (gamma - beta * S0)
	C2 = (beta * (I0 + S0) - gamma) * S0 / (gamma - beta * S0)
	ret = - C1 / C2 * np.log(S2 / (S2 + C2) * (S1 + C2) / S1)
	return ret


def t_second(S1, S2, beta, gamma, S0, I0):
	C1 = S0 / (beta * S0 - 2 * gamma)
	C2 = gamma - gamma * np.log(0.5) - beta * I0 - beta * S0
	ret = np.log(S2 / (S2 + C1 * C2) * (S1 + C1 * C2) / S1) / C2
	return ret


def t_second_V2(S1, S2, beta, gamma, S0, I0):
	C1 = 2 * S0 / (2 * beta * S0 - 3 * gamma)
	C2 = gamma - gamma * np.log(2 / 3) - beta * I0 - beta * S0
	ret = np.log(S2 / (S2 + C1 * C2) * (S1 + C1 * C2) / S1) / C2
	return ret


def area_against_beta():
	beta_range = np.arange(10, 0.15, -0.01)
	S0 = 1
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	I0 = I0_global * S0
	areas_est = []
	areas_act = []
	ts = []
	percentages = []
	for beta in beta_range:
		S_peak = gamma / beta
		t_first_half = t_first(S0, S0 / 3 * 2, beta, gamma, S0, I0)
		t_second_half = t_second_V2(S0 / 3 * 2, S_peak, beta, gamma, S0, I0)
		t_peak_est = t_first_half + t_second_half
		ts.append(t_peak_est)
		areas_est.append(t_first_half * 2 / 3 * S0 + t_second_half * S_peak)
		# S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
		# areas_act.append(np.mean(S) * t_vac)
		# percentages.append(areas_est[-1] / areas_act[-1])
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(beta_range, areas_est, label='estimated')
	# ax1.plot(beta_range, areas_act, label='actual')
	ax2.plot(beta_range, ts)
	ax1.set_xlabel('beta')
	ax1.set_ylabel('area')
	ax2.set_xlabel('beta')
	ax2.set_ylabel('t_peak_est')
	ax_tw = ax1.twinx()
	# ax_tw.plot(beta_range, percentages, linestyle=':', color='r')
	ax_tw.set_ylim(0.5, 1)
	ax_tw.set_ylabel('est/act')
	ax1.legend()
	print(min(areas_est))
	plt.show()
	return


def main():
	# POA_plot()
	# t_peak_area_comparison()
	area_against_beta()
	return


if __name__ == '__main__':
	main()
