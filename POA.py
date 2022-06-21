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


def t_peak_comparison():
	S0 = 1
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	beta = 0.2
	I0 = I0_global * S0
	S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(t_range, S, label='S')
	ax1.plot(t_range, I, label='I')

	S_peak = gamma / beta
	S_peak_idx = [i for i in range(len(S)) if S[i] >= S_peak][-1]
	t_peak = t_range[S_peak_idx]
	ax1.vlines(x=t_peak, ymin=0, ymax=S_peak, label='t_peak', color='red', linestyle=':')

	C1 = S0 / (gamma - beta * S0)
	C2 = (beta * (I0 + S0) - gamma) * S0 / (gamma - beta * S0)
	t_first_half = - C1 / C2 * np.log((S0 / 2) / (S0 / 2 + C2) * (S0 + C2) / S0)
	C1 = S0 / (beta * S0 - 2 * gamma)
	C2 = gamma - gamma * np.log(0.5) - beta * I0 - beta * S0
	t_second_half = 1 / C2 * np.log((gamma / beta) / (gamma / beta + C1 * C2) * (S0 / 2 + C1 * C2) / (S0 / 2))
	t_peak_est = t_first_half + t_second_half
	ax1.vlines(x=t_peak_est, ymin=0, ymax=S_peak, label='t_peak_est', color='blue', linestyle=':')

	ax1.vlines(x=t_first_half, ymin=0, ymax=S0 / 2, color='grey')
	ax1.hlines(y=S0 / 2, xmin=0, xmax=t_first_half, color='grey')
	ax1.hlines(y=S_peak, xmin=t_first_half, xmax=t_peak, label='S_peak', color='grey', linestyle=':')
	print(t_first_half)

	# ax1.set_xlim(0, 20)
	ax1.set_title(f'beta={round(beta, 3)}')
	ax1.legend()

	plt.show()
	return


def main():
	# POA_plot()
	t_peak_comparison()
	return


if __name__ == '__main__':
	main()
