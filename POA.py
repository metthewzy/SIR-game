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
	S0 = 0.2
	I0_global = 0.0001
	t_vac = 400
	gamma = 0.5100049992467981
	beta = 2.938402854858076
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


def area_lower_bound_against_beta():
	beta_range = np.arange(10, 0.15, -0.1)
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
		S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
		areas_act.append(np.mean(S) * t_vac)
		percentages.append(areas_est[-1] / areas_act[-1])
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(beta_range, areas_est, label='estimated')
	ax1.plot(beta_range, areas_act, label='actual')
	ax2.plot(beta_range, ts)
	ax1.set_xlabel('beta')
	ax1.set_ylabel('area')
	ax2.set_xlabel('beta')
	ax2.set_ylabel('t_peak_est')
	ax_tw = ax1.twinx()
	ax_tw.plot(beta_range, percentages, linestyle=':', color='r')
	# ax_tw.set_ylim(0.5, 1)
	ax_tw.set_ylabel('est/act')
	ax1.legend()
	print(min(areas_est))
	plt.show()
	return


def area_upper_bound_against_beta():
	beta_range = np.arange(10, 0.15, -0.1)
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
		areas_est.append(t_first_half * S0 + t_second_half * 2 / 3 * S0 + (t_vac - t_peak_est) * S_peak)
		S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
		areas_act.append(np.mean(S) * t_vac)
		percentages.append(areas_act[-1] / areas_est[-1])
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(beta_range, areas_est, label='estimated')
	ax1.plot(beta_range, areas_act, label='actual')
	ax2.plot(beta_range, ts)
	ax1.set_xlabel('beta')
	ax1.set_ylabel('area')
	ax2.set_xlabel('beta')
	ax2.set_ylabel('t_peak_est')
	ax_tw = ax1.twinx()
	ax_tw.plot(beta_range, percentages, linestyle=':', color='r')
	# ax_tw.set_ylim(0.5, 1)
	ax_tw.set_ylabel('est/act')
	ax1.legend()
	print(min(areas_est))
	plt.show()
	return


def second_derivative():
	phi_step = 0.01
	phi_range = np.arange(phi_step, 1 + phi_step, phi_step)
	phi = 0.2
	beta = 1
	S0 = phi
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	I0 = I0_global * phi
	S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
	S_peak = gamma / beta
	print(f'S_peak={S_peak}')
	dS = [0]
	dS.extend([S[i] - S[i - 1] for i in range(1, len(S))])
	d2S = [0]
	d2S.extend([dS[i] - dS[i - 1] for i in range(1, len(dS))])

	flip_idx1 = [i for i in range(1, len(d2S)) if (S[i] - I[i] - S_peak) * (S[i - 1] - I[i - 1] - S_peak) <= 0][-1]
	print(f'S-I intercepts S_peak at {t_range[flip_idx1]}')

	flip_idx2 = [i for i in range(1, len(d2S)) if d2S[i] * d2S[i - 1] <= 0][-1]
	print(f'd2S flipping at {t_range[flip_idx2]}')

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(t_range, S, label='S')
	ax1.plot(t_range, I, label='I')

	ax1.plot(t_range, [I0 * np.exp(t * (beta * phi - gamma)) for t in t_range])
	ax1.plot(t_range, [I0 * np.exp(t * (beta * phi * (1 - 0.5) - gamma)) for t in t_range])
	# ax1.plot(t_range, [S[i] - I[i] for i in range(len(S))], label='S-I')
	ax2.plot(t_range, d2S, label='d2S')

	ax1.axhline(phi * 0.5)
	# ax1.axhline(S_peak, label='S_peak', linestyle=':')
	ax2.axhline(0, linestyle=':')

	# ax1.axvline(t_range[flip_idx1], linestyle=':')
	ax2.axvline(t_range[flip_idx2], linestyle=':')

	ax1.legend()
	ax2.legend()
	# ax1.set_xlim(0, 60)
	# ax1.set_ylim(0, phi)
	plt.show()
	return


def area_comparison():
	phi_step = 0.01
	phi_range = np.arange(phi_step, 1, phi_step)
	S1_areas = []
	S2_areas = []
	approximated_areas1 = []
	approximated_areas2 = []
	t1s = []
	t2s = []
	payment_ratio = 10
	I0_global = 0.0001
	t_vac = 1000
	gamma = 1 / 14
	for phi in phi_range:
		beta = 1
		S0 = phi
		I0 = I0_global * S0
		S1, I1, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
		S_peak = gamma / beta
		S1_area = np.mean(S1) * t_vac * payment_ratio
		S1_areas.append(S1_area)
		t1 = t1_searcher(S1, I1, t_range, S_peak, beta, t_vac)
		t1s.append(t1)
		i = list(t_range).index(t1)
		approximated_area1 = (S0 * t1 + (S1[i] + S_peak) * (t_vac - t1) / 2) * payment_ratio
		approximated_areas1.append(approximated_area1)

		# plot_area(S1, t_range, S_peak, i, t_vac, S1_area, approximated_area1)

		beta = 0.5
		S0 = 1 - phi
		I0 = I0_global * S0
		S2, I2, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
		S_peak = gamma / beta
		S2_area = np.mean(S2) * t_vac
		S2_areas.append(S2_area)
		t2 = t1_searcher(S2, I2, t_range, S_peak, beta, t_vac)
		t2s.append(t2)
		i = list(t_range).index(t2)
		approximated_area2 = S0 * t2 + (S2[i] + S_peak) * (t_vac - t2) / 2
		approximated_areas2.append(approximated_area2)

		plot_area(S2, t_range, S_peak, i, t_vac, S2_area, approximated_area2)

	fig = plt.figure()
	# ax1 = fig.add_subplot()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(phi_range, [S1_areas[i] + S2_areas[i] for i in range(len(S1_areas))], label='actual utility')
	ax2.plot(phi_range, [approximated_areas1[i] + approximated_areas2[i] for i in range(len(approximated_areas1))],
			 label='approx utility')
	ax1.plot(phi_range, S1_areas, label='actual 1')
	ax1.plot(phi_range, S2_areas, label='actual 2')
	ax2.plot(phi_range, approximated_areas1, label='approx 1')
	ax2.plot(phi_range, approximated_areas2, label='approx 2')
	# ax2.plot(phi_range, t1s, label='t1')
	ax1.set_xlabel('phi_1')
	# ax2.set_ylabel('area')

	ax2.set_xlabel('phi')
	ax1.legend()
	ax2.legend()
	plt.show()
	return


def t1_searcher(S, I, t_range, S_peak, beta, t_vac):
	for i in range(len(t_range)):
		St = S[i]
		It = I[i]
		t = t_range[i]
		if St < S_peak:
			return t_vac
		dS = - beta * St * It
		if St + dS * (t_vac - t) <= S_peak:
			return t
	return t


def plot_area(S, t_range, S_peak, i, t_vac, S_area, approximated_area):
	phi = S[0]
	t1 = t_range[i]
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(t_range, S, label='S')
	ax1.axhline(S_peak, linestyle=':', label='S_peak')
	ax1.fill_between([0, t1], [phi, phi], alpha=0.5, color='grey')
	ax1.fill_between([t1, t_vac], [S[i], S_peak], alpha=0.5, color='grey')
	fig.suptitle(f'phi={round(phi, 3)}\nratio={round(S_area / approximated_area, 4)}')
	ax1.legend()
	plt.show()
	return


def tmp():
	phi_range = np.arange(0.001, 1, 0.001)
	beta = 1
	gamma = 1 / 14
	epsilon = 0.5
	T = 100
	areas = []
	for phi in phi_range:
		area = phi * T - 1 / (2 * beta * epsilon) * (1 - gamma / (1 - epsilon) / (beta * phi)) * (
					1 + epsilon * gamma / beta / phi)
		areas.append(area)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi_range, areas)
	plt.show()
	return


def main():
	# POA_plot()
	# t_peak_area_comparison()
	# area_lower_bound_against_beta()
	# area_upper_bound_against_beta()

	# second_derivative()

	area_comparison()

	# tmp()
	return


if __name__ == '__main__':
	main()
