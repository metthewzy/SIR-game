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


def t_peak_area_comparison(S0, plot=False):
	print(S0)
	# S0 = 0.16
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	beta = 1
	I0 = I0_global * S0
	S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)
	if plot:
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		ax1.plot(t_range, S, label='S')
		ax1.plot(t_range, I, label='I')

	S_peak = gamma / beta
	k_max = round(np.floor(S0 / S_peak))
	# print('k max=', k_max)
	S_ts = [S0]
	ts = [0]
	t2s = [0]
	r1s = []
	r2s = []
	t3s = [0]

	for k in range(k_max, 0, -1):
		if k * S_peak == S0:
			continue
		S_ts.append(k * S_peak)
		t_k = t_S(beta, gamma, S0, I0, S_ts[-1])
		t2_k, r1, r2 = t2_S(beta, gamma, S0, I0, S_ts[-1])
		r1s.append(r1)
		r2s.append(r2)
		ts.append(t_k)
		t2s.append(t2_k)
		t3_k, r1, r2 = t3_S(beta, gamma, S0, I0, S_ts[-1])
		t3s.append(t3_k)
	# print(t_k, t2_k)
	# print('t_kpeak=', t_k)

	approximated_area = 0
	print(len(S_ts), len(t2s), len(t3s))
	for i in range(len(S_ts)):
		if i == 0:
			continue
		approximated_area += S_ts[i - 1] * (ts[i] - ts[i - 1])
		if plot:
			ax1.hlines(S_ts[i - 1], ts[i - 1], ts[i], color='green')
			ax1.vlines(ts[i], S_ts[i - 1], S_ts[i], color='green')
			ax1.vlines(t2s[i], S_ts[i - 1], S_ts[i], color='red')
			ax1.vlines(t3s[i], S_ts[i - 1], S_ts[i], color='orange')

	approximated_area += S_ts[-1] * (t_vac - ts[-1]) / 2 if ts[-1] < t_vac else S_ts[-1] * (t_vac - ts[-1])

	actual_area = np.mean(S) * t_vac
	# print(approximated_area, actual_area)
	print(ts)
	print(t2s)
	if plot:
		ax1.hlines(S_ts[-1], ts[-1], t_vac, color='green')
		ax1.vlines(t_vac, 0, S_ts[-1], color='green')
		ax1.set_title(f'S0={round(S0, 3)}\napprox ratio={round(approximated_area / actual_area, 3)}')
		ax2.plot(range(len(r1s)), r1s, label='original')
		ax2.plot(range(len(r2s)), r2s, label='X\'/Y\'')
		ax2.legend()
		plt.show()
	return approximated_area, actual_area


def t_S(beta, gamma, S0, I0, St):
	a = beta - gamma / 2 / S0
	b = - beta * (I0 + S0)
	c = gamma * S0 / 2
	dt = []
	for S in np.arange(S0, St + (St - S0) / 10000, (St - S0) / 10000):
		dt.append(1 / (a * S ** 2 + b * S + c))
	# dt.append(1/(-beta * S * (I0+S0-S+gamma/beta*(S/S0-1)/(S/S0))))
	ret = np.mean(dt) * (St - S0)
	# print(np.mean(dt))
	return ret


def t2_S(beta, gamma, S0, I0, St):
	def F0(x):
		ret = (np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x)
		       - np.log(np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * x)) \
		      / np.sqrt(b ** 2 - 4 * a * c)

		# ret = (np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x)
		# 	   - np.log(np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * x)) \
		# 	  / (beta * S0 - gamma)
		# ret = np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x) / np.sqrt(b ** 2 - 4 * a * c)
		return ret

	def F(x):
		# ret = (np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x)
		# 		- np.log(np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * x)) \
		# 	   / np.sqrt(b ** 2 - 4 * a * c)
		ret = np.log((beta * S0 - beta * x - gamma / 2) / (beta * x - gamma / 2)) \
		      / (beta * S0 - gamma)
		return ret

	a = beta - gamma / 2 / S0
	b = - beta * (I0 + S0)
	c = gamma * S0 / 2
	ret = F(St) - F0(S0)
	# print(np.sqrt(b**2 - 4 * a * c), beta * S0 - gamma)
	# print('X=', np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * St)
	# print('Y=', np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * St)
	r1 = (np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * S0) / (np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * S0)
	r2 = (abs(b) - a * S0 - abs(a * c / b)) / (a * S0 - abs(a * c / b))
	# print('X/Y=', r1)
	# print('X\'=', abs(b) - a * St - abs(a * c / b))
	# print('Y\'=', a * St - abs(a * c / b))
	# print('X\'/Y\'', r2)
	return ret, r1, r2


def t3_S(beta, gamma, S0, I0, St):
	def F0(x):
		ret = (np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x)
		       - np.log(np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * x)) \
		      / np.sqrt(b ** 2 - 4 * a * c)

		# ret = (np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x)
		# 	   - np.log(np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * x)) \
		# 	  / (beta * S0 - gamma)
		# ret = np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x) / np.sqrt(b ** 2 - 4 * a * c)
		return ret

	def F(x):
		# ret = (np.log(np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * x)
		# 		- np.log(np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * x)) \
		# 	   / np.sqrt(b ** 2 - 4 * a * c)
		ret = np.log((beta * S0 - beta * x / 2 - gamma / 4) / (beta * x - gamma / 2)) \
		      / (beta * S0 - gamma)
		return ret

	a = beta - gamma / 2 / S0
	b = - beta * (I0 + S0)
	c = gamma * S0 / 2
	ret = F(St) - F0(S0)
	# print(np.sqrt(b**2 - 4 * a * c), beta * S0 - gamma)
	# print('X=', np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * St)
	# print('Y=', np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * St)
	r1 = (np.sqrt(b ** 2 - 4 * a * c) - b - 2 * a * S0) / (np.sqrt(b ** 2 - 4 * a * c) + b + 2 * a * S0)
	r2 = (abs(b) - a * S0 - abs(a * c / b)) / (a * S0 - abs(a * c / b))
	# print('X/Y=', r1)
	# print('X\'=', abs(b) - a * St - abs(a * c / b))
	# print('Y\'=', a * St - abs(a * c / b))
	# print('X\'/Y\'', r2)
	return ret, r1, r2


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
	phi_step = 0.001
	phi_range = np.arange(phi_step, 1, phi_step)
	S1_areas = []
	S2_areas = []
	approximated_areas1 = []
	approximated_areas2 = []
	t1s = []
	t2s = []
	payment_ratio = 5
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	beta1 = 1
	beta2 = 0.5
	for phi in phi_range:
		S0 = phi
		I0 = I0_global * S0
		S1, I1, t_range = simulate(beta1, gamma, S0, I0_global, t_vac, False)
		S1_peak = gamma / beta1
		S1_area = np.mean(S1) * t_vac * payment_ratio
		S1_areas.append(S1_area)
		t1 = t1_searcher(S1, I1, t_range, S1_peak, beta1, t_vac)
		t1s.append(t1)
		i1 = list(t_range).index(t1)
		approximated_area1 = (S0 * t1 + (S1[i1] + S1_peak) * (t_vac - t1) / 2) * payment_ratio
		approximated_areas1.append(approximated_area1)

		S0 = 1 - phi
		I0 = I0_global * S0
		S2, I2, t_range = simulate(beta2, gamma, S0, I0_global, t_vac, False)
		S2_peak = gamma / beta2
		S2_area = np.mean(S2) * t_vac
		S2_areas.append(S2_area)
		t2 = t1_searcher(S2, I2, t_range, S2_peak, beta2, t_vac)
		t2s.append(t2)
		i2 = list(t_range).index(t2)
		approximated_area2 = S0 * t2 + (S2[i2] + S2_peak) * (t_vac - t2) / 2
		approximated_areas2.append(approximated_area2)

		if 0.171 < phi < 0.173:
			plot_area(beta1, gamma, S1, I1, t_range, S1_peak, i1, t_vac, payment_ratio, S1_area, approximated_area1)
			plot_area(beta2, gamma, S2, I2, t_range, S2_peak, i2, t_vac, 1, S2_area, approximated_area2)

	# plot_area(beta1, gamma, S1, I1, t_range, S1_peak, i1, t_vac, payment_ratio, S1_area, approximated_area1)
	# plot_area(beta2, gamma, S2, I2, t_range, S2_peak, i2, t_vac, 1, S2_area, approximated_area2)

	S_areas = [S1_areas[i] + S2_areas[i] for i in range(len(S1_areas))]
	max_utility = max(S_areas)
	max_idx = S_areas.index(max_utility)
	dU1 = -(S1_areas[max_idx - 1] - S1_areas[max_idx]) / phi_step
	dU2 = -(S2_areas[max_idx - 1] - S2_areas[max_idx]) / phi_step
	print('max when phi=', phi_range[max_idx])
	print(f'At max dU1/d phi={round(dU1, 5)}, dU2/d phi={round(dU2, 5)}')
	approximated_areas = [approximated_areas1[i] + approximated_areas2[i] for i in range(len(approximated_areas1))]
	fig = plt.figure()
	# ax1 = fig.add_subplot()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(phi_range, S_areas, label='actual utility')
	ax2.plot(phi_range, approximated_areas, label='approx utility')
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


def plot_area(beta, gamma, S, I, t_range, S_peak, i, t_vac, payment_ratio, S_area, approximated_area):
	phi = S[0]
	# t1 = t_range[i]
	t1 = t_vac - (phi - S_peak) / (beta * S_peak * (I[0] + phi - S_peak + S_peak * np.log(S_peak / phi)))
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(t_range, S, label='S')
	ax1.plot(t_range, I, label='I')
	ax1.axhline(S_peak, color='b', linestyle=':', label='S_peak')
	ax1.axhline(I[0] + phi - S_peak + S_peak * np.log(S_peak / phi), color='r', linestyle=':', label='I_peak')
	ax1.fill_between([0, t1, t_vac], [phi, phi, S_peak], alpha=0.5, color='grey')
	# ax1.fill_between([t1, t_vac], [S[i], S_peak], alpha=0.5, color='grey')
	fig.suptitle(
		f'phi={round(phi, 3)}\nratio={round(payment_ratio * (t_vac * phi - (t_vac - t1) * (phi - S_peak) / 2) / S_area, 4)}')

	if S[0] >= gamma / beta:
		t_peak_est = (np.log(S[0] / I[0]) - np.log(S_peak) - np.log(
			S_peak / (S[0] + I[0] + S_peak * np.log(S_peak / S[0]) - S_peak))) / beta / (
				             S[0] + I[0] + S_peak * np.log(S_peak / S[0]))
		ax1.axvline(t_peak_est, linestyle=':')
	ax1.legend()
	plt.show()
	return


def tmp():
	beta1 = 1
	beta2 = 1
	gamma = 1 / 14
	PR = 2
	NE = PR / (np.exp(beta1 / gamma) - beta1 / gamma)
	OPT = PR * gamma / beta1 + gamma / beta2
	print(OPT / NE)
	return


def group_utility(beta, gamma, phi, I0_global, t_vac, payment_ratio, num_steps=10000):
	S0 = phi
	S1, I1, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False, num_steps)
	S1_area = np.mean(S1) * t_vac * payment_ratio
	return S1_area


def group_peak_searcher():
	phi_step = 0.001
	phi_range = np.arange(phi_step, 1, phi_step)
	S1_areas = []
	S2_areas = []
	payment_ratio = 5
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	beta1 = 2
	beta2 = 0.5
	mids = []
	lefts = []
	rights = []
	U_mids = []
	U_lefts = []
	U_rights = []
	for phi in phi_range:
		S0 = phi
		I0 = I0_global * S0
		S1, I1, t_range = simulate(beta1, gamma, S0, I0_global, t_vac, False)
		S1_area = np.mean(S1) * t_vac * payment_ratio
		S1_areas.append(S1_area)
	max_utility1 = max(S1_areas)
	max_idx1 = S1_areas.index(max_utility1)
	left, mid, right = phi_range[max_idx1 - 1], phi_range[max_idx1], phi_range[max_idx1 + 1]
	for _ in range(20):
		print(mid)
		U_mid = group_utility(beta1, gamma, mid, I0_global, t_vac, payment_ratio)
		mids.append(mid)
		lefts.append(left)
		rights.append(right)

		left2, right2 = left, right
		U_left = group_utility(beta1, gamma, left2, I0_global, t_vac, payment_ratio)
		U_right = group_utility(beta1, gamma, right2, I0_global, t_vac, payment_ratio)

		U_mids.append(U_mid)
		U_lefts.append(U_left)
		U_rights.append(U_right)
		while (U_mid - U_left) * (U_mid - U_right) >= 0:
			left2 = mid + (left2 - mid) / 2
			right2 = mid + (right2 - mid) / 2
			U_left = group_utility(beta1, gamma, left2, I0_global, t_vac, payment_ratio)
			U_right = group_utility(beta1, gamma, right2, I0_global, t_vac, payment_ratio)

		if U_left < U_mid:
			left = mid
		else:
			right = mid

		mid = (left + right) / 2

	U_mid = group_utility(beta1, gamma, mid, I0_global, t_vac, payment_ratio)
	U_left = group_utility(beta1, gamma, left, I0_global, t_vac, payment_ratio)
	U_right = group_utility(beta1, gamma, right, I0_global, t_vac, payment_ratio)

	mids.append(mid)
	lefts.append(left)
	rights.append(right)

	U_mids.append(U_mid)
	U_lefts.append(U_left)
	U_rights.append(U_right)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(phi_range, S1_areas)
	ax1.axvline(mid, linestyle=':')
	ax1.axhline(U_mid, linestyle=':')
	ax2.plot(mids, U_mids, label='mid')
	ax2.plot(lefts, U_lefts, label='left')
	ax2.plot(rights, U_rights, label='right')
	ax2.legend()
	plt.show()

	S0 = mid
	print(S0)
	S1, I1, t_range = simulate(beta1, gamma, S0, I0_global, t_vac, False)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(t_range, S1)
	ax1.axhline(gamma / beta1, linestyle=':')
	plt.show()
	return


def approximated_area_comparison():
	approx_areas = []
	act_areas = []
	# S0s = []
	S0_range = np.arange(0.01, 1, 0.01)
	for S0 in S0_range:
		approx_area, act_area = t_peak_area_comparison(S0)
		approx_areas.append(approx_area)
		act_areas.append(act_area)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(S0_range, approx_areas, label='approx')
	ax1.plot(S0_range, act_areas, label='act')
	ax2.plot(S0_range, [act_areas[i] / approx_areas[i] for i in range(len(act_areas))])
	ax1.legend()
	plt.show()
	return


def t0_area_comparison(S0):
	I0_global = 0.0001
	t_vac = 100
	gamma = 1 / 14
	beta = 1
	I0 = I0_global * S0
	S_peak = gamma / beta
	print('S_peak=', S_peak)
	S, I, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False)

	for t_peak_idx in range(len(t_range)):
		if S[t_peak_idx] <= S_peak:
			break
	actual_area = np.mean(S[:t_peak_idx]) * t_range[t_peak_idx]
	print(f'actual:\n{actual_area}')

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	# ax2 = fig.add_subplot(122)
	ax1.plot(t_range, S, label='S')
	ax1.plot(t_range, I, label='I')
	k_max = round(np.ceil(S0 / S_peak)) - 1

	S_ts = [S0]
	# print(S_ts[-1], S0)
	ts = [0]
	t2s = [0]

	for k in range(k_max, 0, -1):
		S_ts.append(k * S_peak)
		t_k = t_S(beta, gamma, S0, I0, S_ts[-1])
		t2_k, r1, r2 = t3_S(beta, gamma, S0, I0, S_ts[-1])
		ts.append(t_k)
		t2s.append(t2_k)

	approx_area = 0
	approx_area2 = 0
	approx_yz = 0
	for i in range(2, len(ts)):
		approx_area += (ts[i] - ts[i - 1]) * S_ts[i - 1]
		approx_area2 += (t2s[i] - t2s[i - 1]) * S_ts[i - 1]

	for i in range(1, k_max - 1):
		approx_yz += gamma * (i + 1) / (S0 * beta - gamma) / beta * (np.log(3) + gamma / S0 / beta)
	print(approx_area)
	print(approx_area2)
	print(approx_yz)

	approx_area += t_S(beta, gamma, S0, I0, S0 - S_peak) * S0
	approx_area2 += t3_S(beta, gamma, S0, I0, S0 - S_peak)[0] * S0
	print(f'approx1(green):\n{approx_area}, {round(approx_area / actual_area * 100, 2)}%')
	print(f'approx2(red):\n{approx_area2}, {round(approx_area2 / actual_area * 100, 2)}%')

	for t_idx in range(len(S_ts)):
		if t_idx == 0:
			continue
		ax1.hlines(S_ts[t_idx - 1], ts[t_idx - 1], ts[t_idx], color='green')
		ax1.vlines(ts[t_idx], S_ts[t_idx - 1], S_ts[t_idx], color='green')
		ax1.vlines(t2s[t_idx], S_ts[t_idx - 1], S_ts[t_idx], color='red')

	print(ts)
	print(t_S(beta, gamma, S0, I0, S0 - S_peak))
	print(t2s)
	print(t3_S(beta, gamma, S0, I0, S0 - S_peak)[0])
	plt.show()
	return


def S_infinity_comparison(beta, gamma, t_vac):
	phi_step = 0.001
	phi_range = np.arange(phi_step, 1, phi_step)
	I0_global = 0.0001
	# t_vac = 10000
	# t_vacs = [100, 200, 500, 1000, 10000]
	# gamma = 1 / 14
	# beta = 0.1
	S_infinity = []
	S_lower = []
	S_upper = []
	for phi in phi_range:
		S0 = phi
		I0 = I0_global * S0
		S1, I1, t_range = simulate(beta, gamma, S0, I0_global, t_vac, False, num_steps=100000)
		S_infinity.append(S1[-1])
		S_lower.append(phi / (np.exp(phi * beta / gamma) - phi * beta / gamma))
		S_upper.append(
			phi * (phi * beta / gamma - np.log(phi * beta / gamma)) /
			(phi * beta / gamma +
			 np.exp(phi * beta / gamma) * (phi * beta / gamma - 1 - np.log(phi * beta / gamma))))

	fig = plt.figure(figsize=(6, 12))
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.plot(phi_range, S_infinity, label=r'$S_\infty$')
	ax1.plot(phi_range, S_lower, label=f'Lower')
	ax1.plot(phi_range, S_upper, label=f'Upper')
	ax1.axvline(gamma / beta, color='grey', linestyle=':', label=r'$\gamma/\beta$')
	ax2.plot(phi_range, [S_infinity[i] / phi_range[i] for i in range(len(phi_range))], label=r'$S_\infty$')
	ax2.plot(phi_range, [S_lower[i] / phi_range[i] for i in range(len(phi_range))], label=f'Lower')
	ax2.plot(phi_range, [S_upper[i] / phi_range[i] for i in range(len(phi_range))], label=f'Upper')
	ax2.axvline(gamma / beta, color='grey', linestyle=':', label=r'$\gamma/\beta$')
	y1, y2 = ax1.get_ylim()
	ax1.set_ylim(0, y2)
	y1, y2 = ax2.get_ylim()
	ax2.set_ylim(0, y2)
	ax1.set_xlim(0, 1)
	ax2.set_xlim(0, 1)
	fig.suptitle(rf'$\beta={round(beta, 3)},\gamma={round(gamma, 3)},T={t_vac}$')
	ax1.set_title('Group utility')
	ax1.set_xlabel(r'$\phi$')
	ax1.legend()
	ax2.set_title('Individual utility')
	ax2.set_xlabel(r'$\phi$')
	ax2.legend()
	fig.savefig(f'fig/b={round(beta, 3)}g={round(gamma, 3)}T={t_vac}.png')
	# plt.show()
	return


def main():
	# POA_plot()
	# approximated_area_comparison()
	# t_peak_area_comparison(0.45, True)
	# t0_area_comparison(0.07142857142857142 * 5 + 0.01)

	# area_lower_bound_against_beta()
	# area_upper_bound_against_beta()

	# second_derivative()

	# area_comparison()
	# group_peak_searcher()

	# tmp()
	S_infinity_comparison(1 / 7, 1 / 14, 10000)
	S_infinity_comparison(0.1, 1 / 14, 10000)
	S_infinity_comparison(0.25, 1 / 14, 10000)
	S_infinity_comparison(0.5, 1 / 14, 10000)
	S_infinity_comparison(1, 1 / 14, 10000)
	S_infinity_comparison(2, 1 / 14, 10000)
	S_infinity_comparison(5, 1 / 14, 10000)
	return


if __name__ == '__main__':
	main()
