import numpy as np
import matplotlib.pyplot as plt


def zero_searcher(f, left, right):
	mid = (left + right) / 2
	for _ in range(20):
		if f(mid) < 0:
			left = mid
		else:
			right = mid
		mid = (left + right) / 2
	return mid


def g_plotter(beta=2 / 14, gamma=1 / 14, phi=0.5):
	S_step = 0.001
	S_range = np.arange(0, 1 + S_step, S_step)

	def g(S):
		ret = S - np.exp(-phi * beta / gamma * (1 - S))
		return ret

	gs = []
	for S in S_range:
		gs.append(g(S))

	S_peak = gamma / (phi * beta) * np.log(gamma / (phi * beta)) + 1
	g_peak = g(S_peak)
	g_0 = g(0)
	S_inf = zero_searcher(g, 0, S_peak)
	upper_bound = (phi * beta + gamma * np.log(gamma / (phi * beta))) / \
				  (phi * beta + gamma * np.exp(phi * beta / gamma) * ((phi * beta) / gamma - 1 + np.log(
					  gamma / (phi * beta))))
	lower_bound = gamma / (np.exp(phi * beta / gamma) * gamma - phi * beta)

	plt.rcParams.update({'font.size': 18})
	fig = plt.figure(figsize=(12, 8.5))
	ax1 = fig.add_subplot()
	# ax1.set_aspect('equal')
	# ax1.grid(True, which='both')
	ax1.axhline(y=0, color='k')
	ax1.axvline(x=0, color='k')
	ax1.plot(S_range, gs, label=r'$g(\overline{S})$')
	ax1.plot([0, S_peak], [g_0, g_peak], linestyle='--', c='grey')
	ax1.plot([0, lower_bound], [g_0, 0], linestyle='--', c='grey')
	ax1.plot(S_peak, g_peak, marker='o', c='k', markersize=5)
	ax1.plot(0, g_0, marker='o', c='k', markersize=5)
	ax1.plot(S_inf, 0, marker='o', c='k', markersize=5)
	ax1.plot(upper_bound, 0, marker='o', c='k', markersize=5)
	ax1.plot(lower_bound, 0, marker='o', c='k', markersize=5)
	ax1.annotate(r'$g_p$', (S_peak, g_peak), textcoords="offset points", xytext=(5, -15), ha='left')
	ax1.annotate(r'$g(0)$', (0, g_0), textcoords="offset points", xytext=(15, 0), va='center')
	ax1.annotate(r'$\overline{S}(\infty)$', (S_inf, 0), textcoords="offset points", xytext=(0, -25), ha='left')
	ax1.annotate(r'$UB$', (upper_bound, 0), textcoords="offset points", xytext=(0, -25), ha='left')
	ax1.annotate(r'$LB$', (lower_bound, 0), textcoords="offset points", xytext=(-5, 10), ha='right')

	ax1.set_xlabel(r'$\overline{S}$')
	# ax1.set_title(r'$g(\overline{S})$')
	ax1.legend()
	# fig.savefig('g(S).png', bbox_inches='tight')
	plt.show()
	return


def main():
	g_plotter(beta=3 / 14, gamma=1 / 14, phi=0.5)
	return


if __name__ == '__main__':
	main()
