import numpy as np
import matplotlib.pyplot as plt

binary_iterations = 100
OPT_iterations = 30
NE_iterations = 50


def final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon, plot):
	"""
	binary search the final sizes of 2 groups interacting
	"""
	S2s = []
	f2s = []
	phi2 = 1 - phi1
	S2_l = 0
	S2_r = phi2 * (1 - epsilon)
	for _ in range(binary_iterations):
		S2_m = (S2_l + S2_r) / 2
		S1 = S1_final_searcher(S2_m, beta, beta_ratio, gamma, epsilon, phi1)
		f = f2([S1, S2_m], phi1, beta, beta_ratio, gamma, epsilon)
		S2s.append(S2_m)
		f2s.append(f)
		if f > 0:
			S2_r = S2_m
		else:
			S2_l = S2_m
	S2 = S2_m
	S1 = S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1)
	f = f2([S1, S2_m], phi1, beta, beta_ratio, gamma, epsilon)
	S2s.append(S2_m)
	f2s.append(f)

	if plot:
		fig = plt.figure()
		ax1 = fig.add_subplot()
		[ax1.plot([i, i + 1], [S2s[i], S2s[i + 1]], c='red' if S2s[i + 1] > S2s[i] else 'green') for i in
		 range(len(S2s) - 1)]
		# ax1.axhline(0, c='grey', linestyle=':')
		# ax1.scatter(S2s[-1], f2s[-1], c='red')
		plt.show()
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


def final_size_test(phi1, beta, beta_ratio, gamma, epsilon, plot=False):
	S1, S2 = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon, plot)
	return


def final_size_plotter(phi1, beta, beta_ratio, gamma, epsilon, plot):
	phi2 = 1 - phi1
	S2_step = 0.001
	S2_range = np.arange(0, phi2 + S2_step, S2_step)
	S1s = []
	for S2 in S2_range:
		S1s.append(S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1))

	f2s = []
	for S1, S2 in zip(S1s, S2_range):

		f2s.append(f2([S1, S2], phi1, beta, beta_ratio, gamma,epsilon))
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(S2_range, f2s)
	ax1.axhline(0, c='grey')
	plt.show()
	return


def main():
	final_size_plotter(phi1=0.5, beta=1.5 / 14, beta_ratio=0.9, gamma=1 / 14, epsilon=0.0001, plot=True)
	# final_size_test(phi1=0.5, beta=1.5 / 14, beta_ratio=0.9, gamma=1 / 14, epsilon=0.0001, plot=True)
	return


if __name__ == '__main__':
	main()
