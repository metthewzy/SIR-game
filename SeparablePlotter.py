import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ConvexProgram import one_group_binary_search, separable_two_group_POA_comparison
import concurrent.futures

phi_step = 0.001


def dS_by_dPhi(beta, gamma, epsilon, S, phi):
	ret = S * (1 / phi - beta / gamma) / (1 - beta / gamma * S)
	return ret


def one_group_derivative(beta, gamma, epsilon):
	phi_range = np.arange(phi_step, 1, phi_step)
	Ss = []
	dS = [1]
	dS2 = []
	for phi in phi_range:
		Ss.append(one_group_binary_search(beta, gamma, epsilon, phi))
		dS2.append(dS_by_dPhi(beta, gamma, epsilon, Ss[-1], phi))
	dS.extend([(Ss[i] - Ss[i - 1]) / phi_step for i in range(1, len(Ss))])
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi_range, Ss, label='S')
	ax1.plot(phi_range, dS, label='num')
	ax1.plot(phi_range, dS2, label='formula')
	ax1.axhline(0, color='grey', linestyle=':')
	ax1.axvline(gamma / beta, color='grey', linestyle=':')
	ax1.legend()
	plt.show()
	return


def two_group_social(b1, b2, gamma, epsilon, p2):
	phi1_range = np.arange(phi_step, 1, phi_step)
	phi1_range = np.arange(0, 1 + phi_step, phi_step)
	U1s = []
	U2s = []
	social = []
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		U1s.append(one_group_binary_search(b1, gamma, epsilon, phi1))
		U2s.append(p2 * one_group_binary_search(b2, gamma, epsilon, phi2))
		social.append(U1s[-1] + U2s[-1])
	phi_peak = phi1_range[social.index(max(social))]
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1_range, U1s, label='group1')
	ax1.plot(phi1_range, U2s, label='group2')
	ax1.plot(phi1_range, social, label='social')
	ax1.axvline(phi_peak, linestyle=':', color='grey')

	ax1.axvline(gamma / b1, linestyle=':', color='blue')
	ax1.legend()
	plt.show()
	return


def two_group_POA_plotter(b1, b2, gamma, epsilon):
	"""
	Plot the worst POA
	"""
	# phi1_range = np.arange(phi_step, 1, phi_step)
	phi1_range = np.arange(0, 1 + phi_step, phi_step)
	S1s = [0]
	S1_by_phis = [1 - epsilon]
	S2 = one_group_binary_search(b2, gamma, epsilon, 1)
	S2s = [S2]
	S2_by_phis = [S2]
	for phi1 in phi1_range[1:-1]:
		phi2 = 1 - phi1
		S1 = one_group_binary_search(b1, gamma, epsilon, phi1)
		S2 = one_group_binary_search(b2, gamma, epsilon, phi2)
		S1s.append(S1)
		S2s.append(S2)
		S1_by_phis.append(S1 / phi1)
		S2_by_phis.append(S2 / phi2)

	S1 = one_group_binary_search(b1, gamma, epsilon, 1)
	S1s.append(S1)
	S1_by_phis.append(S1)
	S2s.append(0)
	S2_by_phis.append(1 - epsilon)

	p2 = S1_by_phis[-1] / S2_by_phis[-1]
	print(f"p2={round(p2, 5)}")
	U1s = S1_by_phis.copy()
	UG1s = S1s.copy()
	U2s = [p2 * S2_by_phi for S2_by_phi in S2_by_phis]
	UG2s = [p2 * S2 for S2 in S2s]
	social = [UG1 + UG2 for UG1, UG2 in zip(UG2s, UG1s)]

	fig = plt.figure(figsize=(9, 4))
	ax1 = fig.add_subplot(121)

	ax1.plot(phi1_range, UG1s, label='Group 1')
	ax1.plot(phi1_range, UG2s, label='Group 2')
	ax1.plot(phi1_range, social, label='Social')
	ax1.axhline(max(social), color='red', linestyle='--', label='OPT')
	ax1.axhline(social[-1], color='grey', linestyle='--', label='NE')
	ax1.axvline(phi1_range[social.index(max(social))], color='red', linestyle=':')
	ax1.set_xlabel(r"$\phi_1$")
	ax1.set_title('Group utility')
	ax1.legend()

	ax2 = fig.add_subplot(122)
	ax2.plot(phi1_range, U1s, label='Group 1')
	ax2.plot(phi1_range, U2s, label='Group 2')
	ax2.axvline(phi1_range[social.index(max(social))], color='red', linestyle=':')
	ax2.set_xlabel(r"$\phi_1$")
	ax2.set_title('Individual utility')
	ax2.legend()
	ax1.set_xlim(0, 1)
	ax2.set_xlim(0, 1)

	u, l = ax1.get_ylim()
	ax1.set_ylim(0, l)
	ax2.set_ylim(0, 1.05)

	plt.tight_layout()
	# fig.savefig('SeparablePOA.png')
	R0, POA, POA_bound = worst_POA_comparison(b1 / gamma, b2, gamma, epsilon)
	print(f'POA=\n{round(POA, 5)}')
	print(f'POA bound=\n{round(POA_bound, 5)}')
	plt.show()
	return


def two_group_POA_bound(b2, gamma, epsilon):
	"""
	Compare the POA with the bound
	"""
	R0_range = np.arange(1, 10.01, 0.25)
	POAs = []
	POA_bounds = []
	R0s = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
		results = [executor.submit(worst_POA_comparison, R0, b2, gamma, epsilon) for R0 in R0_range]
		for f in concurrent.futures.as_completed(results):
			R0, POA, POA_bound = f.result()
			R0s.append(R0)
			POAs.append(POA)
			POA_bounds.append(POA_bound)
	R0s, POAs, POA_bounds = zip(*sorted(zip(R0s, POAs, POA_bounds)))
	# for R0 in R0_range:
	# 	b1 = R0 * gamma
	# 	p2 = one_group_binary_search(b1, gamma, epsilon, 1)
	# 	POA, POA_bound = separable_two_group_POA_comparison(b1, b2, gamma, epsilon, p2, False)
	# 	POAs.append(POA)
	# 	POA_bounds.append(POA_bound)

	fig = plt.figure(figsize=(6, 4))
	ax1 = fig.add_subplot()
	ax1.plot(R0s, [POA_bound / POA for POA_bound, POA in zip(POA_bounds, POAs)], marker='o', label='bound/actual')
	ax1.axhline(1, color='grey', linestyle=':')
	ax1.set_xlabel(r"$R_0$")
	ax1.set_title("POA bound / POA")
	# ax1.plot(R0_range, POAs, label='POA')
	# ax1.plot(R0_range, POA_bounds, label='bound')
	# ax1.legend()
	plt.tight_layout()
	# fig.savefig('SeparablePOA_bound_ratio.png')
	plt.show()
	return


def worst_POA_comparison(R0, b2, gamma, epsilon):
	b1 = R0 * gamma
	# b2 = b1 * b_ratio
	p2 = (1 - epsilon) / one_group_binary_search(b1, gamma, epsilon, 1)
	POA, POA_bound = separable_two_group_POA_comparison(b1, b2, gamma, epsilon, p2, False)
	return R0, POA, POA_bound


def three_group_social(b1, b2, b3, gamma, epsilon, p2, p3):
	U1s = []
	U2s = []
	U3s = []
	social = []
	phi1s = []
	phi2s = []
	phi3s = []
	for i in range(101):
		for j in range(101):
			k = 100 - i - j
			if k < 0:
				continue
			phi1 = i / 100
			phi2 = j / 100
			phi3 = k / 100
			phi1s.append(phi1)
			phi2s.append(phi2)
			phi3s.append(phi3)
			if i == 0:
				U1s.append(0)
			else:
				U1s.append(one_group_binary_search(b1, gamma, epsilon, phi1))
			if j == 0:
				U2s.append(0)
			else:
				U2s.append(p2 * one_group_binary_search(b2, gamma, epsilon, phi2))
			if k == 0:
				U3s.append(0)
			else:
				U3s.append(p3 * one_group_binary_search(b3, gamma, epsilon, phi3))
			social.append(U1s[-1] + U2s[-1] + U3s[-1])
	fig = plt.figure()
	ax1 = fig.add_subplot(projection='3d')
	ax1.plot_trisurf(phi1s, phi2s, social, cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	plt.show()
	return


def f(S, phi, beta, gamma, epsilon):
	ret = S - (1 - epsilon) * phi * np.exp(beta / gamma * (S - phi))
	return ret


def f_plotter(beta, gamma, epsilon):
	phi_range = np.linspace(0, 1, 100)
	S_range = np.linspace(0, 0, 100)
	f_values = []
	for phi in phi_range:
		f_values.append([])
		for S in S_range:
			f_values[-1].append(f(S, phi, beta, gamma, epsilon))

	X, Y = np.meshgrid(phi_range, S_range)
	fig = plt.figure()
	ax1 = fig.add_subplot(projection='3d')
	ax1.plot_surface(X, Y, np.array(f_values), cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\varphi$')
	ax1.set_ylabel(r'$\bar{S}$')
	plt.show()
	return


def f_plotter_trisurf(beta, gamma, epsilon):
	phi_range = np.linspace(0, 1, 100)
	S_range = np.linspace(0, 1, 100)
	f_values = []
	f_values2 = []
	Ss = []
	phis = []
	Ss2 = []
	phis2 = []
	for phi in phi_range:
		for S in S_range:
			f_value = f(S, phi, beta, gamma, epsilon)
			# if S <= phi:
			if f_value > 0:
				f_values.append(f_value)
				Ss.append(S)
				phis.append(phi)
			else:
				f_values2.append(f_value)
				Ss2.append(S)
				phis2.append(phi)

	f_values = np.array(f_values)
	fig = plt.figure()
	ax1 = fig.add_subplot()
	# ax1 = fig.add_subplot(projection='3d')

	# ax1.plot_trisurf(phis, Ss, f_values, cmap=cm.coolwarm)
	# ax1.scatter(phis, Ss, f_values, color='blue')
	# ax1.scatter(phis2, Ss2, f_values2, color='red')

	ax1.scatter(phis2, Ss2, color='red')
	ax1.set_xlabel(r'$\phi$')
	ax1.set_ylabel(r'$S$')
	plt.show()
	return


def SIR_plotter():
	gamma = 1 / 10
	epsilon = 0.0001
	R0s = [2, 5, 5]
	phis = [1, 1, 0.1]
	T = 200
	dt = 0.5
	fig = plt.figure(figsize=(6, 8))
	for i in range(len(R0s)):
		R0 = R0s[i]
		phi = phis[i]
		beta = R0 * gamma
		S = [phi * (1 - epsilon)]
		I = [phi * epsilon]
		R = [0]
		t_range = np.arange(0, T + dt, dt)
		for t in t_range[1:]:
			dS = -beta * S[-1] * I[-1] * dt
			dI = -dS - gamma * I[-1] * dt
			dR = gamma * I[-1] * dt
			S.append(S[-1] + dS)
			I.append(I[-1] + dI)
			R.append(R[-1] + dR)

		ax1 = fig.add_subplot(3, 1, i + 1)
		ax1.plot(t_range, S, label='S')
		ax1.plot(t_range, I, label='I')
		ax1.plot(t_range, R, label='R')
		ax1.set_xlabel('Time')
		ax1.set_ylabel('Population')
		ax1.legend()
		ax1.set_title(rf'$\beta$={round(beta, 3)}, $\gamma$={round(gamma, 3)}, $\phi$={round(phi, 3)} ')
	plt.tight_layout()
	fig.savefig('SIR.png')
	plt.close(fig)
	return


def main():
	# f_plotter(beta=4 / 14, gamma=1 / 14, epsilon=0.0001)
	# f_plotter_trisurf(beta=3 / 14, gamma=1 / 14, epsilon=0.0001)

	# one_group_derivative(beta=2 / 14, gamma=1 / 14, epsilon=0.0001)
	# two_group_social(b1=5 / 14, b2=5 / 14, gamma=1 / 14, epsilon=0.0001, p2=1)
	two_group_POA_plotter(b1=2 / 14, b2=0.5 / 14, gamma=1 / 14, epsilon=0.0001)
	# two_group_POA_bound(b2=0.5 / 14, gamma=1 / 14, epsilon=0.5)
	# three_group_social(b1=8 / 14, b2=7 / 14, b3=6 / 14, gamma=1 / 14, epsilon=0.0001, p2=1, p3=1)

	# SIR_plotter()
	return


if __name__ == '__main__':
	main()
