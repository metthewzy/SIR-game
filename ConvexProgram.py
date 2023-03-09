import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform as uni
from matplotlib import cm
from scipy.optimize import minimize
import cvxpy as cp
from TwoGroup import final_size_searcher_binary

binary_iter = 40
phi_step = 0.001
phi_steps_3D = 50
phi_steps_3D_sep = 60
cdict = {
	'red':
		[1, 0.0, 0.0],
	'green':
		[0.0, 1, 0.0],
	'blue':
		[0.0, 0.0, 1]
}


def l_max(l1, l2, l3):
	"""
	return list of booleans where l1>l2 and l1>l3
	"""
	return [l1[i] >= l2[i] and l1[i] >= l3[i] for i in range(len(l1))]


def f(s, phi, betas, gamma, s_vec):
	return


def two_group():
	R0 = 2
	k = 0.8
	s1 = cp.Variable()
	s2 = cp.Variable()
	return


def h1(paras):
	"""
	constraint 1 for three-group
	"""
	S1, S2, S3, beta1, beta2, beta3, gamma, epsilon, phi1, phi2, phi3 = paras
	X = beta1 / gamma * (S1 - phi1) + beta2 / gamma * (S2 - phi2) + beta3 / gamma * (S3 - phi3)
	ret = S1 - (1 - epsilon) * phi1 * np.exp(X)
	return ret


def h2(paras):
	"""
	constraint 2 for three-group
	"""
	S2, S1, S3, beta1, beta2, beta3, gamma, epsilon, phi1, phi2, phi3 = paras
	X = beta1 / gamma * (S1 - phi1) + beta2 / gamma * (S2 - phi2) + beta3 / gamma * (S3 - phi3)
	ret = S2 - (1 - epsilon) * phi2 * np.exp(X)
	return ret


def h3(paras):
	"""
	constraint 3 for three-group
	"""
	S3, S1, S2, beta1, beta2, beta3, gamma, epsilon, phi1, phi2, phi3 = paras
	X = beta1 / gamma * (S1 - phi1) + beta2 / gamma * (S2 - phi2) + beta3 / gamma * (S3 - phi3)
	ret = S3 - (1 - epsilon) * phi3 * np.exp(X)
	return ret


def g(paras):
	"""
	constraint function of one-group
	"""
	S, beta, gamma, epsilon, phi = paras
	ret = S - (1 - epsilon) * np.exp(-phi * beta / gamma * (1 - S))
	return ret


def f1(paras):
	"""
	constraint 1 for two-group
	"""
	S1, S2, beta1, beta2, gamma, epsilon, phi1, phi2 = paras
	X = beta1 / gamma * (S1 - phi1) + beta2 / gamma * (S2 - phi2)
	ret = S1 - (1 - epsilon) * phi1 * np.exp(X)
	return ret


def f2(paras):
	"""
	constraint 2 for two-group
	"""
	S2, S1, beta1, beta2, gamma, epsilon, phi1, phi2 = paras
	X = beta1 / gamma * (S1 - phi1) + beta2 / gamma * (S2 - phi2)
	ret = S2 - (1 - epsilon) * phi2 * np.exp(X)
	return ret


def zero_searcher_backup(func, left, right, paras):
	"""
	search for zero point of increasing function
	"""
	mid = (left + right) / 2
	for _ in range(binary_iter):
		if func([mid] + paras) < 0:
			left = mid
		else:
			right = mid
		mid = (left + right) / 2
	return mid


def zero_searcher(func, left, right, paras):
	"""
	search for zero point of increasing function
	"""
	f_left = func([left] + paras)
	f_right = func([right] + paras)
	mid = (left + right) / 2
	for _ in range(binary_iter):
		f_mid = func([mid] + paras)
		if f_mid == 0:
			return mid
		if f_left * f_mid < 0:
			right = mid
			f_right = f_mid
		else:
			left = mid
			f_left = f_mid
		mid = (left + right) / 2
	return mid


def one_group_binary_search(beta, gamma, epsilon, phi):
	"""
	Solve the one-group final size using binary search
	"""
	S_peak = gamma / (phi * beta) * np.log(gamma / ((1 - epsilon) * phi * beta)) + 1
	S_inf = zero_searcher(g, 0, min(1, S_peak), [beta, gamma, epsilon, phi]) * phi
	return S_inf


def one_group_cvxpy(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, phi=0.5):
	"""
	Solve the one-group final size using convex programming
	"""
	s1 = cp.Variable()
	constraints = [s1 >= 0,
				   s1 <= 1 - epsilon,
				   s1 - (1 - epsilon) * cp.exp(phi * beta / gamma * (s1 - 1)) >= 0]
	obj = cp.Minimize(s1)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	# print("status:", prob.status)
	# print("optimal value", prob.value)
	# print("optimal var", s1.value)
	S_inf = s1.value * phi
	# S_inf = s1.value
	return S_inf


def two_group_cvxpy(betas, gamma=1 / 14, epsilon=0.0001, phi1=0.5):
	"""
	Solve the two-group final size using convex programming
	"""
	b11, b12, b21, b22 = betas

	s1 = cp.Variable()
	s2 = cp.Variable()
	phi2 = 1 - phi1
	s10 = (1 - epsilon) * phi1
	s20 = (1 - epsilon) * phi2
	constraints = [s1 >= 0,
				   s1 <= s10,
				   s2 >= 0,
				   s2 <= s20,
				   s1 - s10 * cp.exp(b11 / gamma * (s1 - s10) +
									 b12 / gamma * (s2 - s20)) >= 0,
				   s2 - s20 * cp.exp(b21 / gamma * (s1 - s10) +
									 b22 / gamma * (s2 - s20)) >= 0]
	obj = cp.Minimize(s1 + s2)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	# print("status:", prob.status)
	# print("optimal value", prob.value)
	# print("optimal var", s1.value, s2.value)
	return s1.value, s2.value


def three_group_cvxpy(betas, gamma=1 / 14, epsilon=0.0001, phi1=0.4, phi2=0.3, phi3=0.3):
	"""
	Solve the three-group final size using convex programming
	"""
	b11, b12, b13, b21, b22, b23, b31, b32, b33 = betas
	s1 = cp.Variable()
	s2 = cp.Variable()
	s3 = cp.Variable()
	s10 = (1 - epsilon) * phi1
	s20 = (1 - epsilon) * phi2
	s30 = (1 - epsilon) * phi3
	constraints = [s1 >= 0,
				   s1 <= s10,
				   s2 >= 0,
				   s2 <= s20,
				   s3 >= 0,
				   s3 <= s30,
				   s1 - s10 * cp.exp(b11 / gamma * (s1 - s10) +
									 b12 / gamma * (s2 - s20) +
									 b13 / gamma * (s3 - s30)) >= 0,
				   s2 - s20 * cp.exp(b21 / gamma * (s1 - s10) +
									 b22 / gamma * (s2 - s20) +
									 b23 / gamma * (s3 - s30)) >= 0,
				   s3 - s30 * cp.exp(b31 / gamma * (s1 - s10) +
									 b32 / gamma * (s2 - s20) +
									 b33 / gamma * (s3 - s30)) >= 0]
	obj = cp.Minimize(s1 + s2 + s3)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	# print("status:", prob.status)
	# print("optimal value", prob.value)
	# print("optimal var", s1.value, s2.value)
	return s1.value, s2.value, s3.value


def separable_two_group_POA_comparison(beta1=2 / 14, beta2=1 / 14, gamma=1 / 14, epsilon=0.0001, payment_ratio=2.0):
	"""
	compare POA of 2-group separable
	"""
	R1 = beta1 / gamma
	R2 = beta2 / gamma
	phi_range = np.arange(phi_step, 1, phi_step)
	S1s = []
	S2s = []
	for phi in phi_range:
		S1s.append(one_group_cvxpy(beta1, gamma, epsilon, phi))
		S2s.append(one_group_cvxpy(beta2, gamma, epsilon, 1 - phi))
	# POA, idx = separable_POA_searcher(beta1, beta2, gamma, epsilon, phi_range, S1s, S2s, payment_ratio)

	# payment_ratio = 17
	UG1s = [S1 * payment_ratio for S1 in S1s]
	UG2s = [S2 for S2 in S2s]
	U1s = [UG1 / phi for UG1, phi in zip(UG1s, phi_range)]
	U2s = [UG2 / (1 - phi) for UG2, phi in zip(UG2s, phi_range)]
	social = [UG1 + UG2 for UG1, UG2 in zip(UG1s, UG2s)]

	# compute OPT
	if social[0] >= social[1]:
		phi_OPT = phi_range[0]
		OPT = social[0]
	elif social[-1] >= social[-2]:
		phi_OPT = phi_range[-1]
		OPT = social[-1]
	else:
		social_max = max(social)
		idx_max = social.index(social_max)
		m = phi_range[idx_max]
		l = phi_range[idx_max - 1]
		r = phi_range[idx_max + 1]
		# l = phi_range[0]
		# r = phi_range[-1]
		# m = (l + r) / 2
		phi_OPT, OPT = separable_OPT_searcher(beta1, beta2, gamma, epsilon, payment_ratio, l, r, m)

	# compute NE
	if U2s[0] >= U1s[0]:
		NE = social[0]
		phi_NE = phi_range[0]
	elif U1s[-1] >= U2s[-1]:
		NE = social[-1]
		phi_NE = phi_range[-1]
	else:
		l = max([phi_range[i] for i in range(len(phi_range)) if U1s[i] > U2s[i]])
		r = min([phi_range[i] for i in range(len(phi_range)) if U1s[i] < U2s[i]])
		m = (l + r) / 2
		phi_NE, NE = separable_NE_searcher(beta1, beta2, gamma, epsilon, payment_ratio, l, r, m)

	# plot
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(phi_range, UG1s, label='1')
	ax1.plot(phi_range, UG2s, label='2')
	ax1.plot(phi_range, social, label='social')
	ax2.plot(phi_range, U1s, label='1')
	ax2.plot(phi_range, U2s, label='2')
	ax1.axvline(phi_OPT, linestyle=':', color='grey')
	ax1.axhline(OPT, linestyle=':', color='grey')
	ax2.axvline(phi_NE, linestyle=':', color='grey')
	ax2.axhline(NE, linestyle=':', color='grey')
	ax1.legend()
	ax2.legend()
	POA = OPT / NE
	print("POA simulated=\n", POA)
	print("POA bound=\n", np.exp(R1) / R1 + 1 / R2 - 1)
	print((S2s[-1] / (1 - phi_range[-1])) / (S1s[-1] / phi_range[-1]))
	print(OPT)
	plt.show()
	return


def separable_three_group_POA_comparison(beta1=2 / 14, beta2=1 / 14, beta3=1 / 14, gamma=1 / 14, epsilon=0.0001, p2=0.5,
										 p3=0.3):
	phi_range = [i / phi_steps_3D_sep for i in range(1, phi_steps_3D_sep)]
	S1s = [0]
	S2s = [0]
	S3s = [0]
	for phi in phi_range:
		S1s.append(one_group_cvxpy(beta1, gamma, epsilon, phi))
		S2s.append(one_group_cvxpy(beta2, gamma, epsilon, phi))
		S3s.append(one_group_cvxpy(beta3, gamma, epsilon, phi))

	UG1 = []
	UG2 = []
	UG3 = []
	INDIV1 = []
	INDIV2 = []
	INDIV3 = []
	X = []
	Y = []
	social = []
	surface = []
	surface1 = []
	X1 = []
	Y1 = []
	surface2 = []
	X2 = []
	Y2 = []
	surface3 = []
	X3 = []
	Y3 = []
	colors = []
	for i in range(1, phi_steps_3D_sep):
		for j in range(1, phi_steps_3D_sep):
			k = phi_steps_3D_sep - i - j
			if k <= 0:
				continue
			phi1 = i / phi_steps_3D_sep
			phi2 = j / phi_steps_3D_sep
			phi3 = k / phi_steps_3D_sep
			S1 = S1s[i]
			S2 = S2s[j]
			S3 = S3s[k]
			X.append(phi1)
			Y.append(phi2)
			UG1.append(S1)
			UG2.append(S2 * p2)
			UG3.append(S3 * p3)
			social.append(S1 + S2 * p2 + S3 * p3)
			INDIV1.append(S1 / phi1)
			INDIV2.append(S2 * p2 / phi2)
			INDIV3.append(S3 * p3 / phi3)

			if INDIV1[-1] > INDIV2[-1]:
				if INDIV1[-1] > INDIV3[-1]:
					surface.append(INDIV1[-1])
					surface1.append(INDIV1[-1])
					colors.append(cdict['red'])
					X1.append(phi1)
					Y1.append(phi2)
				else:
					surface.append(INDIV3[-1])
					surface3.append(INDIV3[-1])
					colors.append(cdict['blue'])
					X3.append(phi1)
					Y3.append(phi2)
			elif INDIV2[-1] > INDIV3[-1]:
				surface.append(INDIV2[-1])
				surface2.append(INDIV2[-1])
				colors.append(cdict['green'])
				X2.append(phi1)
				Y2.append(phi2)
			else:
				surface.append(INDIV3[-1])
				surface3.append(INDIV3[-1])
				colors.append(cdict['blue'])
				X3.append(phi1)
				Y3.append(phi2)

	fig = plt.figure()
	ax1 = fig.add_subplot(221, projection='3d')
	ax2 = fig.add_subplot(222, projection='3d')
	ax3 = fig.add_subplot(223, projection='3d')
	# ax4 = fig.add_subplot(224, projection='3d')

	ax1.plot_trisurf(X, Y, social, cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	ax1.set_title('social')

	ax2.plot_trisurf(X, Y, surface, cmap=cm.coolwarm)
	# ax2.set_xlabel(r'$\phi_1$')
	# ax2.set_ylabel(r'$\phi_2$')
	# ax2.set_title('NE')

	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_ylabel(r'$\phi_2$')
	ax2.set_title('Individual')
	# ax2.set_zlim(0, 1)
	# ax2.plot_trisurf(X1, Y1, surface1, color='red', label='1')
	# ax2.plot_trisurf(X2, Y2, surface2, color='green', label='2')
	# ax2.plot_trisurf(X3, Y3, surface3, color='blue', label='3')

	ax3.plot_trisurf(X1, Y1, surface1, color='red', label='1')
	ax3.plot_trisurf(X2, Y2, surface2, color='green', label='2')
	ax3.plot_trisurf(X3, Y3, surface3, color='blue', label='3')

	# ax2.legend()
	plt.show()
	return


def separable_NE_searcher(beta1, beta2, gamma, epsilon, payment_ratio, l, r, m):
	"""
	search for Nash equilibrium in given instance
	"""
	# phi_ms = []
	# individuals = []
	U1 = one_group_cvxpy(beta1, gamma, epsilon, m) * payment_ratio / m
	U2 = one_group_cvxpy(beta2, gamma, epsilon, 1 - m) / (1 - m)
	for _ in range(binary_iter):
		if U1 > U2:
			l = m
		else:
			r = m
		m = (l + r) / 2
		U1 = one_group_cvxpy(beta1, gamma, epsilon, m) * payment_ratio / m
		U2 = one_group_cvxpy(beta2, gamma, epsilon, 1 - m) / (1 - m)
	return m, U1


def separable_OPT_searcher(beta1, beta2, gamma, epsilon, payment_ratio, l, r, m):
	"""
	search for social OPT in given instance
	"""
	# phi_ms = []
	# socials = []
	UG_m = social_evaluator(beta1, beta2, gamma, epsilon, payment_ratio, m)
	# phi_ms.append(m)
	# socials.append(UG_m)
	for _ in range(binary_iter):
		l2 = (l + m) / 2
		r2 = (m + r) / 2
		UG_l = social_evaluator(beta1, beta2, gamma, epsilon, payment_ratio, l2)
		UG_r = social_evaluator(beta1, beta2, gamma, epsilon, payment_ratio, r2)
		if UG_l <= UG_m and UG_r <= UG_m:
			l = l2
			r = r2
		elif UG_l > UG_m:
			r = m
			m = (l + r) / 2
			UG_m = social_evaluator(beta1, beta2, gamma, epsilon, payment_ratio, m)
		# phi_ms.append(m)
		# socials.append(UG_m)
		else:
			l = m
			m = (l + r) / 2
			UG_m = social_evaluator(beta1, beta2, gamma, epsilon, payment_ratio, m)
	# phi_ms.append(m)
	# socials.append(UG_m)
	# fig = plt.figure()
	# ax1 = fig.add_subplot()
	# ax1.plot(phi_ms, socials)
	# plt.show()
	return m, UG_m


def social_evaluator(beta1, beta2, gamma, epsilon, payment_ratio, phi):
	"""
	evaluate the social utility at given point
	"""
	S1 = one_group_cvxpy(beta1, gamma, epsilon, phi)
	S2 = one_group_cvxpy(beta2, gamma, epsilon, 1 - phi)
	social = S1 * payment_ratio + S2
	return social


def one_group_comparison(beta=3 / 14, gamma=1 / 14, epsilon=0.0001):
	"""
	compare the one-group plots of binary search and convex program
	"""
	phi_range = np.arange(phi_step, 1, phi_step)
	final_size_binary = []
	final_size_cvxpy = []
	for phi in phi_range:
		final_size_binary.append(one_group_binary_search(beta, gamma, epsilon, phi))
		final_size_cvxpy.append(one_group_cvxpy(beta, gamma, epsilon, phi))

	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi_range, final_size_binary, label='binary')
	ax1.plot(phi_range, final_size_cvxpy, label='CVXPY')
	ax1.set_xlabel(r'$\phi$')
	ax1.set_ylabel(r'$S(\infty)$')
	ax1.legend()
	plt.show()
	return


def two_group_comparison(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.9):
	"""
	compare the two-group plots of binary search and convex program
	"""
	phi1_range = np.arange(phi_step, 1, phi_step)
	b11 = beta
	b12 = b21 = kappa * beta
	b22 = kappa * kappa * beta
	betas = [b11, b12, b21, b22]
	S1_binary = []
	S2_binary = []
	S1_cvxpy = []
	S2_cvxpy = []
	for phi1 in phi1_range:
		phi2 = 1 - phi1
		S1, S2 = final_size_searcher_binary(phi1, beta, kappa, gamma, epsilon)
		S1_binary.append(S1)
		S2_binary.append(S2)
		S1, S2 = two_group_cvxpy(betas, gamma, epsilon, phi1)
		S1_cvxpy.append(S1)
		S2_cvxpy.append(S2)

	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1_range, S1_binary, label='S1_binary')
	ax1.plot(phi1_range, S2_binary, label='S2_binary')
	ax1.plot(phi1_range, S1_cvxpy, label='S1_cvxpy')
	ax1.plot(phi1_range, S2_cvxpy, label='S2_cvxpy')
	ax1.set_xlabel(r'$\phi$')
	ax1.set_ylabel(r'$S(\infty)$')
	ax1.legend()
	plt.show()
	return


def two_group_utility_cvxpy(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.9, payment2=0.8):
	"""
	two-group utility plots of convex program
	"""
	phi1_range = np.arange(phi_step, 1, phi_step)

	# b11 = kappa * beta
	# b12 = b21 = beta
	# b22 = kappa * beta

	b11 = beta
	b12 = b21 = kappa * beta
	b22 = kappa * kappa * beta
	betas = [b11, b12, b21, b22]

	UG1 = []
	UG2 = []
	INDIV1 = []
	INDIV2 = []
	social = []

	for phi1 in phi1_range:
		S1, S2 = two_group_cvxpy(betas, gamma, epsilon, phi1)
		UG1.append(S1)
		UG2.append(S2 * payment2)
		social.append(S1 + S2 * payment2)
		INDIV1.append(S1 / phi1)
		INDIV2.append(S2 * payment2 / (1 - phi1))

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(phi1_range, UG1, label='G1')
	ax1.plot(phi1_range, UG2, label='G2')
	ax1.plot(phi1_range, social, label='Social')
	ax2.plot(phi1_range, INDIV1, label='G1')
	ax2.plot(phi1_range, INDIV2, label='G2')
	ax1.set_title('Group utility')
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$S(\infty)$')
	ax2.set_title('Individual utility')
	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_ylabel(r'$S(\infty)/\phi$')
	ax1.legend()
	ax2.legend()
	plt.show()
	return


def three_group_utility_cvxpy_tri(betas, gamma=1 / 14, epsilon=0.0001, payment2=1, payment3=1):
	"""
	three-group utility plots of convex program
	"""

	# print(betas)

	# b11, b12, b13, b21, b22, b23, b31, b32, b33 = betas

	UG1 = []
	UG2 = []
	UG3 = []
	INDIV1 = []
	INDIV2 = []
	INDIV3 = []
	social = []
	X = []
	Y = []

	for i in range(1, phi_steps_3D):
		for j in range(1, phi_steps_3D):
			k = phi_steps_3D - i - j
			if k <= 0:
				continue
			phi1 = i / phi_steps_3D
			phi2 = j / phi_steps_3D
			phi3 = k / phi_steps_3D
			S1, S2, S3 = three_group_cvxpy(betas, gamma, epsilon, phi1, phi2, phi3)
			X.append(phi1)
			Y.append(phi2)
			UG1.append(S1)
			UG2.append(S2 * payment2)
			UG3.append(S3 * payment3)
			social.append(S1 + S2 * payment2 + S3 * payment3)
			INDIV1.append(S1 / phi1)
			INDIV2.append(S2 * payment2 / phi2)
			INDIV3.append(S3 * payment3 / phi3)
	# print(phi1)

	ret = True
	# for i in range(len(X)):
	# 	if INDIV1[i] > INDIV2[i]:
	# 		print('Group 1 above Group 2')
	# 		ret = False
	# 		break
	# 	if INDIV2[i] > INDIV3[i]:
	# 		print('Group 2 above Group 3')
	# 		ret = False
	# 		break

	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')
	# ax3 = fig.add_subplot(223, projection='3d')
	# ax4 = fig.add_subplot(224, projection='3d')

	ax1.plot_trisurf(X, Y, social, cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	ax1.set_title('social')

	ax2.plot_trisurf(X, Y, INDIV1, color='red')
	# ax2.plot_trisurf(X, Y, INDIV2, color='yellow')
	# ax2.plot_trisurf(X, Y, INDIV3, color='green')
	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_ylabel(r'$\phi_2$')
	ax2.set_title('Individual')
	plt.show()
	return ret


def two_group_feasibility(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.9, phi1=0.5):
	"""
	plot the feasible region of two-group convex program
	"""

	# # decomposable betas
	# b11 = beta
	# b12 = b21 = kappa * beta
	# b22 = kappa * kappa * beta

	# # interaction heavy betas
	# b11 = kappa * beta
	# b12 = b21 = beta
	# b22 = kappa * beta

	b11, b12, b21, b22 = np.random.normal(beta, 0.2, 4)
	b11 = max(b11, 0.05)
	b12 = max(b12, 0.05)
	b21 = max(b21, 0.05)
	b22 = max(b22, 0.05)

	betas = [b11, b12, b21, b22]

	phi2 = 1 - phi1
	S_step = 0.002
	# S1_range = np.arange(S_step, phi1, S_step)
	# S2_range = np.arange(S_step, phi2, S_step)
	S1_range = np.arange(0, phi1 + S_step, S_step)
	S2_range = np.arange(0, phi2 + S_step, S_step)

	f1_S1 = []
	f1_S2 = []
	for S2 in S2_range:
		S1 = zero_searcher(f1, 0, phi1, [S2, b11, b12, gamma, epsilon, phi1, phi2])
		if 0 <= S1 <= phi1 or 0 <= S2 <= phi2:
			f1_S1.append(S1)
			f1_S2.append(S2)

	f2_S1 = []
	f2_S2 = []
	for S1 in S1_range:
		S2 = zero_searcher(f2, 0, phi2, [S1, b21, b22, gamma, epsilon, phi1, phi2])
		if 0 <= S1 <= phi1 or 0 <= S2 <= phi2:
			f2_S1.append(S1)
			f2_S2.append(S2)

	S1, S2 = two_group_cvxpy(betas, gamma, epsilon, phi1)

	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(f1_S1, f1_S2, label='f1')
	ax1.plot(f2_S1, f2_S2, label='f2')
	# ax1.plot([0, phi1], [0, phi2])
	ax1.axhline(S2, linestyle=':', color='grey')
	ax1.axvline(S1, linestyle=':', color='grey')
	ax1.plot(S1, S2, marker="o", markersize=5, c='red')
	width = min(phi1 - S1, S1)
	height = min(phi2 - S2, S2)
	ax1.set_xlim(S1 - width, S1 + width)
	ax1.set_ylim(S2 - height, S2 + height)
	ax1.legend()
	ax1.set_aspect(width / height)
	ax1.set_title(f'betas=[{round(b11, 3), round(b12, 3), round(b21, 3), round(b22, 3)}]')
	plt.show()
	return


def three_group_denominator(b, kappas, gamma=1 / 14, epsilon=0.0001):
	"""
	compute the denominator in Sherman Morrison formula
	"""
	# b11, b12, b13, b21, b22, b23, b31, b32, b33 = b
	b0 = b[0]
	denominators = []
	phi1s = []
	phi2s = []
	# phi3s = []
	D1 = []
	N1 = []
	D2 = []
	N2 = []
	D3 = []
	N3 = []

	for i in range(1, phi_steps_3D):
		for j in range(1, phi_steps_3D):
			k = phi_steps_3D - i - j
			if k <= 0:
				continue
			phi1 = i / phi_steps_3D
			phi2 = j / phi_steps_3D
			phi3 = k / phi_steps_3D
			S1, S2, S3 = three_group_cvxpy(b, gamma, epsilon, phi1, phi2, phi3)
			S = [S1, S2, S3]
			denominator = 1 - sum(
				[kappas[l] ** 2 * b0 * S[l]
				 /
				 (gamma - (kappas[l] - kappas[l] ** 2) * b0 * S[l])
				 for l in range(3)]
			)
			t = 0
			D1.append(gamma - (kappas[t] - kappas[t] ** 2) * b0 * S[t])
			N1.append(kappas[t] ** 2 * b0 * S[t])
			t = 1
			D2.append(gamma - (kappas[t] - kappas[t] ** 2) * b0 * S[t])
			N2.append(kappas[t] ** 2 * b0 * S[t])
			t = 2
			D3.append(gamma - (kappas[t] - kappas[t] ** 2) * b0 * S[t])
			N3.append(kappas[t] ** 2 * b0 * S[t])
			denominators.append(denominator)
			phi1s.append(phi1)
			phi2s.append(phi2)
			if denominator <= 0:
				print("**********************")
				print(f'beta={b0}')
				print(f'kappas={kappas}')
				print(f'phi={phi1, phi2, phi3}')

	fig = plt.figure()
	ax1 = fig.add_subplot(221, projection='3d')
	ax2 = fig.add_subplot(222, projection='3d')
	ax3 = fig.add_subplot(223, projection='3d')
	ax4 = fig.add_subplot(224, projection='3d')
	ax1.plot_trisurf(phi1s, phi2s, denominators, cmap=cm.coolwarm)
	ax2.plot_trisurf(phi1s, phi2s, [N1[i] / D1[i] for i in range(len(D1))], cmap=cm.coolwarm)
	ax3.plot_trisurf(phi1s, phi2s, [N2[i] / D2[i] for i in range(len(D2))], cmap=cm.coolwarm)
	ax4.plot_trisurf(phi1s, phi2s, [N3[i] / D3[i] for i in range(len(D3))], cmap=cm.coolwarm)
	# ax2.plot_trisurf(phi1s, phi2s, D1, cmap=cm.coolwarm)
	# ax3.plot_trisurf(phi1s, phi2s, D2, cmap=cm.coolwarm)
	# ax4.plot_trisurf(phi1s, phi2s, D3, cmap=cm.coolwarm)

	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_ylabel(r'$\phi_2$')
	ax3.set_xlabel(r'$\phi_1$')
	ax3.set_ylabel(r'$\phi_2$')
	ax4.set_xlabel(r'$\phi_1$')
	ax4.set_ylabel(r'$\phi_2$')
	ax1.set_title(fr'$\kappa=${kappas}')
	ax2.set_title('Group 1')
	ax3.set_title('Group 2')
	ax4.set_title('Group 3')
	plt.show()
	return


def three_group_phi_surface(b, kappas, gamma=1 / 14, epsilon=0.0001):
	"""
	plot the phi surface
	"""
	# b11, b12, b13, b21, b22, b23, b31, b32, b33 = b
	b0 = b[0]
	denominators = []
	phi1s = []
	phi2s = []
	surfaces = []

	for i in range(1, phi_steps_3D):
		for j in range(1, phi_steps_3D):
			k = phi_steps_3D - i - j
			if k <= 0:
				continue
			phi1 = i / phi_steps_3D
			phi2 = j / phi_steps_3D
			phi3 = k / phi_steps_3D
			phis = [phi1, phi2, phi3]
			S1, S2, S3 = three_group_cvxpy(b, gamma, epsilon, phi1, phi2, phi3)
			S = [S1, S2, S3]
			denominator = 1 - sum(
				[kappas[l] ** 2 * b0 * S[l]
				 /
				 (gamma - (kappas[l] - kappas[l] ** 2) * b0 * S[l])
				 for l in range(3)]
			)
			denominators.append(denominator)
			surface = 1 - sum(
				[kappas[l] ** 2 * b0 * phis[l]
				 /
				 (gamma - (kappas[l] - kappas[l] ** 2) * b0 * phis[l])
				 for l in range(3)]
			)
			surfaces.append(surface)
			phi1s.append(phi1)
			phi2s.append(phi2)
			if denominator <= 0:
				print("**********************")
				print(f'beta={b0}')
				print(f'kappas={kappas}')
				print(f'phi={phi1, phi2, phi3}')

	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')
	surfaces = np.array(surfaces)
	phi1s = np.array(phi1s)
	phi2s = np.array(phi2s)
	ax1.plot_trisurf(phi1s, phi2s, denominators, cmap=cm.coolwarm)
	ax2.plot_trisurf(phi1s[surfaces > 0], phi2s[surfaces > 0], surfaces[surfaces > 0], color='red')
	ax2.plot_trisurf(phi1s[surfaces <= 0], phi2s[surfaces <= 0], surfaces[surfaces <= 0], color='blue')

	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_ylabel(r'$\phi_2$')
	ax1.set_title(fr'$\kappa=${kappas}')
	ax2.set_title(r'$\phi$ surface')
	plt.show()
	return


def three_group_path(b, k, gamma=1 / 14, epsilon=0.0001, phi3=0.5):
	"""
	d_phi1 fixing phi3 varying phi1 and phi2
	"""
	b11, b12, b13, b21, b22, b23, b31, b32, b33 = b
	phi1_runs = 200
	INDIV1 = []
	phi1s = []
	phi2s = []
	derivatives = []
	# derivatives2 = [0]
	denominators = []
	for i in range(1, phi1_runs):
		phi1 = (1 - phi3) * i / phi1_runs
		phi2 = 1 - phi1 - phi3
		# print(phi1, phi2, phi3)
		S1, S2, S3 = three_group_cvxpy(b, gamma, epsilon, phi1, phi2, phi3)
		S = [S1, S2, S3]
		denominator = 1 - sum(
			[(k[j] ** 2 * b[0] * S[j])
			 /
			 (gamma - (k[j] - k[j] ** 2) * b[0] * S[j]) for j in range(3)]
		)
		denominators.append(denominator)
		S1_h = S1 / phi1
		S2_h = S2 / phi2
		S3_h = S3 / phi3
		phi1s.append(phi1)
		phi2s.append(phi2)
		INDIV1.append(S1_h)
		derivatives.append(
			S1_h * (b11 / gamma * (S1_h - 1) - b12 / gamma * (S2_h - 1))
			+
			(b[0] / gamma * (S1_h - 1) - b[1] / gamma * (S2_h - 1))
			*
			k[0] * b[0] * S1_h * S1 / (gamma - (k[0] - k[0] ** 2) * b[0] * S[0]) / denominator
			+
			(b[3] / gamma * (S1_h - 1) - b[4] / gamma * (S2_h - 1))
			*
			k[1] * b[0] * S1_h * S2 / (gamma - (k[1] - k[1] ** 2) * b[0] * S[1]) / denominator
			+
			(b[6] / gamma * (S1_h - 1) - b[7] / gamma * (S2_h - 1))
			*
			k[2] * b[0] * S1_h * S3 / (gamma - (k[2] - k[2] ** 2) * b[0] * S[2]) / denominator
		)
	derivatives2 = [(INDIV1[i] - INDIV1[i - 1]) / ((1 - phi3) / phi1_runs) for i in range(1, len(INDIV1))]
	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(phi1s, derivatives, label='formula')
	ax1.plot(phi1s[1:], derivatives2, label='numerical')
	ax1.legend()
	ax1.set_title('derivative')
	ax1.set_xlabel(r'$\phi_1$')
	# ax1.set_ylim(-0. - 5, 1.05)
	plt.show()
	return


def three_group_utility(b, kappas, gamma=1 / 14, epsilon=0.0001, p2=0.5, p3=0.5):
	"""
	plot the utility of 3 group interacting
	"""
	# b11, b12, b13, b21, b22, b23, b31, b32, b33 = b
	b0 = b[0]
	denominators = []
	phi1s = []
	phi2s = []
	social = []
	surfaces = []
	U1s = []
	U2s = []
	U3s = []

	for i in range(1, phi_steps_3D):
		for j in range(1, phi_steps_3D):
			k = phi_steps_3D - i - j
			if k <= 0:
				continue
			phi1 = i / phi_steps_3D
			phi2 = j / phi_steps_3D
			phi3 = k / phi_steps_3D
			phis = [phi1, phi2, phi3]
			S1, S2, S3 = three_group_cvxpy(b, gamma, epsilon, phi1, phi2, phi3)
			S = [S1, S2, S3]
			social.append(S1 + S2 * p2 + S3 * p3)
			U1s.append(S1 / phi1)
			U2s.append(p2 * S2 / phi2)
			U3s.append(p3 * S3 / phi3)
			phi1s.append(phi1)
			phi2s.append(phi2)
			if i == phi_steps_3D - 2:
				print((S1 / phi1) / (S2 / phi2))
				print((S1 / phi1) / (S3 / phi3))

	R0 = b[0] / gamma
	print('R0=', R0)
	print('POA=', max(social) / min(social))
	print(np.exp(R0) / R0)

	phi1s = np.array(phi1s)
	phi2s = np.array(phi2s)
	U1s = np.array(U1s)
	U2s = np.array(U2s)
	U3s = np.array(U3s)
	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')
	# ax3 = fig.add_subplot(223, projection='3d')
	surfaces = np.array(surfaces)
	phi1s = np.array(phi1s)
	phi2s = np.array(phi2s)
	ax1.plot_trisurf(phi1s, phi2s, social, cmap=cm.coolwarm)

	# ax2.plot_trisurf(phi1s[l_max(U1s, U2s, U3s)],
	# 				 phi2s[l_max(U1s, U2s, U3s)],
	# 				 U1s[l_max(U1s, U2s, U3s)],
	# 				 color='red')
	# ax2.plot_trisurf(phi1s[l_max(U2s, U1s, U3s)],
	# 				 phi2s[l_max(U2s, U1s, U3s)],
	# 				 U2s[l_max(U2s, U1s, U3s)],
	# 				 color='green')
	# ax2.plot_trisurf(phi1s[l_max(U3s, U2s, U1s)],
	# 				 phi2s[l_max(U3s, U2s, U1s)],
	# 				 U3s[l_max(U3s, U2s, U1s)],
	# 				 color='blue')
	# ax2.plot_trisurf(phi1s, phi2s, U2s, color='green')
	# ax2.plot_trisurf(phi1s, phi2s, U3s, color='blue')

	ax2.plot_trisurf(phi1s, phi2s, U1s, color='red')
	ax2.plot_trisurf(phi1s, phi2s, U2s, color='green')
	ax2.plot_trisurf(phi1s, phi2s, U3s, color='blue')

	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	ax2.set_xlabel(r'$\phi_1$')
	ax2.set_ylabel(r'$\phi_2$')
	ax1.set_title('social')
	ax2.set_title('individual')
	plt.show()
	return


def make_betas_dec(b0, kappas):
	k1, k2, k3 = kappas
	betas = [k1 * k1, k1 * k2, k1 * k3,
			 k2 * k1, k2 * k2, k2 * k3,
			 k3 * k1, k3 * k2, k3 * k3]
	betas = [i * b0 for i in betas]
	return betas


def make_betas_net(b0, kappas):
	k1, k2, k3 = kappas
	betas = [k1, k1 * k2, k1 * k3,
			 k2 * k1, k2, k2 * k3,
			 k3 * k1, k3 * k2, k3]
	betas = [i * b0 for i in betas]
	return betas


def three_group():
	beta = 3 / 14

	# betas = np.random.normal(beta, 0.2, 9)
	# betas = [max(0.05, beta) for beta in betas]

	# kappas = [1, 0.6, 0.4]
	# betas = make_betas(beta, kappas)
	# three_group_denominator(beta, kappas, gamma=1 / 14, epsilon=0.0001)

	# b1 = kappa1 * kappa1 * beta
	# b2 = kappa2 * kappa2 * beta
	# b3 = kappa3 * kappa3 * beta

	# ret = True
	# for _ in range(10):
	# 	betas = [b1, np.random.uniform(b2, b1), np.random.uniform(b3, b1),
	# 			 np.random.uniform(b2, b1), b2, np.random.uniform(b3, b2),
	# 			 np.random.uniform(b3, b1), np.random.uniform(b3, b2), b3]
	# 	print(betas)
	# 	ret = three_group_utility_cvxpy_tri(betas, gamma=1 / 14, epsilon=0.0001, payment2=1, payment3=1)
	# 	if not ret:
	# 		break
	# print('Passed' if ret else 'Failed')
	# betas = [b1, uni(0, b1), uni(0, b1),
	# 		 uni(0, b1), uni(0, b1), uni(0, b1),
	# 		 uni(0, b1), uni(0, b1), uni(0, b1)]
	# three_group_utility_cvxpy_tri(betas, gamma=1 / 14, epsilon=0.0001, payment2=1, payment3=1)

	# for phi3 in np.arange(0.1, 1, 0.1):
	# 	three_group_path(betas, kappas, gamma=1 / 14, epsilon=0.0001, phi3=phi3)

	# test the denominator
	# kappas = [1, 0.6, 0.4]
	# betas = make_betas(beta, kappas)
	# three_group_denominator(betas, kappas, gamma=1 / 14, epsilon=0.0001)

	kappas = [1, 0.8, 0.3]
	# betas = make_betas_net(beta, kappas)
	betas = make_betas_dec(beta, kappas)
	# three_group_denominator(betas, kappas, gamma=1 / 14, epsilon=0.0001)
	# three_group_phi_surface(betas, kappas, gamma=1 / 14, epsilon=0.0001)
	three_group_utility(betas, kappas, gamma=1 / 14, epsilon=0.0001, p2=0.5771771080881023, p3=0.14607704672221766)

	# kappas = [1, 0.9, 0.2]
	# betas = make_betas(beta, kappas)
	# three_group_denominator(betas, kappas, gamma=1 / 14, epsilon=0.0001)
	# kappas = [1, 0.2, 0.1]
	# betas = make_betas(beta, kappas)
	# three_group_denominator(betas, kappas, gamma=1 / 14, epsilon=0.0001)
	return


def main():
	# one_group_comparison()
	# separable_two_group_POA_comparison(beta1=4 / 14, beta2=1 / 14, gamma=1 / 14, epsilon=0.0001,
	# 								   payment_ratio=50.22135161418591)
	# separable_three_group_POA_comparison(beta1=8 / 14, beta2=7 / 14, beta3=6 / 14, gamma=1 / 14, epsilon=0.0001,
	# 									 p2=0.9, p3=0.7)
	# two_group_comparison(beta=2 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3)
	# two_group_utility_cvxpy(beta=2 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3, payment2=1.1)

	# two_group_feasibility(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3, phi1=0.5)

	# three_group_feasibility_scatter(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3, phi1=0.4, phi2=0.3)
	three_group()
	return


if __name__ == '__main__':
	main()
