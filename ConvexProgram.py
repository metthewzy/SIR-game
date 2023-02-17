import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform as uni
from matplotlib import cm
from scipy.optimize import minimize
import cvxpy as cp
from TwoGroup import final_size_searcher_binary

binary_iter = 40
phi_step = 0.005
phi_steps_3D = 40


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


def three_group_denominator(beta, kappas, gamma=1 / 14, epsilon=0.0001):
	"""
	compute the denominator in Sherman Morrison formula
	"""
	kappa1, kappa2, kappa3 = kappas
	betas = [kappa1, kappa1 * kappa2, kappa1 * kappa3,
			 kappa2 * kappa1, kappa2, kappa2 * kappa3,
			 kappa3 * kappa1, kappa3 * kappa2, kappa3]
	D = []
	D2 = []
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
			S = [S1, S2, S3]
			X.append(phi1)
			Y.append(phi2)
			D.append(1 + sum([(kappas[l] * kappas[l] * beta * S[l]) /
							  (kappas[l] * (1 - kappas[l]) * beta * S[l] - gamma)
							  for l in range(3)]))
			D2.append(sum([kappas[l] * beta * S[l] for l in range(3)]))
	fig = plt.figure()
	ax1 = fig.add_subplot(projection='3d')
	ax1.plot_trisurf(X, Y, D, cmap=cm.coolwarm)
	ax1.set_xlabel(r'$\phi_1$')
	ax1.set_ylabel(r'$\phi_2$')
	# ax1.plot(range(len(D2)), D2)
	# ax1.axhline(gamma)
	plt.show()
	return


def three_group_path(b, k, gamma=1 / 14, epsilon=0.0001, phi3=0.5):
	"""
	d_phi1 on a straight path fixing the ratio of phi2 and phi3
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


def three_group():
	beta = 1.5 / 14

	# betas = np.random.normal(beta, 0.2, 9)
	# betas = [max(0.05, beta) for beta in betas]

	kappa1 = 1
	kappa2 = 0.6
	kappa3 = 0.4

	kappas = [kappa1, kappa2, kappa3]

	betas = [kappa1, kappa1 * kappa2, kappa1 * kappa3,
			 kappa2 * kappa1, kappa2, kappa2 * kappa3,
			 kappa3 * kappa1, kappa3 * kappa2, kappa3]
	betas = [i * beta for i in betas]
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

	for phi3 in np.arange(0.1, 1, 0.1):
		three_group_path(betas, kappas, gamma=1 / 14, epsilon=0.0001, phi3=phi3)
	return


def main():
	# one_group_comparison()
	# two_group_comparison(beta=2 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3)
	# two_group_utility_cvxpy(beta=2 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3, payment2=1.1)

	# two_group_feasibility(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3, phi1=0.5)

	# three_group_feasibility_scatter(beta=3 / 14, gamma=1 / 14, epsilon=0.0001, kappa=0.3, phi1=0.4, phi2=0.3)
	three_group()
	return


if __name__ == '__main__':
	main()
