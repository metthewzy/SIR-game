import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def f(S, phi, beta, gamma, epsilon):
	ret = S - (1 - epsilon) * np.exp(beta / gamma * phi * (S - 1))
	return ret


def f_plotter(beta, gamma, epsilon):
	phi_range = np.linspace(0, 1, 100)
	S_range = np.linspace(0, 1, 100)
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


def main():
	f_plotter(beta=4 / 14, gamma=1 / 14, epsilon=0.0001)
	return


if __name__ == '__main__':
	main()
