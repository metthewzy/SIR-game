import numpy as np
import matplotlib.pyplot as plt


def g_plotter():
	beta = 0.25
	gamma = 1 / 14
	phi = 0.5
	S_step = 0.001
	S_range = np.arange(0, 1 + S_step, S_step)

	def g(S):
		ret = S - np.exp(-phi * beta / gamma * (1 - S))
		return ret
	gs = []
	for S in S_range:
		gs.append(g(S))

	fig = plt.figure()
	ax1 = fig.add_subplot()
	ax1.plot(S_range, gs)
	# ax1.set_aspect('equal')
	# ax1.set_ylim(-0.3, 0.3)
	ax1.grid(True, which='both')

	ax1.axhline(y=0, color='k')
	ax1.axvline(x=0, color='k')
	ax1.set_xlabel(r'$\overline{S}$')
	ax1.set_ylabel(r'$g(\overline{S})$')
	fig.savefig('g(S).png')
	# plt.show()
	return


def main():
	g_plotter()
	return


if __name__ == '__main__':
	main()
