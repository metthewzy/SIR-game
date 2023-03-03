import numpy as np
import matplotlib.pyplot as plt


def new_model():
	n0 = 1000
	dt = 0.01
	epsilon = 0.001
	S = [n0 * (1 - epsilon)]
	I = [n0 * epsilon]
	R = [0]
	S2 = [n0 * (1 - epsilon)]
	I2 = [n0 * epsilon]
	R2 = [0]
	T = 100
	beta = 0.005 / 14
	gamma = 1 / 14
	t_range = np.arange(0, T, dt)
	for _ in t_range[1:]:
		dS = min((beta * S[-1] * I[-1] ** 2) * dt, S[-1])
		dI = dS - gamma * I[-1] * dt
		dR = (gamma * I[-1]) * dt
		S.append(S[-1] - dS)
		I.append(I[-1] + dI)
		R.append(R[-1] + dR)

		dS2 = min((beta * S2[-1] * I2[-1]) * dt, S2[-1])
		dI2 = dS2 - gamma * I2[-1] * dt
		dR2 = (gamma * I2[-1]) * dt
		S2.append(S2[-1] - dS2)
		I2.append(I2[-1] + dI2)
		R2.append(R2[-1] + dR2)

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	ax1.plot(t_range, S, label='S')
	ax1.plot(t_range, I, label='I')
	ax1.plot(t_range, R, label='R')
	ax1.legend()

	ax2.plot(t_range, S2, label='S')
	ax2.plot(t_range, I2, label='I')
	ax2.plot(t_range, R2, label='R')
	ax2.legend()
	plt.show()
	return


def main():
	new_model()
	return


if __name__ == '__main__':
	main()
