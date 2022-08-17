import numpy as np
import matplotlib.pyplot as plt


def two_group_simulate(phi1, phi2, beta1, beta2, gamma, epsilon, T, num_steps, plot=False):
	S1 = [phi1 * (1 - epsilon)]
	I1 = [phi1 * epsilon]
	R1 = [0]
	S2 = [phi2 * (1 - epsilon)]
	I2 = [phi2 * epsilon]
	R2 = [0]
	dt = T / num_steps
	t_range = np.arange(0, T + dt, dt)
	for t in t_range[1:]:
		dS1 = -(beta1 * I1[-1] + beta2 * I2[-1]) * S1[-1] * dt
		dI1 = ((beta1 * I1[-1] + beta2 * I2[-1]) * S1[-1] - gamma * I1[-1]) * dt
		dR1 = gamma * I1[-1] * dt
		dS2 = -(beta1 * I1[-1] + beta2 * I2[-1]) * S2[-1] * dt
		dI2 = ((beta1 * I1[-1] + beta2 * I2[-1]) * S2[-1] - gamma * I2[-1]) * dt
		dR2 = gamma * I2[-1] * dt
		S1.append(S1[-1] + dS1)
		I1.append(I1[-1] + dI1)
		R1.append(R1[-1] + dR1)
		S2.append(S2[-1] + dS2)
		I2.append(I2[-1] + dI2)
		R2.append(R2[-1] + dR2)
	if plot:
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(122)
		ax1.plot(t_range[-round(num_steps / 10):], S1[-round(num_steps / 10):], label='S1')
		ax1.plot(t_range[-round(num_steps / 10):], S2[-round(num_steps / 10):], label='S2')
		ax2.plot(t_range, I1, label='I1')
		ax2.plot(t_range, I2, label='I2')
		ax1.legend()
		ax2.legend()
		plt.show()
	return t_range, S1, S2


def main():
	two_group_simulate(0.1, 0.9, 1, 0.5, 1/14, 0.0001, 1000, 10000, True)
	return


if __name__ == '__main__':
	main()
