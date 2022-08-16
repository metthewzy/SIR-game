import numpy as np
import matplotlib.pyplot as plt


def two_group_simulate(phi1, phi2, beta1, beta2, gamma, epsilon, T, num_steps):
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
	return


def main():
	return


if __name__ == '__main__':
	main()
