import numpy as np
import matplotlib.pyplot as plt

S_0 = 1
I_0 = 0.0001
gamma = 1 / 7
beta = 0.5


def I_t(S_t):
	return I_0 + gamma / beta * np.log(S_t / S_0) - (S_t - S_0)


def tests():
	print('I_0=', I_t(S_0))
	print('S_peak=', gamma / beta)
	print('I_peak=', I_t(gamma / beta))
	S_range = np.arange(S_0, 0, -0.00001)
	# print(S_range)
	Is = I_t(S_range)
	# print(Is)
	for i in range(1, len(S_range)):
		S_left = S_range[i]
		S_right = S_range[i - 1]
		if Is[i] < I_0:
			break
	print(S_left, S_right)

	while True:
		S_middle = (S_left + S_right) / 2
		I_middle = I_t(S_middle)
		if abs(I_middle - I_0) / I_0 < 0.0001:
			break
		if I_middle < I_0:
			S_left = S_middle
		else:
			S_right = S_middle
	S_end = S_middle
	I_end = I_t(S_end)
	print(f'S_end={S_end} I_end={I_end}')
	S_step = (S_0 - S_end) / 100000
	S_range = np.arange(S_0, S_end - S_step, - S_step)
	# print(S_range)
	dts = del_t(S_range)
	print('t_end=', np.mean(dts) * (S_end - S_0))
	return


def del_t(S):
	value = - 1 / (beta * S * I_0 + gamma * S * np.log(S) - beta * S ** 2 + 1)
	return value


def simulate():
	S = [S_0]
	I = [I_0]
	dt = 0.01
	t = np.arange(0, 10000.5 * dt, dt)
	for i in range(10000):
		dS = (- beta * S[-1] * I[-1]) * dt
		dI = (beta * S[-1] - gamma) * I[-1] * dt
		S.append(S[-1] + dS)
		I.append(I[-1] + dI)

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(t, S, label='S')
	ax.plot(t, I, label='I')
	ax.legend()
	plt.show()
	return


def main():
	tests()
	simulate()
	return


if __name__ == '__main__':
	main()
