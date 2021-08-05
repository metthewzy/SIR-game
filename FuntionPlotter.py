import numpy as np
import matplotlib as plt

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
		if abs(I_middle - I_0) / I_0 < 0.001:
			break
		if I_middle < I_0:
			S_left = S_middle
		else:
			S_right = S_middle
	S_end = S_middle
	I_end = I_t(S_end)
	print(f'S_end={S_end} I_end={I_end}')

	return


def main():
	tests()
	return


if __name__ == '__main__':
	main()
