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
		if Is[i] <= I_0:
			print('S=', S_range[i], 'I=', Is[i])
			break
	print(i, len(S_range))
	return


def main():
	tests()
	return


if __name__ == '__main__':
	main()
