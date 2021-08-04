import numpy as np
import matplotlib as plt

S_0 = 1
I_0 = 0.0001
gamma = 1 / 7
beta = 0.5


def I_t(S_t):
    return I_0 + gamma / beta * np.log(S_t / S_0) - (S_t - S_0)


def main():
    print('S_peak=', gamma / beta)
    print('I_peak=', I_t(gamma / beta))
    return


if __name__ == '__main__':
    main()
