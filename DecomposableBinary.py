import matplotlib.pyplot as plt
import numpy as np
import random

epsilon = 0.001


def binary_verification(R0, n):
    random.seed()
    kappa = [random.random() for _ in range(n)]
    kappa = [k / max(kappa) for k in kappa]
    kappa.sort(reverse=True)

    phi = [random.random() for _ in range(n)]
    population = sum(phi)
    phi = [p / population for p in phi]
    print('kappa=', kappa)
    print('phi=', phi)

    sums = []
    steps = 200
    low, high = -R0, 0
    X0_range = np.arange(low, high + (high - low) / steps, (high - low) / steps)
    for X0 in X0_range:
        S = [(1 - epsilon) * phi_i * np.exp(kappa_i * X0) for phi_i, kappa_i in zip(phi, kappa)]
        curr_sum = R0 * sum(
            [kappa[i] * (S[i] - phi[i]) for i in range(n)]
        )
        sums.append(curr_sum)
    figure = plt.figure()
    ax1 = figure.add_subplot()
    ax1.plot(X0_range, X0_range, label='X0 guess')
    ax1.plot(X0_range, sums, label='feedback summation')
    ax1.legend()
    plt.show()
    return


def main():
    binary_verification(R0=5, n=10)
    return


if __name__ == '__main__':
    main()
