import matplotlib.pyplot as plt
import numpy as np


# function for X0 binary search over 2 groups
def f_2group(X0, R0, phi, kappa, epsilon):
    ret = sum(kappa_i * phi_i * ((1 - epsilon) * np.exp(kappa_i * X0) - 1)
              for kappa_i, phi_i in zip(kappa, phi))
    ret *= R0
    ret -= X0
    return ret


# binary search X0 with f(X0)=0
def X0_binary_2group(R0, phi, kappa, epsilon, n_iterations):
    left, right = -R0, 0
    mid = (left + right) / 2
    f_mid = f_2group(mid, R0, phi, kappa, epsilon)
    i = 0
    while f_mid != 0 and i < n_iterations:
        if f_mid > 0:
            left = mid
        else:
            right = mid
        mid = (left + right) / 2
        f_mid = f_2group(mid, R0, phi, kappa, epsilon)
    return mid


def final_size_2group():
    p = [1, 0.2]
    R0 = 2
    kappa = [1, 0.01]
    epsilon = 0.001
    phi1_step = 0.01
    phi1_range = np.arange(0, 1 + phi1_step, phi1_step)
    X0s = []
    S1_bars = []
    S2_bars = []
    U1s = []
    U2s = []
    # S1s = []
    # S2s = []
    UG1s = []
    UG2s = []
    for phi1 in phi1_range:
        phi2 = 1 - phi1
        phi = [phi1, phi2]
        X0s.append(X0_binary_2group(R0, phi, kappa, epsilon, n_iterations=30))
        S1_bars.append((1 - epsilon) * np.exp(kappa[0] * X0s[-1]))
        S2_bars.append((1 - epsilon) * np.exp(kappa[1] * X0s[-1]))
        U1s.append(p[0] * S1_bars[-1])
        U2s.append(p[1] * S2_bars[-1])
        # S1s.append((1 - epsilon) * phi1 * np.exp(kappa[0] * X0s[-1]))
        # S2s.append((1 - epsilon) * phi2 * np.exp(kappa[1] * X0s[-1]))
        UG1s.append(phi1 * U1s[-1])
        UG2s.append(phi2 * U2s[-1])
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # ax1.plot(phi1_range, S1s, label=r'$S_1$')
    # ax1.plot(phi1_range, S2s, label=r'$S_2$')
    ax1.plot(phi1_range, UG1s, label=r'$UG_1$')
    ax1.plot(phi1_range, UG2s, label=r'$UG_2$')
    ax1.plot(phi1_range, [UG1 + UG2 for UG1, UG2 in zip(UG1s, UG2s)], label='Social')
    ax1.set_title("Social welfare")
    ax1.legend()
    ax2.plot(phi1_range, U1s, label=r'$U_1$')
    ax2.plot(phi1_range, U2s, label=r'$U_2$')
    ax2.set_title("Individual utility")
    ax2.legend()
    plt.show()
    return


def test():
    R0 = 4
    kappa = [1, 0.2]
    phi = [0.5, 0.5]
    epsilon = 0.001
    X0_step = 0.001
    # X0 possible range from -R0 to 0
    X0_range = np.arange(-R0, 0 + X0_step, X0_step)
    f_values = [f_2group(X0, R0, phi, kappa, epsilon) for X0 in X0_range]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(X0_range, f_values, label='test')
    ax1.axhline(0, linestyle=':', color='grey')
    # ax1.legend()
    plt.show()
    return


def main():
    final_size_2group()


if __name__ == '__main__':
    main()
