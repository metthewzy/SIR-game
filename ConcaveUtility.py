import numpy as np
import matplotlib.pyplot as plt

"""
Utility function:
U_i(S_bar) is concave increasing
Group index i < j ==> U_i(S) > U_j(S) and dU_i/dS > dU_j/dS for all S
"""


def U(a, S_bar):
    """
    U_i = a * sqrt(S_bar)
    """
    return a * np.sqrt(S_bar)


def main():
    X_low, X_high, X_step = -5, 1, 0.01
    X_range = np.arange(X_low, X_high + X_step, X_step)
    k1 = 1
    k2 = 0.8
    a1 = 1
    a2 = 0.8
    S1 = [np.exp(k1 * X) for X in X_range]
    S2 = [np.exp(k2 * X) for X in X_range]
    U1 = [U(a1, S) for S in S1]
    U2 = [U(a2, S) for S in S2]
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(X_range, U1, label=rf'$U_1$')
    ax1.plot(X_range, U2, label=rf'$U_2$')
    ax1.axvline(0, color='grey', linestyle=':')
    ax1.set_xlabel(rf'$X_0$')
    ax1.set_ylabel('Utility')
    ax1.legend()

    S_low, S_high, S_step = 0, 1, 0.01
    S_range = np.arange(S_low, S_high+S_step, S_step)
    U1 = [U(a1, S) for S in S_range]
    U2 = [U(a2, S) for S in S_range]
    ax2 = fig.add_subplot(122)
    ax2.plot(S_range, U1, label=rf'$U_1$')
    ax2.plot(S_range, U2, label=rf'$U_2$')
    ax2.set_xlabel(r'$\bar{S}$')
    ax2.set_ylabel('Utility')
    ax2.legend()
    plt.show()
    return


if __name__ == '__main__':
    main()
