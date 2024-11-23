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


def f2(S):
    return 0.5 * np.log2(S+1)


def f1(S):
    return S


def df(S, U):
    ret = [(U[i] - U[i - 1]) / (S[i] - S[i - 1]) for i in range(1, len(U))]
    return ret


def concave_function_comparison():
    S_low, S_high, S_step = 0, 1, 0.01
    S_range = np.arange(S_low, S_high + S_step, S_step)
    U1 = [f1(S) for S in S_range]
    U2 = [f2(S) for S in S_range]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(S_range, U1, label=rf'$U_1$')
    ax1.plot(S_range, U2, label=rf'$U_2$')
    ax1.legend()
    plt.show()
    return


def count_intersection(A1, A2):
    return sum([1 if (A1[i] - A2[i]) * (A1[i - 1] - A2[i - 1]) < 0 else 0 for i in range(1, len(A1))])


def U_vs_X0():
    X_low, X_high, X_step = -40, 1, 0.01
    X_range = np.arange(X_low, X_high + X_step, X_step)
    k1 = 1
    k2 = 0.95
    S1 = [np.exp(k1 * X) for X in X_range]
    S2 = [np.exp(k2 * X) for X in X_range]
    U1 = [f1(S) for S in S1]
    U2 = [f2(S) for S in S2]
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(X_range, U1, label=rf'$U_1$')
    ax1.plot(X_range, U2, label=rf'$U_2$')
    ax1.axvline(0, color='grey', linestyle=':')
    ax1.set_xlabel(rf'$X_0$')
    ax1.set_ylabel('Utility')
    ax1.legend()

    ax4 = fig.add_subplot(223)
    dU1 = df(X_range, U1)
    dU2 = df(X_range, U2)
    ax4.plot(X_range[1:], dU1, label=r'$dU_1/dX_0$')
    ax4.plot(X_range[1:], dU2, label=r'$dU_2/dX_0$')
    ax4.axvline(0, c='grey', linestyle=':')
    # ax4.set_ylabel('derivative')
    # ax4.set_xlabel(r'$X_0$')
    ax4.legend()
    print('count=', count_intersection(dU1, dU2))

    S_low, S_high, S_step = 0, 1, 0.01
    S_range = np.arange(S_low, S_high + S_step, S_step)
    U1 = [f1(S) for S in S_range]
    U2 = [f2(S) for S in S_range]
    ax2 = fig.add_subplot(222)
    ax2.plot(S_range, U1, label=rf'$U_1$')
    ax2.plot(S_range, U2, label=rf'$U_2$')
    ax2.set_xlabel(r'$\bar{S}$')
    ax2.set_ylabel('Utility')
    ax2.legend()

    ax3 = fig.add_subplot(224)
    X_range = np.arange(X_low, 1, X_step)
    ax3.plot(X_range, [np.exp(k1 * X) for X in X_range], label=r'$\bar{S}_1$')
    ax3.plot(X_range, [np.exp(k2 * X) for X in X_range], label=r'$\bar{S}_2$')
    ax3.axvline(0, color='grey', linestyle=':')
    ax3.set_ylabel(r'$\bar{S}$')
    ax3.set_xlabel(r'$X_0$')
    ax3.legend()

    plt.show()

    return


def tmp():
    X_low, X_high, X_step = -10, 1, 0.1
    X_range = np.arange(X_low, X_high + X_step, X_step)
    k1 = 1
    k2 = 0.5
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(X_range, [k1 / k2 * np.exp((k1 - k2) * X) for X in X_range])
    ax1.axhline(1, c='grey', linestyle=':')
    plt.show()


def main():
    U_vs_X0()
    # concave_function_comparison()
    # tmp()


if __name__ == '__main__':
    main()
