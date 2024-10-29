import numpy as np
import matplotlib.pyplot as plt

k1 = 1
k2 = 0.8
R0 = 2
epsilon = 0.01
binary_iters = 20
phi1 = 0.6
phi2 = 1 - phi1
S_step = 0.01


def eq1(para):
    S1, S2 = para
    res = S1 - (1 - epsilon) * np.exp(k1 * X0(S1, S2))
    return res


def eq2(para):
    S2, S1 = para
    res = S2 - (1 - epsilon) * np.exp(k2 * X0(S1, S2))
    return res


def binary(f, para, lower, upper):
    f_lower, f_upper = f([lower] + para), f([upper] + para)
    for _ in range(binary_iters):
        mid = (lower + upper) / 2
        f_mid = f([mid] + para)
        if f_mid * f_lower > 0:
            lower, f_lower = mid, f_mid
        else:
            upper, f_upper = mid, f_mid
    return (lower + upper) / 2


def X0(S1, S2):
    res = R0 * (k1 * phi1 * (S1 - 1) + k2 * phi2 * (S2 - 1))
    return res


def f1(S):
    res = 5 * np.sqrt(3 * S / 5)
    return res


def f2(S):
    res = 1.45 * np.log(12 * (S + 1 / 12))
    return res


def plot_S_bar_feasible_region():
    fig = plt.figure()
    ax1 = fig.add_subplot()
    S1_range = S2_range = np.arange(0, 1 - epsilon + S_step, S_step)

    S1_eq1 = []
    for S2 in S2_range:
        S1_eq1.append(binary(eq1, [S2], 0, 1 - epsilon))
    S2_eq2 = []
    for S1 in S1_range:
        S2_eq2.append(binary(eq2, [S1], 0, 1 - epsilon))

    ax1.plot(S1_eq1, S2_range, label='eq1')
    ax1.plot(S1_range, S2_eq2, label='eq2')
    ax1.set_xlabel(r'$\overline{S}_1$')
    ax1.set_ylabel(r'$\overline{S}_2$')
    ax1.legend()
    plt.show()
    return


def main():
    plot_S_bar_feasible_region()


if __name__ == '__main__':
    main()
