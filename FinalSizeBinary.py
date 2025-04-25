import numpy as np
import matplotlib.pyplot as plt

binary_iterations = 200
OPT_iterations = 30
NE_iterations = 50


def final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon, plot):
    """
    binary search the final sizes of 2 groups interacting
    """
    S2s = []
    f2s = []
    phi2 = 1 - phi1
    S2_l = 0
    S2_r = phi2 * (1 - epsilon)
    for _ in range(binary_iterations):
        S2_m = (S2_l + S2_r) / 2
        S1 = S1_final_searcher(S2_m, beta, beta_ratio, gamma, epsilon, phi1)
        f = f2([S1, S2_m], phi1, beta, beta_ratio, gamma, epsilon)
        S2s.append(S2_m)
        f2s.append(f)
        if f > 0:
            S2_r = S2_m
        else:
            S2_l = S2_m
    S2 = S2_m
    S1 = S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1)
    f = f2([S1, S2_m], phi1, beta, beta_ratio, gamma, epsilon)
    S2s.append(S2_m)
    f2s.append(f)

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot()
        [ax1.plot([i, i + 1], [S2s[i], S2s[i + 1]], c='red' if S2s[i + 1] > S2s[i] else 'green') for i in
         range(len(S2s) - 1)]
        # ax1.axhline(0, c='grey', linestyle=':')
        # ax1.scatter(S2s[-1], f2s[-1], c='red')
        plt.show()
    return S1, S2


def S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1):
    S_trace = []
    f_trace = []
    S1_l = 0
    S1_r = phi1 * (1 - epsilon)
    for _ in range(binary_iterations):
        S1_m = (S1_l + S1_r) / 2
        f = f1([S1_m, S2], phi1, beta, beta_ratio, gamma, epsilon)
        S_trace.append(S1_m)
        f_trace.append(f)
        if f > 0:
            S1_r = S1_m
        else:
            S1_l = S1_m
    # print(S_trace)
    # print(f_trace)
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax1.plot(S_trace, f_trace)
    # plt.show()
    return S1_m


def S2_final_searcher(S1, beta, beta_ratio, gamma, epsilon, phi1):
    S_trace = []
    f_trace = []
    S2_l = 0
    S2_r = phi1 * (1 - epsilon)
    for _ in range(binary_iterations):
        S2_m = (S2_l + S2_r) / 2
        f = f2([S1, S2_m], phi1, beta, beta_ratio, gamma, epsilon)
        S_trace.append(S2_m)
        f_trace.append(f)
        if f > 0:
            S2_r = S2_m
        else:
            S2_l = S2_m
    # print(S_trace)
    # print(f_trace)
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax1.plot(S_trace, f_trace)
    # plt.show()
    return S2_m


def f1(point, phi1, beta, beta_ratio, gamma, epsilon):
    [S1, S2] = point
    phi2 = 1 - phi1
    b11 = beta
    b12 = beta * beta_ratio
    # b22 = beta * beta_ratio * beta_ratio
    S1_0 = phi1 * (1 - epsilon)
    # S2_0 = phi2 * (1 - epsilon)
    ret = S1 - S1_0 * np.exp(b11 / gamma * (S1 - phi1) + b12 / gamma * (S2 - phi2))
    return ret


def f2(point, phi1, beta, beta_ratio, gamma, epsilon):
    [S1, S2] = point
    phi2 = 1 - phi1
    # b11 = beta
    b21 = beta * beta_ratio
    b22 = beta * beta_ratio * beta_ratio
    # S1_0 = phi1 * (1 - epsilon)
    S2_0 = phi2 * (1 - epsilon)
    ret = S2 - S2_0 * np.exp(b21 / gamma * (S1 - phi1) + b22 / gamma * (S2 - phi2))
    return ret


def f2_final_size_plotter(phi1, beta, beta_ratio, gamma, epsilon, plot):
    phi2 = 1 - phi1
    b11 = beta
    b12 = b21 = beta * beta_ratio
    b22 = beta * beta_ratio * beta_ratio
    S2_step = 0.001
    S2_range = np.arange(0, phi2 + S2_step, S2_step)
    S1s = []
    for S2 in S2_range:
        S1s.append(S1_final_searcher(S2, beta, beta_ratio, gamma, epsilon, phi1))
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax1.plot(S2_range, S1s)
    # ax1.set_xlabel('S2')
    # ax1.set_ylabel('S1(S2)')
    # plt.show()
    # plt.close(fig)

    df2s = []
    f2s = []
    for S1, S2 in zip(S1s, S2_range):
        X = (b11 * (S1 - phi1) + b12 * (S2 - phi2)) / gamma
        Y = (b21 * (S1 - phi1) + b22 * (S2 - phi2)) / gamma
        df2s.append(1 - phi2 * np.exp(Y) * (b21 / gamma * S1 * b12 / gamma / (1 - S1 * b11 / gamma) + b22 / gamma))
        f2s.append(f2([S1, S2], phi1, beta, beta_ratio, gamma, epsilon))

    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax1.plot(S2_range, df2s)
    # ax1.set_xlabel('S2')
    # ax1.set_ylabel('dfs/dS2')
    # plt.show()
    # plt.close(fig)

    S2_range_final = np.arange(0, phi2 + S2_step, S2_step)
    S1_final, S2_final = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon, False)
    f2s_final = []
    for S2 in S2_range_final:
        f2s_final.append(f2([S1_final, S2], phi1, beta, beta_ratio, gamma, epsilon))
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.set_title(rf'f2 comparison $\phi_1$={phi1}')
    ax1.set_xlabel('S2')
    l1 = ax1.plot(S2_range, f2s, label='S1(S2)')
    l2 = ax1.plot(S2_range_final, f2s_final, label='final S1')
    ax2.plot(S2_range, df2s, label='df2', c='red')
    ax1.axhline(0, c='grey', linestyle=':')
    ax1.legend()
    plt.show()
    return


def f1_final_size_plotter(phi1, beta, beta_ratio, gamma, epsilon, plot):
    phi2 = 1 - phi1
    S1_step = 0.001
    S1_range = np.arange(0, phi2 + S1_step, S1_step)
    S2s = []
    for S1 in S1_range:
        S2s.append(S2_final_searcher(S1, beta, beta_ratio, gamma, epsilon, phi1))
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(S1_range, S2s)
    plt.show()
    plt.close(fig)

    f1s = []
    for S1, S2 in zip(S1_range, S2s):
        f1s.append(f1([S1, S2], phi1, beta, beta_ratio, gamma, epsilon))

    S1_range_final = np.arange(0, 2 + S1_step, S1_step)
    S1_final, S2_final = final_size_searcher_binary(phi1, beta, beta_ratio, gamma, epsilon, False)
    f1s_final = []
    for S1 in S1_range_final:
        f1s_final.append(f1([S1, S2_final], phi1, beta, beta_ratio, gamma, epsilon))
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_title('f1 comparison')
    ax1.plot(S1_range, f1s, label='S2(S1)')
    ax1.plot(S1_range_final, f1s_final, label='final S2')
    ax1.axhline(0, c='grey', linestyle=':')
    ax1.legend()
    plt.show()
    return


def interacting_binary():
    phis = [0.6, 0.3, 0.1]
    kappas = [1, 0.8, 0.5]
    R0 = 1.5
    epsilon = 0.001

    def g():
        S_bars = [(S1 ** kappa) * ((1 - epsilon) ** (1 - kappa)) for kappa in kappas]
        X = 0
        for S_bar, phi, kappa in zip(S_bars, phis, kappas):
            X += kappa * R0 * phi * (S_bar - 1)
        return S1 - (1 - epsilon) * np.exp(X)

    step = 0.001
    S1_range = np.arange(step, (1 - epsilon) + step, step)
    gs = []
    for S1 in S1_range:
        gs.append(g())

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(S1_range, gs)
    ax1.axhline(0, color='grey', linestyle=':')
    plt.show()
    return


def main():
    # f2_final_size_plotter(phi1=0.2, beta=2 / 14, beta_ratio=0.7, gamma=1 / 14, epsilon=0.0001, plot=True)
    # f1_final_size_plotter(phi1=0.5, beta=2 / 14, beta_ratio=0.5, gamma=1 / 14, epsilon=0.0001, plot=True)
    interacting_binary()
    return


if __name__ == '__main__':
    main()
