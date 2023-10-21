import matplotlib.pyplot as plt
import numpy as np
import random

epsilon = 0.0001


def F1(X, beta, gamma, phi):
    X1, X2 = X
    b11, b12, b21, b22 = beta
    phi1, phi2 = phi
    res = (b11 / gamma * phi1 * ((1 - epsilon) * np.exp(X1) - 1)
           + b12 / gamma * phi2 * ((1 - epsilon) * np.exp(X2) - 1)
           - X1)
    return res


def F2(X, beta, gamma, phi):
    X1, X2 = X
    b11, b12, b21, b22 = beta
    phi1, phi2 = phi
    res = (b21 / gamma * phi1 * ((1 - epsilon) * np.exp(X1) - 1)
           + b22 / gamma * phi2 * ((1 - epsilon) * np.exp(X2) - 1)
           - X2)
    return res


def feasible_plotter():
    beta = [random.random() for _ in range(4)]
    beta = [b / max(beta) * 0.6 for b in beta]
    beta = [0.02768752084575399, 0.536165900406655, 0.6, 0.1569916097724648]
    k1, k2 = 0.3, 0.3
    # beta = [k1 * k1, k1 * k2, k2 * k1, k2 * k2]
    print('beta=', beta)
    gamma = 1 / 14
    phi1 = 0.6
    phi = [phi1, 1 - phi1]
    X_steps = 50
    fig = plt.figure()
    ax1 = fig.add_subplot()

    # binary search for X2
    X1_range = np.arange(-2, 0 + 2 / X_steps, 2 / X_steps)
    X2_curve = []
    for X1 in X1_range:
        l, r = -2, 0
        F_l = F1([X1, l], beta, gamma, phi)
        F_r = F1([X1, r], beta, gamma, phi)
        for _ in range(50):
            m = (l + r) / 2
            F_m = F1([X1, m], beta, gamma, phi)
            if F_l * F_m >= 0:
                l = m
                F_l = F_m
            else:
                r = m
                F_r = F_m
        X2_curve.append(m)

    # binary search for X1
    X2_range = np.arange(-2, 0 + 2 / X_steps, 2 / X_steps)
    X1_curve = []
    for X2 in X2_range:
        l, r = -2, 0
        F_l = F2([l, X2], beta, gamma, phi)
        F_r = F2([r, X2], beta, gamma, phi)
        for _ in range(50):
            m = (l + r) / 2
            F_m = F2([m, X2], beta, gamma, phi)
            if F_l * F_m >= 0:
                l = m
                F_l = F_m
            else:
                r = m
                F_r = F_m
        X1_curve.append(m)

    ax1.plot(X1_range, X2_curve, label='F1')
    ax1.plot(X1_curve, X2_range, label='F2')
    ax1.legend()
    plt.show()
    return


def F1F2_trisurf():
    beta = [random.random() for _ in range(4)]
    beta = [b / max(beta) * 0.6 for b in beta]
    beta = [0.02768752084575399, 0.536165900406655, 0.6, 0.1569916097724648]
    k1, k2 = 0.3, 0.3
    # beta = [k1 * k1, k1 * k2, k2 * k1, k2 * k2]
    print('beta=', beta)
    gamma = 1 / 14
    phi1 = 0.6
    phi = [phi1, 1 - phi1]
    X_steps = 30
    fig = plt.figure()
    ax1 = fig.add_subplot(projection ='3d')
    X1_range = np.arange(-2, 0 + 2 / X_steps, 2 / X_steps)
    X2_range = np.arange(-2, 0 + 2 / X_steps, 2 / X_steps)
    X1s = []
    X2s = []
    F1s = []
    F2s = []
    for X1 in X1_range:
        for X2 in X2_range:
            X1s.append(X1)
            X2s.append(X2)
            F1s.append(F1([X1, X2], beta, gamma, phi))
            F2s.append(F2([X1, X2], beta, gamma, phi))
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.plot_trisurf(X1s, X2s, F1s, color='blue')
    ax1.plot_trisurf(X1s, X2s, F2s, color='green')
    plt.show()
    return


def main():
    # feasible_plotter()
    F1F2_trisurf()
    return


if __name__ == '__main__':
    main()
