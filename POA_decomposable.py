import matplotlib.pyplot as plt
import numpy as np


# function for X0 binary search over multiple groups
def f(X0, R0, phi, kappa, epsilon):
    ret = sum(kappa_i * phi_i * ((1 - epsilon) * np.exp(kappa_i * X0) - 1)
              for kappa_i, phi_i in zip(kappa, phi))
    ret *= R0
    ret -= X0
    return ret


# binary search X0 with f(X0)=0 over multiple groups
def X0_binary(R0, phi, kappa, epsilon, n_iterations):
    left, right = -R0, 0
    mid = (left + right) / 2
    f_mid = f(mid, R0, phi, kappa, epsilon)
    i = 0
    while f_mid != 0 and i < n_iterations:
        if f_mid > 0:
            left = mid
        else:
            right = mid
        mid = (left + right) / 2
        f_mid = f(mid, R0, phi, kappa, epsilon)
    return mid


def final_size_2group():
    p = [1, 0.8]
    R0 = 2
    kappa = [1, 0.1]
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
        X0s.append(X0_binary(R0, phi, kappa, epsilon, n_iterations=30))
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


def final_size_3group():
    p = [1, 0.8, 0.7]
    R0 = 2
    kappa = [1, 0.6, 0.3]
    epsilon = 0.001
    phi_step = 0.05
    phi1_range = np.arange(0, 1 + phi_step, phi_step)
    phi2_range = np.arange(0, 1 + phi_step, phi_step)
    X0s = []
    S1_bars = []
    S2_bars = []
    S3_bars = []
    U1s = []
    U2s = []
    U3s = []
    UG1s = []
    UG2s = []
    UG3s = []
    X, Y = [], []
    social = []
    for phi1 in phi1_range:
        for phi2 in phi2_range:
            if phi1 + phi2 > 1:
                break
            X.append(phi1)
            Y.append(phi2)
            phi3 = 1 - phi1 - phi2
            phi = [phi1, phi2, phi3]
            X0s.append(X0_binary(R0, phi, kappa, epsilon, n_iterations=30))
            S1_bars.append((1 - epsilon) * np.exp(kappa[0] * X0s[-1]))
            S2_bars.append((1 - epsilon) * np.exp(kappa[1] * X0s[-1]))
            S3_bars.append((1 - epsilon) * np.exp(kappa[2] * X0s[-1]))
            U1s.append(p[0] * S1_bars[-1])
            U2s.append(p[1] * S2_bars[-1])
            U3s.append(p[2] * S3_bars[-1])
            UG1s.append(phi1 * U1s[-1])
            UG2s.append(phi2 * U2s[-1])
            UG3s.append(phi3 * U3s[-1])
            social.append(UG1s[-1] + UG2s[-1] + UG3s[-1])
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot_trisurf(X, Y, U1s, label=r'$U_1$')
    ax1.plot_trisurf(X, Y, U2s, label=r'$U_2$')
    ax1.plot_trisurf(X, Y, U3s, label=r'$U_3$')
    ax1.legend()
    plt.show()
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot_trisurf(X, Y, UG1s, label=r'$UG_1$')
    ax1.plot_trisurf(X, Y, UG2s, label=r'$UG_2$')
    ax1.plot_trisurf(X, Y, UG3s, label=r'$UG_3$')
    ax1.plot_trisurf(X, Y, social, label='Social')
    ax1.legend()
    plt.show()
    return


def POA_alignment_2group(R0=10):
    kappa = [1, 0.01]
    epsilon = 0.001
    phi1_step = 0.01
    phi1_range = np.arange(0, 1 + phi1_step, phi1_step)
    X0s = []
    S1_bars = []
    S2_bars = []
    U1s = []
    U2s = []
    UG1s = []
    UG2s = []
    for phi1 in phi1_range:
        phi2 = 1 - phi1
        phi = [phi1, phi2]
        X0s.append(X0_binary(R0, phi, kappa, epsilon, n_iterations=30))
        S1_bars.append((1 - epsilon) * np.exp(kappa[0] * X0s[-1]))
        S2_bars.append((1 - epsilon) * np.exp(kappa[1] * X0s[-1]))
    p2 = S1_bars[-1] / S2_bars[-1]
    p = [1, p2]
    for i in range(len(phi1_range)):
        phi1 = phi1_range[i]
        phi2 = 1 - phi1
        U1s.append(p[0] * S1_bars[i])
        U2s.append(p[1] * S2_bars[i])
        UG1s.append(phi1 * U1s[i])
        UG2s.append(phi2 * U2s[i])
    social = [UG1 + UG2 for UG1, UG2 in zip(UG1s, UG2s)]
    POA = max(social) / social[-1]
    bound1 = np.exp(R0) / R0
    bound2 = np.exp(R0)
    return POA, bound1, bound2


def test_POA_against_bounds_2group():
    POAs = []
    bounds1 = []
    bounds2 = []
    R0_range = np.arange(1, 50, 1)
    for R0 in R0_range:
        POA, bound1, bound2 = POA_alignment_2group(R0)
        POA, bound1, bound2 = np.log(POA), np.log(bound1), np.log(bound2)
        POAs.append(POA)
        bounds1.append(bound1)
        bounds2.append(bound2)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(R0_range, POAs, label='POA')
    ax1.plot(R0_range, bounds1, label=r'$e^{R_0}/R_0$')
    ax1.plot(R0_range, bounds2, label=r'$e^{R_0}$')
    ax1.legend()
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
    f_values = [f(X0, R0, phi, kappa, epsilon) for X0 in X0_range]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(X0_range, f_values, label='test')
    ax1.axhline(0, linestyle=':', color='grey')
    # ax1.legend()
    plt.show()
    return


def main():
    # final_size_2group()
    final_size_3group()
    # test_POA_against_bounds_2group()


if __name__ == '__main__':
    main()
