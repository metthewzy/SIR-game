from TwoGroup import final_size_searcher_binary
from ConvexProgram import two_group_cvxpy
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def two_group_NE_finder(U1s, U2s):
    """
    return phi1 value at NE
    """
    for i in range(len(U1s)):
        if U1s[i] <= U2s[i]:
            return i
    return len(U1s) - 1


def two_group_incentive():
    path = 'incentive'
    csv_name = f'{path}/Sbar.csv'
    if not os.path.exists(path):
        os.makedirs(path)

    # save_two_group_Sbar(csv_name)

    # plot_two_group(csv_name)

    plot_two_group_incentive(csv_name)
    return


def plot_two_group_incentive(csv_name):
    phi1s, Sbar1s, Sbar2s = read_two_group_Sbar(csv_name)
    p1 = 1
    U1s = [p1 * Sbar1 for Sbar1 in Sbar1s]
    p2_orig = 0.9
    phi1_NEs = []
    Bs = []
    socials = []
    UG1s = []
    UG2s = []
    delta_step = 0.0005
    deltas = np.arange(0, p1 - p2_orig + delta_step * 3, delta_step)
    for delta in deltas:
        p2 = p2_orig + delta
        U2s = [p2 * Sbar2 for Sbar2 in Sbar2s]
        idx_NE = two_group_NE_finder(U1s, U2s)
        phi1_NE = phi1s[idx_NE]
        phi2_NE = 1 - phi1_NE
        phi1_NEs.append(phi1_NE)
        UG1s.append(phi1_NE * U1s[idx_NE])
        UG2s.append(phi2_NE * U2s[idx_NE])
        socials.append(phi1_NE * U1s[idx_NE] + phi2_NE * U2s[idx_NE])
        Bs.append(phi2_NE * delta)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(deltas, socials, label='Social')
    # ax1.plot(deltas, Bs, label='Budget')
    ax1.plot(deltas, phi1_NEs, label=r'$\phi_1$')
    ax1.plot(deltas, [social - B for social, B in zip(socials, Bs)], label='Net social')
    ax1.plot(deltas, UG1s, label='UG1')
    ax1.plot(deltas, UG2s, label='UG2')
    ax1.axvline(p1 - p2_orig, linestyle=':')
    ax1.set_xlabel(r'$\Delta$')
    ax1.set_ylabel('Social welfare')
    ax1.legend()
    plt.show()

    return


def plot_two_group(csv_name):
    phi1s, Sbar1s, Sbar2s = read_two_group_Sbar(csv_name)
    p1 = 1
    p2 = 0.95
    U1s = [p1 * Sbar1 for Sbar1 in Sbar1s]
    U2s = [p2 * Sbar2 for Sbar2 in Sbar2s]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(phi1s, U1s, label=r'$U_1$')
    ax1.plot(phi1s, U2s, label=r'$U_2$')
    ax1.set_xlabel(r'$\phi_1$')
    ax1.set_ylabel('Individual Utility')
    ax1.legend()
    plt.show()


def read_two_group_Sbar(csv_name):
    df = pd.read_csv(csv_name)
    phi1s = df['phi1'].tolist()
    Sbar1s = df['Sbar1'].tolist()
    Sbar2s = df['Sbar2'].tolist()
    return phi1s, Sbar1s, Sbar2s


def save_two_group_Sbar(csv_name):
    """
    save the Sbar to csv
    """
    beta0 = 2 / 7
    gamma = 1 / 7
    epsilon = 0.0001
    k1 = 1
    k2 = 0.6
    betas = [beta0 * k1 * k1,
             beta0 * k1 * k2,
             beta0 * k2 * k1,
             beta0 * k2 * k2]
    phi1_step = 0.01
    phi1s = np.arange(0, 1 + phi1_step, phi1_step)
    phi2s = [1 - phi1 for phi1 in phi1s]
    S1s, S2s = zip(*[two_group_cvxpy(betas, gamma, epsilon, phi1) for phi1 in phi1s])
    X0s = [
        beta0 * (k1 * (S1 - phi1) + k2 * (S2 - phi2)) for S1, S2, phi1, phi2 in zip(S1s, S2s, phi1s, phi2s)
    ]
    Sbar1s = [(1 - epsilon) * np.exp(k1 * X0) for X0 in X0s]
    Sbar2s = [(1 - epsilon) * np.exp(k2 * X0) for X0 in X0s]
    df = pd.DataFrame(
        {'phi1': phi1s,
         'Sbar1': Sbar1s,
         'Sbar2': Sbar2s}
    )
    df.to_csv(csv_name, index=False)
    return


def main():
    two_group_incentive()
    return


if __name__ == '__main__':
    main()
