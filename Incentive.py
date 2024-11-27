from TwoGroup import final_size_searcher_binary
from ConvexProgram import two_group_cvxpy, three_group_cvxpy
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

color_map = ['red', 'green', 'blue']


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


def read_three_group_Sbar(csv_name):
    df = pd.read_csv(csv_name)
    phi1s = df['phi1'].tolist()
    phi2s = df['phi2'].tolist()
    phi3s = df['phi3'].tolist()
    Sbar1s = df['Sbar1'].tolist()
    Sbar2s = df['Sbar2'].tolist()
    Sbar3s = df['Sbar3'].tolist()
    return phi1s, phi2s, phi3s, Sbar1s, Sbar2s, Sbar3s


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


def save_three_group_Sbar(csv_name, paras):
    """
    save the Sbar to csv
    """
    beta0, gamma, epsilon, k1, k2, k3, ps = paras
    betas = [
        beta0 * k1 * k1, beta0 * k1 * k2, beta0 * k1 * k3,
        beta0 * k2 * k1, beta0 * k2 * k2, beta0 * k2 * k3,
        beta0 * k3 * k1, beta0 * k3 * k2, beta0 * k3 * k3
    ]
    num_step = 100
    phi1s = []
    phi2s = []
    phi3s = []
    Ss = []
    for i in range(num_step + 1):
        for j in range(num_step + 1):
            if i + j > num_step:
                continue
            phi1 = i / num_step
            phi2 = j / num_step
            phi3 = (num_step - i - j) / num_step
            phi1s.append(phi1)
            phi2s.append(phi2)
            phi3s.append(phi3)
            S1, S2, S3 = three_group_cvxpy(betas, gamma, epsilon, phi1, phi2, phi3)
            Ss.append([S1, S2, S3])
    X0s = []
    for S, phi1, phi2, phi3 in zip(Ss, phi1s, phi2s, phi3s):
        S1, S2, S3 = S
        X0s.append(beta0 * (k1 * (S1 - phi1) + k2 * (S2 - phi2) + k3 * (S3 - phi3)))

    Sbar1s = [(1 - epsilon) * np.exp(k1 * X0) for X0 in X0s]
    Sbar2s = [(1 - epsilon) * np.exp(k2 * X0) for X0 in X0s]
    Sbar3s = [(1 - epsilon) * np.exp(k3 * X0) for X0 in X0s]
    df = pd.DataFrame(
        {'phi1': phi1s,
         'phi2': phi2s,
         'phi3': phi3s,
         'Sbar1': Sbar1s,
         'Sbar2': Sbar2s,
         'Sbar3': Sbar3s}
    )
    df.to_csv(csv_name, index=False)
    return


def plot_three_groups(csv_name, paras):
    beta0, gamma, epsilon, k1, k2, k3, ps = paras
    phi1s, phi2s, phi3s, Sbar1s, Sbar2s, Sbar3s = read_three_group_Sbar(csv_name)
    p1, p2, p3 = ps
    U1s = [p1 * Sbar1 for Sbar1 in Sbar1s]
    U2s = [p2 * Sbar2 for Sbar2 in Sbar2s]
    U3s = [p3 * Sbar3 for Sbar3 in Sbar3s]
    fig = plt.figure()
    ax1 = fig.add_subplot(projection='3d')
    ax1.plot_trisurf(phi1s, phi2s, U1s, color='red')
    ax1.plot_trisurf(phi1s, phi2s, U2s, color='green')
    ax1.plot_trisurf(phi1s, phi2s, U3s, color='blue')
    plt.show()


def plot_three_groups_incentive(csv_name, paras):
    beta0, gamma, epsilon, k1, k2, k3, ps = paras
    phi1s, phi2s, phi3s, Sbar1s, Sbar2s, Sbar3s = read_three_group_Sbar(csv_name)
    p1, p2, p3 = ps
    U1s = [p1 * Sbar1 for Sbar1 in Sbar1s]
    U2s = [p2 * Sbar2 for Sbar2 in Sbar2s]
    U3s = [p3 * Sbar3 for Sbar3 in Sbar3s]
    Us = zip(U1s, U2s, U3s)
    colors = [np.argmax(U) for U in Us]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    for cur_color in range(3):
        color_filter = [c == cur_color for c in colors]
        cur_phi1s = [phi for phi, c in zip(phi1s, color_filter) if c]
        cur_phi2s = [phi for phi, c in zip(phi2s, color_filter) if c]
        ax1.scatter(cur_phi1s, cur_phi2s, s=10, color=color_map[cur_color], label=f'Group {cur_color + 1}')
    ax1.legend()
    ax1.set_xlabel(r'$\phi_1$')
    ax1.set_ylabel(r'$\phi_2$')
    plt.show()


def three_group_incentive():
    path = 'incentive_3G'
    csv_name = f'{path}/Sbar.csv'
    if not os.path.exists(path):
        os.makedirs(path)
    beta0 = 3 / 7
    gamma = 1 / 7
    epsilon = 0.0001
    k1 = 1
    k2 = 0.8
    k3 = 0.7

    ps = [1, 0.95, 0.918]
    paras = [beta0, gamma, epsilon, k1, k2, k3, ps]

    # save_three_group_Sbar(csv_name, paras)
    # plot_three_groups(csv_name, paras)
    plot_three_groups_incentive(csv_name, paras)
    return


def main():
    # two_group_incentive()
    three_group_incentive()
    return


if __name__ == '__main__':
    main()
