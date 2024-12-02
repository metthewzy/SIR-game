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
    csv_name, paras = two_group_parameter_scenario(1)  # 0: low infection; 1: high infection
    # save_two_group_Sbar(csv_name, paras)
    plot_two_group(csv_name, paras)
    plot_two_group_incentive(csv_name, paras)
    return


def two_group_parameter_scenario(scenario=0):
    """
    scenario:   0: low R0, no peak when phi1 low
                1: high R0, always peak
    """
    if scenario:
        path = 'incentive'
        csv_name = f'{path}/Sbar2.csv'
        if not os.path.exists(path):
            os.makedirs(path)
        beta0 = 2 / 7
        gamma = 1 / 7
        epsilon = 0.0001
        k1 = 1
        k2 = 0.8
        ps = [1, 0.97]
    else:
        path = 'incentive'
        csv_name = f'{path}/Sbar.csv'
        if not os.path.exists(path):
            os.makedirs(path)
        beta0 = 2 / 7
        gamma = 1 / 7
        epsilon = 0.0001
        k1 = 1
        k2 = 0.6
        ps = [1, 0.95]
    paras = [beta0, gamma, epsilon, k1, k2, ps]
    return csv_name, paras


def two_group_Delta_finder(csv_name, paras):
    """
    boost the lower group to make the current phi a Nash equilibrium
    """
    beta0, gamma, epsilon, k1, k2, ps = paras
    phi1s, Sbar1s, Sbar2s = read_two_group_Sbar(csv_name)
    p1, p2 = ps
    Delta1s = []
    Delta2s = []
    Bs = []
    for phi1, Sbar1, Sbar2 in zip(phi1s, Sbar1s, Sbar2s):
        phi2 = 1 - phi1
        if p2 * Sbar2 >= p1 * Sbar1:
            Delta1 = p2 * Sbar2 / Sbar1 - p1
            Delta2 = 0
        else:
            Delta1 = 0
            Delta2 = p1 * Sbar1 / Sbar2 - p2
        Delta1s.append(Delta1)
        Delta2s.append(Delta2)
        Bs.append(phi1 * Delta1 + phi2 * Delta2)
    return Delta1s, Delta2s, Bs


def plot_two_group_incentive(csv_name, paras):
    beta0, gamma, epsilon, k1, k2, ps = paras
    phi1s, Sbar1s, Sbar2s = read_two_group_Sbar(csv_name)
    p1, p2 = ps
    socials = [p1 * phi1 * Sbar1 + p2 * (1 - phi1) * Sbar2 for phi1, Sbar1, Sbar2 in zip(phi1s, Sbar1s, Sbar2s)]
    Delta1s, Delta2s, Bs = two_group_Delta_finder(csv_name, paras)
    new_p1s = [p1 + Delta1 for Delta1 in Delta1s]
    new_p2s = [p2 + Delta2 for Delta2 in Delta2s]
    NEs = [max(p1 * Sbar1, p2 * Sbar2) for phi1, Sbar1, Sbar2 in zip(phi1s, Sbar1s, Sbar2s)]

    # metric1: Social (new NE) / Budget
    metric1 = [0 if B == 0 else NE / B for NE, B in zip(NEs, Bs)]
    # metric2: Social improvement / Budget
    metric2 = [0 if B == 0 else (NE - social) / B for NE, social, B in zip(NEs, socials, Bs)]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(phi1s, metric1, label='metric 1')
    ax1.set_xlabel(r'$\phi_1$')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(phi1s, metric2, label='metric 2')
    ax2.set_xlabel(r'$\phi_1$')
    ax2.legend()

    plt.show()
    return


def plot_two_group(csv_name, paras):
    beta0, gamma, epsilon, k1, k2, ps = paras
    phi1s, Sbar1s, Sbar2s = read_two_group_Sbar(csv_name)
    p1, p2 = ps
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


def save_two_group_Sbar(csv_name, paras):
    """
    save the Sbar to csv
    """
    beta0, gamma, epsilon, k1, k2, ps = paras
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
    two_group_incentive()
    # three_group_incentive()
    return


if __name__ == '__main__':
    main()
