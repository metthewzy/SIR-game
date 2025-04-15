import numpy as np
import matplotlib.pyplot as plt
import random


def try_support_set(support, beta, gamma, epsilon, p, N, n_groups):
    """
    try finding a Nash for the current support set
    """
    NE = []
    phi = [0] * n_groups
    for i in range(n_groups):
        bit = 1 << i
        if bit & support:
            NE.append(i)
    # print(f'\ntrying groups {NE}:')
    A = []
    A_full = []
    B = []
    for i in range(n_groups):
        row = [
            beta[i][j] / gamma * (N / p[j] - 1) +
            np.log((1 - epsilon) * p[i])
            for j in NE
        ]
        A_full.append(row)
        if i in NE:
            A.append(row + [-1])
            B.append(0)
    A.append(
        [1] * len(NE) + [0]
    )
    B.append(1)

    # for row, b in zip(A, B):
    #     print(f'{row} : {b}')

    try:
        X = np.linalg.solve(A, B)
    except Exception as e:
        print('solver exception', e)
        return False, [], 0
    for i in range(len(NE)):
        if X[i] < 0:
            # print(f"group {NE[i]} is negative")
            return False, [], 0
        phi[NE[i]] = X[i]
    RHS = X[-1]
    # print('solution:', X)
    for i in range(n_groups):
        if i in NE:
            continue
        row = A_full[i]
        cur_sum = sum(
            [a * x for a, x in zip(row, X[:-1])]
        )
        if cur_sum > RHS:
            # print(f'group {i} is better', cur_sum, '>', RHS)
            # print(f'group {i} is better')
            return False, [], 0
    # print("found!")
    return True, phi, RHS


def three_groups():
    n_groups = 3
    # beta_low, beta_high = 0.5, 3
    # beta = [random.uniform(beta_low, beta_high) for _ in range(n_groups * n_groups)]
    # beta = np.array(beta).reshape(n_groups, n_groups)
    beta = [[1.7725334, 1.65980034, 1.61748513],
            [2.2426902, 2.23867483, 0.95699598],
            [2.68435766, 1.8605644, 2.29758875]]
    gamma = 0.8 * max(
        [max(b) for b in beta]
    )
    epsilon = 0.01
    # p = [random.random() for _ in range(n_groups)]
    p = [0.6820323745556307, 0.5695666965086351, 0.7497492322733994]
    p_max = max(p)
    step = 0.01
    Ns = np.arange(step, 1, step)
    Ns = [N * p_max for N in Ns]
    Us = []
    print('beta:')
    [print(b) for b in beta]
    print(f'gamma:\n{gamma}')
    print(f'p:\n{p}')
    previous_support = 0
    changes = []
    for N in Ns:
        found = False
        for delta in range(2 ** n_groups):
            support = (previous_support + delta) % (2 ** n_groups)
            if support == 0:
                continue
            res, phi, U = try_support_set(support, beta, gamma, epsilon, p, N, n_groups)
            if res:
                # print('******************** FOUND !!! ************************')
                if support != previous_support:
                    print('support set changed at N =', N)
                    changes.append(N)
                    print(f'phi = {phi}')
                # print(f'phi = {phi}')
                # print('*******************************************************')
                Us.append(np.exp(U))
                found = True
                previous_support = support
                break
        if not found:
            print('**************** NOT FOUND !!! ************************')
            print(f'N = {N}')
            print('*******************************************************')
            Us.append(0)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(Ns, Us)
    ax1.plot([min(Ns + Us), max(Ns + Us)], [min(Ns + Us), max(Ns + Us)], color='orange')
    [ax1.axvline(change, linestyle=':', color='grey') for change in changes]
    plt.show()
    return


def main():
    three_groups()
    return


if __name__ == '__main__':
    main()
