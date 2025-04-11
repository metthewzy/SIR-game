import numpy as np


def reduce(support, beta, gamma, epsilon, p, N, n_groups):
    """
    try finding a Nash for the current support set
    """
    NE = []
    phi = [0] * n_groups
    for i in range(n_groups):
        bit = 1 << i
        if bit & support:
            NE.append(i)
    print(f'\ntrying groups {NE}:')
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
            print(f"group {NE[i]} is negative")
            return False, [], 0
        phi[NE[i]] = X[i]
    RHS = X[-1]
    print('solution:', X)
    for i in range(n_groups):
        row = A_full[i]
        cur_sum = sum(
            [a * x for a, x in zip(row, X[:-1])]
        )
        if cur_sum > RHS:
            # print(f'group {i} is better', cur_sum, '>', RHS)
            print(f'group {i} is better')
            return False, [], 0
    # print("found!")
    return True, phi, RHS


def three_groups():
    n_groups = 3
    beta = [
        [1, 2, 3],
        [3, 2, 1],
        [1, 4, 5]
    ]
    gamma = 2
    epsilon = 0.01
    p = [0.5, 1, 0.8]
    Ns = [0.5 * max(p)]
    for N in Ns:
        for support in range(1, 2 ** n_groups):
            res, phi, ln_N = reduce(support, beta, gamma, epsilon, p, N, n_groups)
            if res:
                print('******************** FOUND !!! ************************')
                print(f'phi = {phi}')
                print('*******************************************************')
                # break
    return


def main():
    three_groups()
    return


if __name__ == '__main__':
    main()
