import numpy as np


def reduce(support, beta, gamma, epsilon, p, N, n_groups):
    """
    try finding a Nash for the current support set
    """
    NE = []
    # not_NE = []
    phi = [0] * n_groups
    for i in range(n_groups):
        bit = 1 << i
        if bit & support:
            NE.append(i)
        # else:
        #     not_NE.append(i)
    print('trying ', NE)
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
            A.append(row)
            B.append(np.log(N))
    for row, b in zip(A,B):
        print(f'{row} : {b}')
    # A.append(
    #     [1] * len(NE)
    # )
    # B.append(1)
    try:
        X = np.linalg.solve(A, B)
    except Exception:
        print('solver exception')
        return False, []
    for i in range(len(NE)):
        phi[NE[i]] = X[i]
    if sum(phi) != 1:
        print('not summing to 1', X)
        return False, []

    for i in range(n_groups):
        row = A_full[i]
        if sum(
            [a * x for a, x in zip(row, phi)]
        ) > np.log(N):
            print(f'group {i} is better')
            return False, []
    print("found!")
    return True, phi


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
            res, phi = reduce(support, beta, gamma, epsilon, p, N, n_groups)
            if res:
                print("FOUND!!!!!!!!", phi)
                break
    return


def main():
    three_groups()
    return


if __name__ == '__main__':
    main()
