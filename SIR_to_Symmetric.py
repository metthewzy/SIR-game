import numpy as np


def reduce(support, beta, gamma, epsilon, p, N, n_groups):
    NE = []
    not_NE = []
    for i in range(n_groups):
        bit = 1 << i
        if bit & support:
            NE.append(i)
        else:
            not_NE.append(i)
    print("NE:", NE)
    print("not NE:", not_NE)
    B = []
    return True, []


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
        phi = None
        for support in range(1, 2 ** n_groups):
            res, phi = reduce(support, beta, gamma, epsilon, p, N, n_groups)
            # if res:
            #     break
    return


def main():
    three_groups()
    return


if __name__ == '__main__':
    main()
