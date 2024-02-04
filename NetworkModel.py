import numpy as np
import matplotlib.pyplot as plt


def RHS(X0, alphas, R0, epsilon):
    res = 0
    for alpha in alphas:
        res += alpha * R0 * ((1 - epsilon) * np.exp(alpha * X0) - 1)
    return res


def X0_plot():
    beta = 2
    gamma = 1 / 7
    R0 = beta / gamma
    epsilon = 0.01
    """
    equation : 
    (1) X0 = sum_v alpha_v * kappa_1 * R0 * phi_1^v * (Sbar_1^v - 1)
    (2) Sbar_1^v = (1 - epsilon) * e^(alpha_v * kappa_1 * X0)
    ==>
    X0 = sum_v alpha_v * R0  * ((1 - epsilon) * e^(alpha_v * X0) - 1)
    LHS: X0, -R0*(sum_v alpha_v) <= X0 <= 0
    RHS: summation as a function of X0 and other parameters
    """
    alphas = [1, 0.95, 0.9, 0.8, 0.5]
    X0_slices = 100
    X0_lower = -R0 * sum(alphas)
    X0_upper = X0_lower * .99999
    X0_upper = 0
    X0_range = np.arange(X0_lower, X0_upper, (X0_upper - X0_lower) / X0_slices)
    func = []
    for X0 in X0_range:
        func.append(X0 - RHS(X0, alphas, R0, epsilon))
    print(func[0])
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(X0_range, func)
    ax1.set_xlabel(r'$X_0$')
    ax1.axhline(0, linestyle=':')
    plt.show()
    return


def main():
    X0_plot()
    return


if __name__ == '__main__':
    main()
