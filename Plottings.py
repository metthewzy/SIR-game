import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from ConvexProgram import phi_step
import pandas as pd
phi1_range = np.arange(phi_step, 1, phi_step)


def figure1_and_2():
    json_filename = "figCvx/SimWithPara.json"
    with open(json_filename, "r") as file:
        json_data = json.load(file)

    # use offset to select the 3 scenarios
    idx_offset = 0
    # data = json_data[idx_offset: idx_offset + 3]
    beta_list = [0.2, 0.225, 0.25, 0.275]
    target_beta = beta_list[2]
    target_kappa = 0.1
    data = [d for d in json_data if (d["beta"] == target_beta) & (d["kappa"] == target_kappa)]
    beta = data[0]["beta"]
    kappa = data[0]["kappa"]
    gamma = data[0]["gamma"]
    colorS1 = 'lightgreen'
    colorS2 = 'lightblue'
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    IU1_plotted = False
    for d in data[::-1]:
        UG1 = d["GroupUtility1"]
        UG2 = d["GroupUtility2"]
        social = [U1 + U2 for U1, U2 in zip(UG1, UG2)]
        PaymentRatio = d["paymentRatio"]
        IU1 = d["IndUtility1"]
        IU2 = d["IndUtility2"]
        nash_point = d["PhiNash"]
        nash_value = IU1[d["NashIndex"]]
        # print(type(PaymentRatio))
        # print(len(social))
        ax1.plot(phi1_range, social, label=f"payment ratio={PaymentRatio}")
        # ax2.plot(phi1_range, IU1, color=colorS1)  # label=f"Ind utility 1={PaymentRatio}")
        # ax2.plot(phi1_range, IU2, color=colorS2, label=f"Ind utility 1={PaymentRatio}")
        # ax2.plot(nash_point, nash_value, 'ro', markersize=10, label=f"NE @ {PaymentRatio}")
        if not IU1_plotted:
            IU1_plotted = True
            ax2.plot(phi1_range, IU1, label="Group 1")  # label=f"Ind utility 1={PaymentRatio}")
        ax2.plot(phi1_range, IU2, label=f"Group 2 @{PaymentRatio}")
        ax2.plot(nash_point, nash_value, 'ro', markersize=10)

    ax1.legend()
    ax1.set_title(f"Social\nbeta={beta}, kappa={kappa}, gamma={gamma}")
    ax1.set_xlabel(r"$\phi_1$")
    ax1.set_ylabel("Social")
    fig1.savefig("figCvx/fig1.png")
    ax2.legend()
    ax2.set_title(f"Individual utility and Nash\nbeta={beta}, kappa={kappa}, gamma={gamma}")
    ax2.set_xlabel(r"$\phi_1$")
    ax2.set_ylabel("Individual utility")
    fig2.savefig("figCvx/fig2.png")
    plt.close(fig1)
    plt.close(fig2)
    return


def figure3():
    json_filename = "figCvx/SimWithPara.json"
    with open(json_filename, "r") as file:
        json_data = json.load(file)
    print("number of configurations:", len(json_data))
    print("fileds:", list(json_data[0].keys()))
    beta_list = [0.2, 0.225, 0.25, 0.275]
    target_beta = beta_list[2]
    print("target beta:", target_beta)
    data = [d for d in json_data if d["beta"] == target_beta]
    kappa_set = set([d["kappa"] for d in data])
    kappa_list = sorted(kappa_set)
    n = len(kappa_list)
    paymentRatio_set = set([d["paymentRatio"] for d in data])
    paymentRatio_list = sorted(paymentRatio_set)
    m = len(paymentRatio_list)
    table = np.zeros((m, n))
    for d in data:
        kappa = d["kappa"]
        j = kappa_list.index(kappa)
        paymentRatio = d["paymentRatio"]
        i = paymentRatio_list.index(paymentRatio)
        OptIndex = d["OptIndex"]
        NashIndex = d["NashIndex"]
        UG1 = d["GroupUtility1"]
        UG2 = d["GroupUtility2"]
        social_OPT = UG1[OptIndex] + UG2[OptIndex]
        social_NASH = UG1[NashIndex] + UG2[NashIndex]
        POA = social_OPT / social_NASH
        table[i, j] = POA
    df = pd.DataFrame(table, index=kappa_list, columns=kappa_list)
    ax = sns.heatmap(df, cmap="YlOrRd")
    ax.invert_yaxis()
    ax.set_title(fr"POA  $\beta={target_beta}$")
    fig = ax.get_figure()
    plt.xlabel(r"$\kappa$")
    plt.ylabel("payment ratio")
    fig.savefig("figCvx/fig3.png")


def main():
    figure1_and_2()
    # figure3()


if __name__ == '__main__':
    main()
