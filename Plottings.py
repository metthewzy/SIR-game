import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from ConvexProgram import phi_step
import pandas as pd
from matplotlib.patches import Rectangle

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
    fig1 = plt.figure(figsize=(6, 4.8))
    fig2 = plt.figure(figsize=(6, 4.8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    IU1_plotted = False
    for d in data[-3::-2]:
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
        ax1.plot(phi1_range, social, label=fr"$\eta$={PaymentRatio}")
        # ax2.plot(phi1_range, IU1, color=colorS1)  # label=f"Ind utility 1={PaymentRatio}")
        # ax2.plot(phi1_range, IU2, color=colorS2, label=f"Ind utility 1={PaymentRatio}")
        # ax2.plot(nash_point, nash_value, 'ro', markersize=10, label=f"NE @ {PaymentRatio}")
        if not IU1_plotted:
            IU1_plotted = True
            ax2.plot(phi1_range, IU1, label="Group 1")  # label=f"Ind utility 1={PaymentRatio}")
        ax2.plot(phi1_range, IU2, label=rf"Group 2, $\eta$={PaymentRatio}")
        ax2.plot(nash_point, nash_value, 'ro', markersize=10)
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    ax1.legend()
    ax1.set_title("Social\n"rf"$\beta$={beta}, $\kappa$={kappa}, $\gamma$={gamma}", fontsize=16)
    ax1.set_xlabel(r"$\phi_1$")
    ax1.set_ylabel("Social")
    fig1.tight_layout()
    fig1.savefig("figCvx/fig1.png")
    ax2.legend()
    ax2.set_title("Individual Utility and Nash\n"rf"$\beta$={beta}, $\kappa$={kappa}, $\gamma$={gamma}", fontsize=16)
    ax2.set_xlabel(r"$\phi_1$")
    ax2.set_ylabel("Individual utility")
    fig2.tight_layout()
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
    target_beta = beta_list[3]
    gamma = 1 / 10
    R0 = target_beta / gamma
    print("target beta:", target_beta)
    data = [d for d in json_data if d["beta"] == target_beta]
    kappa_set = set([d["kappa"] for d in data])
    kappa_list = sorted(kappa_set)
    n = len(kappa_list)
    paymentRatio_set = set([d["paymentRatio"] for d in data])
    paymentRatio_list = sorted(paymentRatio_set)
    m = len(paymentRatio_list)
    table = np.zeros((m, n))
    max_POA = 0
    max_i, max_j = float('inf'), float('inf')
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
        if POA > max_POA:
            max_j = j
            max_i = i
            max_POA = POA
        table[i, j] = POA
    df = pd.DataFrame(table, index=kappa_list, columns=kappa_list)
    ax = sns.heatmap(df, cmap="YlOrRd")
    ax.invert_yaxis()
    ax.add_patch(Rectangle(
        (max_j, max_i),
        1,
        1,
        fill=False,
        edgecolor="blue",
        # linestyle="--",
        lw=2,
        clip_on=False
    ))
    ax.set_title("POA\n"rf"$\beta_0={target_beta}$", fontsize=16)
    fig = ax.get_figure()
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$\eta$")
    plt.tight_layout()
    fig.savefig("figCvx/fig3.png")
    print(f"max POA={max_POA} at kappa={kappa_list[max_j]}, eta={paymentRatio_list[max_i]}")
    print("POA bound=", np.exp(R0) / R0)



def main():
    # figure1_and_2()
    figure3()


if __name__ == '__main__':
    main()
