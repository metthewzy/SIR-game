import matplotlib.pyplot as plt
import json
import numpy as np

from ConvexProgram import phi_step

phi1_range = np.arange(phi_step, 1, phi_step)


def figure1():
    json_filename = "figCvx/SimWithPara.json"
    with open(json_filename, "r") as file:
        json_data = json.load(file)

    # use offset to select the 3 scenarios
    idx_offset = 0
    data = json_data[idx_offset: idx_offset + 3]

    beta = data[0]["beta"]
    kappa = data[0]["kappa"]
    gamma = data[0]["gamma"]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    for d in data:
        UG1 = d["GroupUtility1"]
        UG2 = d["GroupUtility2"]
        social = [U1 + U2 for U1, U2 in zip(UG1, UG2)]
        PaymentRatio = d["paymentRatio"]
        # print(type(PaymentRatio))
        # print(len(social))
        ax1.plot(phi1_range, social, label=f"payment ratio={PaymentRatio}")
    ax1.legend()
    ax1.set_title(f"Figure 1\nbeta={beta}, kappa={kappa}, gamma={gamma}")
    ax1.set_xlabel(r"$\phi_1$")
    ax1.set_ylabel("Social")
    fig.savefig("figCvx/fig1.png")
    return


def main():
    figure1()


if __name__ == '__main__':
    main()
