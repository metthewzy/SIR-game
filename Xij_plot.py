import matplotlib.pyplot as plt
from matplotlib import patches


def axis_plotter():
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot()
    ax1.axhline(0, c='k')
    rec1 = patches.Rectangle((0, 0), 10, 1, facecolor="green", alpha=0.3)
    rec2 = patches.Rectangle((0, 0), -10, 1, facecolor="blue", alpha=0.3)
    ax1.scatter(0, 0, color='r', zorder=10)
    ax1.add_patch(rec1)
    ax1.add_patch(rec2)
    ax1.annotate(r'$X_{i,j}^v$', xy=(0, -0.5), horizontalalignment='center')
    ax1.annotate(r'$U_i^v < U_j^v$', xy=(-5, .5), horizontalalignment='center', verticalalignment='center')
    ax1.annotate(r'$U_i^v > U_j^v$', xy=(5, .5), horizontalalignment='center', verticalalignment='center')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-1, 2)
    plt.axis('off')
    # plt.show()
    fig.savefig('X_0_axis.png', bbox_inches='tight')


def main():
    axis_plotter()
    return


if __name__ == '__main__':
    main()
