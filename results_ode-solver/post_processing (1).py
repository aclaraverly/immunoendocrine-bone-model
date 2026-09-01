import matplotlib.pyplot as plt
import pandas as pd

path = "C:/Users/anacl/OneDrive/Área de Trabalho/Bone-estradiol-model-main/Bone-estradiol-model-main/output/"
# path = 'output/'


def plots(t, days, ys):

    ys1 = ys[:, 0]   # D
    ys2 = ys[:, 1]   # M0
    ys3 = ys[:, 2]   # M1
    ys4 = ys[:, 3]   # M2
    ys5 = ys[:, 4]   # c1
    ys6 = ys[:, 5]   # c2
    ys7 = ys[:, 6]   # Cm
    ys8 = ys[:, 7]   # Cb
    ys9 = ys[:, 8]   # Mc
    ys10 = ys[:, 9]  # Mb

    # ============================
    # Primeiro grupo de 4 gráficos
    # D, Macrophages, c1, c2
    # ============================

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Debris
    axs[0, 0].plot(t, ys1, 'r')
    axs[0, 0].set_xlim(0, days)
    axs[0, 0].set_xlabel('Time (days)')
    axs[0, 0].set_ylabel('Concentration (cells/ml)')
    axs[0, 0].set_title('Debris')
    axs[0, 0].legend(["D"])
    axs[0, 0].grid(True)

    # Macrophages
    axs[0, 1].plot(t, ys2, '--k', t, ys3, 'b', t, ys4, 'g')
    axs[0, 1].set_xlim(0, days)
    axs[0, 1].set_xlabel('Time (days)')
    axs[0, 1].set_ylabel('Concentration (cells/ml)')
    axs[0, 1].set_title('Macrophages')
    axs[0, 1].legend(["M0", "M1", "M2"])
    axs[0, 1].grid(True)

    # Pro-inflammatory cytokines
    axs[1, 0].plot(t, ys5, 'g')
    axs[1, 0].set_xlim(0, days)
    axs[1, 0].set_xlabel('Time (days)')
    axs[1, 0].set_ylabel('Concentration (ng/ml)')
    axs[1, 0].set_title('Pro-inflammatory cytokines')
    axs[1, 0].legend(["c1"])
    axs[1, 0].grid(True)

    # Anti-inflammatory cytokines
    axs[1, 1].plot(t, ys6, 'c')
    axs[1, 1].set_xlim(0, days)
    axs[1, 1].set_xlabel('Time (days)')
    axs[1, 1].set_ylabel('Concentration (ng/ml)')
    axs[1, 1].set_title('Anti-inflammatory cytokines')
    axs[1, 1].legend(["c2"])
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}grupo_1_D_macrophages_c1_c2.png', dpi=300)
    print(f'{path}grupo_1_D_macrophages_c1_c2.png')
    plt.show()

    # ============================
    # Segundo grupo de 4 gráficos
    # Cm, Cb, Mc, Mb
    # ============================

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Mesenchymal stem cells
    axs[0, 0].plot(t, ys7, 'b')
    axs[0, 0].set_xlim(0, days)
    axs[0, 0].set_xlabel('Time (days)')
    axs[0, 0].set_ylabel('Concentration (cells/ml)')
    axs[0, 0].set_title('Mesenchymal stem cells')
    axs[0, 0].legend(["Cm"])
    axs[0, 0].grid(True)

    # Osteoblasts
    axs[0, 1].plot(t, ys8, 'y')
    axs[0, 1].set_xlim(0, days)
    axs[0, 1].set_xlabel('Time (days)')
    axs[0, 1].set_ylabel('Concentration (cells/ml)')
    axs[0, 1].set_title('Osteoblasts')
    axs[0, 1].legend(["Cb"])
    axs[0, 1].grid(True)

    # Cartilage
    axs[1, 0].plot(t, ys9, 'g')
    axs[1, 0].set_xlim(0, days)
    axs[1, 0].set_xlabel('Time (days)')
    axs[1, 0].set_ylabel('Concentration (g/ml)')
    axs[1, 0].set_title('Cartilage')
    axs[1, 0].legend(["Mc"])
    axs[1, 0].grid(True)

    # Bone formation
    axs[1, 1].plot(t, ys10, 'r')
    axs[1, 1].set_xlim(0, days)
    axs[1, 1].set_xlabel('Time (days)')
    axs[1, 1].set_ylabel('Concentration (g/ml)')
    axs[1, 1].set_title('Bone formation')
    axs[1, 1].legend(["Mb"])
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}grupo_2_Cm_Cb_Mc_Mb.png', dpi=300)
    print(f'{path}grupo_2_Cm_Cb_Mc_Mb.png')
    plt.show()


def plot_estradiol(t, days, ys):

    ys11 = ys[:, 10]  # E2

    plt.figure(figsize=(8, 5))
    plt.plot(t, ys11, 'm', linewidth=2)
    plt.xlim(0, days)
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (ng/ml)')
    plt.title('Estradiol')
    plt.legend(['E2'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path}E2.png', dpi=300)
    print(f'{path}E2.png')
    plt.show()