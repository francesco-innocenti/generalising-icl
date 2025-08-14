import numpy as np
import matplotlib.pyplot as plt


def plot_ΔWs_alignments(ΔWs_alignments, save_path):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = plt.imshow(
        ΔWs_alignments, 
        vmin=-1, 
        vmax=1, 
        cmap="coolwarm", 
        origin="lower"
    )
    cbar = fig.colorbar(im)
    cbar.set_ticks([1, 0, -1])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(
        "Directional alignment", 
        fontsize=16, 
        labelpad=10
    )
    ax.set_title("$t = 0$", fontsize=18, pad=10)
    ax.set_ylabel("$\Delta W_i(C)$", fontsize=18)
    ax.set_xlabel("$\Delta W_j(C)$", fontsize=18)
    
    ax.tick_params(
        axis="both", 
        which="major", 
        direction="in", 
        length=6, 
        width=1, 
        labelsize=14
    )
    ticks = np.array([0, 50])
    plt.xticks(ticks, ticks + 1)
    plt.yticks(ticks, ticks + 1)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path)
