import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True,
})


def plot_empirical_vs_theory_losses(empirical_losses, theory_losses, save_path):
    n_steps = len(empirical_losses)
    steps = [b+1 for b in range(n_steps)]
    
    _, ax = plt.subplots(figsize=(4, 2)) 
    ax.plot(
        steps, 
        theory_losses, 
        label="theory", 
        linewidth=4,
        linestyle="--",
        color="black"
    )
    ax.plot(
        steps, 
        empirical_losses, 
        label="experiment", 
        linewidth=2,
        linestyle="-",
        color="#636EFA"
    )
    
    ax.legend(fontsize=16)
    ax.set_xlabel("Training step", fontsize=18, labelpad=10)
    ax.set_ylabel("Test loss", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_ΔWs_alignments(ΔWs_alignments, save_path, title=None):
    N = len(ΔWs_alignments)-1
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
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Directional alignment", fontsize=18)
    
    if title is not None:
        ax.set_title(title, fontsize=20, pad=12)

    ax.set_ylabel("$\Delta W_i(C)$", fontsize=18)
    ax.set_xlabel("$\Delta W_j(C)$", fontsize=18)
    
    ax.tick_params(
        axis="both", 
        which="major", 
        direction="in", 
        length=6, 
        width=1, 
        labelsize=18
    )
    ticks = np.array([0, int(N/2), N])
    plt.xticks(ticks, ticks + 1)
    plt.yticks(ticks, ticks + 1)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close("all")
