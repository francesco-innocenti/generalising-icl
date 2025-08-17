import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.formatter.use_mathtext": True
})


def plot_empirical_vs_theory_losses(empirical_losses, theory_losses, save_path):
    n_steps = len(empirical_losses)
    steps = [b+1 for b in range(n_steps)]
    
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300) 
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
    fig.savefig(save_path)
    plt.close("all")


def plot_ΔWs_alignment(ΔWs_alignment, alignment_type, save_path, title=None):
    N = len(ΔWs_alignment) - 1
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    im = plt.imshow(
        ΔWs_alignment, 
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

    ylabel = "$\Delta W_i(C)$" if (
        alignment_type == "tokens" 
    ) else "$\Delta W_\ell(C)$"
    xlabel = "$\Delta W_j(C)$" if (
        alignment_type == "tokens" 
    ) else "$\Delta W_k(C)$"
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
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


def plot_norms(norms, norm_type, save_path, stds=None):
    n_steps, n_blocks = norms.shape
    steps = [b+1 for b in range(n_steps)]
    y_axis_label = "$||\Delta W(C)||_F$" if (
        norm_type == "frob" 
    ) else "$||\Delta W(C)||_2$"
    
    _, ax = plt.subplots(figsize=(6, 3), dpi=300) 
    for block_idx in range(n_blocks):
        ax.plot(
            steps, 
            norms[:, block_idx], 
            label=f"block {block_idx+1}"
        )
        if stds is not None:
            ax.fill_between(
                steps,
                norms[:, block_idx] - stds[:, block_idx],
                norms[:, block_idx] + stds[:, block_idx],
                alpha=0.2
            )
    
    ax.legend(fontsize=16)
    ax.set_xlabel("Training step", fontsize=18, labelpad=10)
    ax.set_ylabel(y_axis_label, fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=14)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_blocks_update_rank(ranks, t, save_path):
    n_blocks = ranks.shape[1]
    blocks = [i for i in range(1, n_blocks + 1)]

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.bar(
        blocks, 
        ranks[t, :].mean(axis=-1), 
        yerr=ranks[t, :].std(axis=-1), 
        capsize=10, 
        color="skyblue", 
        edgecolor="black"
    )

    ax.set_xlabel("Block", fontsize=18)
    ax.set_ylabel(r"$\mathrm{rank}(\Delta W(C))$", fontsize=18)

    ax.set_xticks(blocks)
    ax.set_yticks([0, 1])

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close("all")
