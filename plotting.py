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
    
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)  # 4, 2
    ax.plot(
        steps, 
        theory_losses, 
        label="theory", 
        linewidth=3,
        linestyle="--",
        color="black"
    )
    ax.plot(
        steps, 
        empirical_losses, 
        label="experiment", 
        linewidth=1.5,
        linestyle="-",
        color="#636EFA"
    )
    
    ax.legend(loc="best", fontsize=14)
    ax.set_xlabel("Training step", fontsize=18, labelpad=10)
    ax.set_ylabel("Test loss", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=14)
    plt.grid(True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plot_ΔWs_alignment(ΔWs_alignment, alignment_between, save_path, title=None):
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
    
    if alignment_between == "blocks":
        for i in range(len(ΔWs_alignment)):
            for j in range(len(ΔWs_alignment)):
                ax.text(
                    j, i, 
                    f"{ΔWs_alignment[i, j]:.2f}", 
                    ha="center", 
                    va="center", 
                    color="black" if abs(ΔWs_alignment[i, j]) < 0.5 else "white", 
                    fontsize=14
                )
    
    if title is not None:
        ax.set_title(title, fontsize=20, pad=12)

    ylabel = "$\Delta W_i(C)$" if (
        alignment_between == "tokens" 
    ) else "$\Delta W_\ell(C)$"
    xlabel = "$\Delta W_j(C)$" if (
        alignment_between == "tokens" 
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
    fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plot_blocks_ΔW_norms(mean_norms, std_norms, norm_type, save_path):
    n_steps, n_blocks = mean_norms.shape
    steps = [b+1 for b in range(n_steps)]
    y_axis_label = "$||\Delta W(C)_{(N+1)}||_F$" if (
        norm_type == "frob" 
    ) else "$||\Delta W(C)_{(N+1)}||_2$"
    
    _, ax = plt.subplots(figsize=(6, 3), dpi=300) 
    for block_idx in range(n_blocks):
        ax.plot(
            steps, 
            mean_norms[:, block_idx], 
            label=f"block {block_idx+1}"
        )
        ax.fill_between(
            steps,
            np.maximum(0, mean_norms[:, block_idx] - std_norms[:, block_idx]),
            mean_norms[:, block_idx] + std_norms[:, block_idx],
            alpha=0.2
        )
    
    ax.legend(loc="best", fontsize=12)
    ax.set_xlabel("Training step", fontsize=18, labelpad=10)
    ax.set_ylabel(y_axis_label, fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=16)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plot_blocks_ΔW_rank(mean_ranks, std_ranks, t, save_path):
    n_blocks = mean_ranks.shape[1]
    blocks = [i for i in range(1, n_blocks + 1)]

    fig, ax = plt.subplots(figsize=(3, 4), dpi=300)
    ax.bar(
        blocks, 
        mean_ranks[t], 
        yerr=std_ranks[t], 
        capsize=10, 
        color="skyblue", 
        edgecolor="black"
    )
    
    ax.set_title(f"$t = {t}$", fontsize=20, pad=12)
    ax.set_xlabel("Block", fontsize=18)
    ax.set_ylabel(r"$\mathrm{rank}(\sum_i\Delta W_i(C))$", fontsize=18)

    ax.set_xticks(blocks)
    ax.set_yticks([0, 1])

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plot_metrics(metrics, save_dir):
    
    # --- losses ---
    plot_empirical_vs_theory_losses(
        metrics["test_losses"],
        metrics["theory_test_losses"],
        f"{save_dir}/test_losses.pdf"
    )
    
    # --- updates' norms ---
    ΔWs_frob_norms = metrics["ΔWs_frob_norms"]
    ΔWs_spectral_norms = metrics["ΔWs_spectral_norms"]
    n_steps = ΔWs_frob_norms.shape[0]

    plot_blocks_ΔW_norms(
        mean_norms=ΔWs_frob_norms.mean(axis=-1), 
        std_norms=ΔWs_frob_norms.std(axis=-1),
        norm_type="frob", 
        save_path=f"{save_dir}/blocks_ΔW_frob_norms.pdf"
    )
    plot_blocks_ΔW_norms(
        mean_norms=ΔWs_spectral_norms.mean(axis=-1), 
        std_norms=ΔWs_spectral_norms.std(axis=-1),
        norm_type="spectral", 
        save_path=f"{save_dir}/blocks_ΔW_spectral_norms.pdf"
    )
        
    # --- updates' rank ---
    updates_ranks = metrics["updates_ranks"]
    for t in [0, n_steps - 1]:
        plot_blocks_ΔW_rank(
            mean_ranks=updates_ranks.mean(axis=-1),
            std_ranks=updates_ranks.std(axis=-1),
            t=t,
            save_path=f"{save_dir}/blocks_ΔW_rank_t_{t}.pdf"
        )
