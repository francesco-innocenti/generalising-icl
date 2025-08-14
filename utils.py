import os
import random
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_save_dir(
        save_dir,
        n_tasks,
        seq_len,
        input_dim,
        n_embed,
        n_heads,
        n_blocks,
        block_idx_to_verify,
        use_skips,
        hidden_multiplier,
        n_steps,
        param_lr,
        seed,
        print_config=True
):
    if print_config:
        print(
            f"""
    Starting experiment with config:

    N tasks: {n_tasks}
    Sequence length: {seq_len}
    Input dim: {input_dim}
    N embed: {n_embed}
    N heads: {n_heads}
    N blocks: {n_blocks}
    Block idx to verify: {block_idx_to_verify}
    Use skips: {use_skips}
    Hidden multiplier: {hidden_multiplier}
    Training steps: {n_steps}
    Param lr: {param_lr}
    Seed: {seed}
    """
        )

    save_dir = os.path.join(
        save_dir,
        f"{n_tasks}_tasks",
        f"seq_len_{seq_len}",
        f"input_dim_{input_dim}",
        f"n_embed_{n_embed}",
        f"{n_heads}_heads",
        f"{n_blocks}_blocks",
        f"block_idx_to_verify_{block_idx_to_verify}",
        "skips" if use_skips else "no_skips",
        f"hidden_multiplier_{hidden_multiplier}",
        f"{n_steps}_steps",
        f"param_lr_{param_lr}",
        str(seed)
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def compute_ΔWs_alignment(ΔWs):
    """Computes the normalised Frobenius inner product between
    all possible ΔW_i and ΔW_j.
    
    """
    seq_length = ΔWs.shape[0]
    ΔWs_flat = ΔWs.reshape(seq_length, -1)    # (N, hidden_dim * D)
    
    norms = np.linalg.norm(ΔWs_flat, axis=1)  # (N,)
    inner_prods = ΔWs_flat @ ΔWs_flat.T       # (N, N)
    norm_matrix = np.outer(norms, norms)      # (N, N)

    # (N, N)
    normalised_frob = inner_prods / norm_matrix
    normalised_frob = (normalised_frob + normalised_frob.T) / 2

    return normalised_frob
