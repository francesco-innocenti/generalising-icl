import os
import random
import numpy as np

from jax import vmap
import jax.numpy as jnp


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_save_dir(
        save_dir,
        n_tasks,
        seq_len,
        input_dim,
        n_heads,
        n_blocks,
        use_skips,
        use_layer_norm,
        hidden_multiplier,
        n_steps,
        lr,
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
    N heads: {n_heads}
    N blocks: {n_blocks}
    Use skips: {use_skips}
    Use layer norm: {use_layer_norm}
    Hidden multiplier: {hidden_multiplier}
    Training steps: {n_steps}
    Learning rate: {lr}
    Seed: {seed}
    """
        )

    save_dir = os.path.join(
        save_dir,
        f"{n_tasks}_tasks",
        f"seq_len_{seq_len}",
        f"input_dim_{input_dim}",
        f"{n_heads}_heads",
        f"{n_blocks}_blocks",
        "skips" if use_skips else "no_skips",
        "layer_norm" if use_layer_norm else "no_layer_norm",
        f"hidden_multiplier_{hidden_multiplier}",
        f"{n_steps}_steps",
        f"lr_{lr}",
        str(seed)
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def compute_ΔWs_alignment(ΔWs):
    """Computes the normalised Frobenius inner product between ΔW.

    This is used to estimate the alignment of the implicit weight updates for 
    either all token positions ΔW_i for a given block or all blocks for the 
    last token ΔW_(N + 1). 
    
    """
    iv = ΔWs.shape[0]
    ΔWs_flat = ΔWs.reshape(iv, -1)    # (m, hidden_dim * D)
    
    norms = jnp.linalg.norm(ΔWs_flat, axis=1)  # (m,)
    inner_prods = ΔWs_flat @ ΔWs_flat.T       # (m, m)
    norm_matrix = jnp.outer(norms, norms)      # (m, m)

    # (m, m)
    normalised_frob = inner_prods / norm_matrix
    normalised_frob = (normalised_frob + normalised_frob.T) / 2

    return normalised_frob


def compute_effective_update_rank(ΔWs):
    ΔW_sequence_sum = ΔWs.sum(axis=1)  # (B, H, D)   
    rank_per_batch = vmap(jnp.linalg.matrix_rank)(ΔW_sequence_sum)
    return rank_per_batch  # (B,)
