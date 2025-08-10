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
        block_idx,
        use_layer_norm,
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
    Block idx: {block_idx}
    Use layer norm: {use_layer_norm}
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
        f"block_idx_{block_idx}",
        f"layer_norm" if use_layer_norm else "no_layer_norm",
        f"{n_steps}_steps",
        f"param_lr_{param_lr}",
        str(seed)
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
