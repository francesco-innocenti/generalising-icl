import os
import argparse
import numpy as np

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

from data import generate_linear_tasks
from analytical import compute_implicit_icl_updates
from model import (
    Transformer, 
    forward, 
    train_step, 
    compute_vectorised_theory_preds
)
from utils import (
    set_seed, 
    get_save_dir,
    get_lr,
    compute_ΔWs_alignment, 
    compute_effective_update_rank
)
from plotting import plot_ΔWs_alignment, plot_metrics


def main(
        seed: jr.PRNGKey,
        n_tasks: int,
        seq_len: int,
        input_dim: int,
        n_heads: int,
        n_blocks: int,
        use_skips: bool,
        use_layer_norm: bool,
        hidden_multiplier: int,
        n_steps: int,
        lr: float,
        save_dir: str
):  
    n_embed = input_dim + 1
    assert n_embed % n_heads == 0
    
    alignments_dir = f"{save_dir}/ΔWs_alignments"
    os.makedirs(alignments_dir, exist_ok=True)

    set_seed(seed)
    key = jr.PRNGKey(seed)
    train_key, test_key, model_key = jr.split(key, 3)

    # --- model & optim ---
    model = Transformer(
        n_embed=n_embed,
        n_heads=n_heads,
        n_blocks=n_blocks,
        key=model_key,
        use_skips=use_skips,
        use_layer_norm=use_layer_norm,
        hidden_multiplier=hidden_multiplier
    )
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    # --- metrics ---
    train_losses, test_losses, theory_test_losses = [], [], []
    theory_preds_squared_diffs = np.zeros((n_steps, n_blocks))
    
    ΔWs_steps = np.zeros(
        (n_steps, n_blocks, n_tasks, seq_len+1, 
         hidden_multiplier * n_embed, n_embed)
    )
    updates_ranks = np.zeros((n_steps, n_blocks, n_tasks))
    random_task_idxs = np.random.choice(
        np.arange(0, n_tasks), 
        size=3 if n_tasks >= 3 else 1,
        replace=False
    )
    for t in range(n_steps):
        
        # --- generate data ---
        train_key, step_train_key = jr.split(train_key)
        test_key, step_test_key = jr.split(test_key)

        C_x_train, y_train = generate_linear_tasks(
            n_tasks=n_tasks,
            seq_len=seq_len,
            dim=input_dim,
            key=step_train_key
        )
        C_x_test, y_test = generate_linear_tasks(
            n_tasks=n_tasks,
            seq_len=seq_len,
            dim=input_dim,
            key=step_test_key
        )

        # --- test empirical vs theory preds ---
        preds, block_preds = forward(model, C_x_test, return_activations=True)
        test_loss = 0.5 * jnp.mean((y_test - preds) ** 2)

        all_new_C_x_test = [C_x_test] + block_preds
        for block_idx in range(n_blocks):
            new_C_x_test = all_new_C_x_test[block_idx]
            new_x_test = new_C_x_test[:, -1:]

            ΔWs, Δbs = compute_implicit_icl_updates(
                model=model, 
                C_x=new_C_x_test, 
                x=new_x_test,
                block_idx=block_idx,
                use_skips=use_skips
            )
            ΔWs_steps[t, block_idx] = ΔWs
            updates_ranks[t, block_idx] = compute_effective_update_rank(ΔWs)

            theory_block_preds = compute_vectorised_theory_preds(
                base_model=model, 
                x=new_x_test, 
                ΔWs=ΔWs, 
                Δbs=Δbs, 
                block_idx=block_idx
            )
            theory_preds_squared_diffs[t, block_idx] = ( 
                (block_preds[block_idx] - theory_block_preds)**2 ).sum()
        
        theory_test_loss = 0.5 * jnp.mean(
            (y_test - theory_block_preds[:, -1, -1]) ** 2)

        # --- train ---
        model, opt_state, train_loss = train_step(
            model,
            opt_state,
            C_x_train,
            y_train,
            optim
        )
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        theory_test_losses.append(theory_test_loss)

        if t % 10 == 0:
            print(
                f"Step {t} | train loss: {train_loss:.4f} | "
                f"test loss: {test_loss:.4f}"
            ) 
    
            # --- plot alignment between token positions (for all blocks) ---
            t_str = f"{t:04d}"
            for block_idx in range(n_blocks):
                for task in random_task_idxs:
                    ΔWs_tokens_alignment = compute_ΔWs_alignment(
                        ΔWs_steps[t, block_idx, task]
                    )
                    save_path = (
                        f"{alignments_dir}/ΔWs_tokens_alignment_"
                        f"t_{t_str}_block_{block_idx+1}_task_{task}.pdf"
                    )
                    plot_ΔWs_alignment(
                        ΔWs_tokens_alignment,
                        alignment_between="tokens",
                        save_path=save_path,
                        title=f"$t = {t}$"
                    )
            
            # --- plot alignment between blocks (for last token) ---
            for task in random_task_idxs:
                ΔWs_blocks_alignment = compute_ΔWs_alignment(
                    ΔWs_steps[t, :, task, -1]
                )
                save_path = (
                    f"{alignments_dir}/ΔWs_blocks_alignment_"
                    f"t_{t_str}_task_{task}.pdf"
                )
                plot_ΔWs_alignment(
                    ΔWs_blocks_alignment,
                    alignment_between="blocks",
                    save_path=save_path,
                    title=f"$t = {t}$"
                )   
  
    # --- saving ---
    np.save(f"{save_dir}/train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/theory_test_losses.npy", theory_test_losses)

    np.save(f"{save_dir}/theory_preds_squared_diffs.npy", theory_preds_squared_diffs)
    np.save(f"{save_dir}/ΔWs_steps.npy", ΔWs_steps)
    np.save(f"{save_dir}/updates_ranks.npy", updates_ranks)
    
    # --- compute norms ---
    ΔWs_task_last_token = ΔWs_steps[:, :, :, -1]
    ΔWs_frob_norms = jnp.linalg.norm(                       # (T, L, B)   
        ΔWs_task_last_token, 
        ord="fro", 
        axis=(-2, -1)
    )
    ΔWs_spectral_norms = jnp.linalg.svd(
        ΔWs_task_last_token, compute_uv=False)[:, :, :, 0]  # (T, L, B)  
    
    # --- plotting ---
    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "theory_test_losses": theory_test_losses,
        "theory_preds_squared_diffs": theory_preds_squared_diffs,
        "ΔWs_frob_norms": ΔWs_frob_norms,
        "ΔWs_spectral_norms": ΔWs_spectral_norms,
        "updates_ranks": updates_ranks
    }
    plot_metrics(metrics, save_dir)


def run_single_param_sweeps(base_args, sweeps: dict):
    """
    sweeps: dict mapping parameter name -> list of values
    Example:
        {"n_tasks": [16, 32, 64], "seq_len": [50, 250]}
    """
    for param, values in sweeps.items():
        print(f"\nRunning sweep for {param}")
        for v in values:
            args = argparse.Namespace(**vars(base_args))
            setattr(args, param, v)
            delattr(args, "blocks_analysis")

            args.save_dir = get_save_dir(
                save_dir=base_args.save_dir,
                n_tasks=args.n_tasks,
                seq_len=args.seq_len,
                input_dim=args.input_dim,
                n_heads=args.n_heads,
                n_blocks=args.n_blocks,
                use_skips=args.use_skips,
                use_layer_norm=args.use_layer_norm,
                hidden_multiplier=args.hidden_multiplier,
                n_steps=args.n_steps,
                lr=args.lr,
                seed=args.seed
            )
            main(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="results")
    parser.add_argument('--seed', type=int, default=53093)
    parser.add_argument('--n_tasks', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=5)
    parser.add_argument('--use_skips', type=bool, default=True)
    parser.add_argument('--use_layer_norm', type=bool, default=False)
    parser.add_argument('--hidden_multiplier', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--blocks_analysis', type=bool, default=True)
    args = parser.parse_args()
    
    if args.blocks_analysis:
        
        sweeps = {
            "n_tasks": [2**i for i in range(4, 13)],
            "seq_len": [50, 100, 1000],
            "input_dim": [2, 10],
            "n_heads": [1, 3],
            "use_layer_norm": [False, True]
        }
        args.use_skips = True
        args.hidden_multiplier = 4
        args.n_steps = 100

        n_seeds = 3
        n_blocks_list = [1, 3, 5]
        for seed in range(n_seeds):
            args.seed = seed
            for n_blocks in n_blocks_list:
                args.n_blocks = n_blocks
                args.lr = get_lr(n_blocks)
                run_single_param_sweeps(args, sweeps)
    
    else:
        
        delattr(args, "blocks_analysis")
        args.lr = get_lr(args.n_blocks)
        args.save_dir = get_save_dir(
            save_dir=args.save_dir,
            n_tasks=args.n_tasks,
            seq_len=args.seq_len,
            input_dim=args.input_dim,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            use_skips=args.use_skips,
            use_layer_norm=args.use_layer_norm,
            hidden_multiplier=args.hidden_multiplier,
            n_steps=args.n_steps,
            lr=args.lr,
            seed=args.seed
        )
        main(**vars(args))
