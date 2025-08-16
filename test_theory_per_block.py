import argparse
import numpy as np

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

from utils import (
    set_seed, 
    get_save_dir,
    compute_ΔWs_alignment, 
    compute_effective_update_rank
)
from data import generate_linear_tasks
from model import Transformer, train_step, compute_vectorised_theory_preds
from analytical import compute_icl_updates
from plotting import plot_empirical_vs_theory_losses, plot_ΔWs_alignments


def main(
        seed: jr.PRNGKey,
        n_tasks: int,
        seq_len: int,
        input_dim: int,
        n_heads: int,
        n_blocks: int,
        block_idx_to_verify: int,
        use_skips: bool,
        use_layer_norm: bool,
        hidden_multiplier: int,
        n_steps: int,
        param_lr: float,
        save_dir: str
):  
    n_embed = input_dim + 1
    assert n_embed % n_heads == 0
    assert block_idx_to_verify < n_blocks

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
    optim = optax.adam(param_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    # --- metrics ---
    train_losses, test_losses = [], []
    theory_block_preds_diffs = []
    theory_test_losses = [] if block_idx_to_verify+1 == n_blocks else None
    
    effective_update_ranks = np.zeros((n_steps, n_tasks))
    ΔWs_steps = np.zeros(
        (n_steps, n_tasks, seq_len+1, hidden_multiplier * n_embed, n_embed)
    )
    random_data_idxs = np.random.choice(
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
        preds, block_preds = model(C_x_test, return_activations=True)
        test_loss = 0.5 * jnp.mean((y_test - preds) ** 2)

        all_new_C_x_test = [C_x_test] + block_preds
        new_C_x_test = all_new_C_x_test[block_idx_to_verify]
        new_x_test = new_C_x_test[:, -1:]
        ΔWs, Δbs = compute_icl_updates(
            model=model, 
            C_x=new_C_x_test, 
            x=new_x_test,
            block_idx=block_idx_to_verify,
            use_skips=use_skips
        )
        ΔWs_steps[t] = ΔWs
        effective_update_ranks[t, :] = compute_effective_update_rank(ΔWs)

        theory_block_preds = compute_vectorised_theory_preds(
            base_model=model, 
            x=new_x_test, 
            ΔWs=ΔWs, 
            Δbs=Δbs, 
            block_idx=block_idx_to_verify
        )

        if block_idx_to_verify+1 == n_blocks:
            theory_test_loss = 0.5 * jnp.mean(
                (y_test - theory_block_preds[:, -1, -1]) ** 2)
            theory_test_losses.append(theory_test_loss)

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
        theory_block_preds_diffs.append(
            ( (block_preds[block_idx_to_verify] - theory_block_preds)**2 ).sum()
        )

        if t % 20 == 0:
            print(f"Step {t} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f}")
                  
            for b in random_data_idxs:
                ΔWs_alignments = compute_ΔWs_alignment(ΔWs[b])
                plot_ΔWs_alignments(
                    ΔWs_alignments,
                    save_path=f"{save_dir}/ΔWs_alignments_b_{b}_t_{t}.pdf",
                    title=f"$t = {t}$"
                )

    # --- save ---
    np.save(f"{save_dir}/train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    if theory_test_losses is not None:
        np.save(f"{save_dir}/theory_test_losses.npy", theory_test_losses)

    np.save(f"{save_dir}/theory_block_preds_diffs.npy", theory_block_preds_diffs)
    np.save(f"{save_dir}/ΔWs_steps.npy", ΔWs_steps)
    np.save(f"{save_dir}/effective_update_ranks.npy", effective_update_ranks)
    
    if block_idx_to_verify+1 == n_blocks:
        plot_empirical_vs_theory_losses(
            test_losses,
            theory_test_losses,
            f"{save_dir}/test_losses.pdf"
        )


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

            args.save_dir = get_save_dir(
                base_args.save_dir,
                n_tasks=args.n_tasks,
                seq_len=args.seq_len,
                input_dim=args.input_dim,
                n_heads=args.n_heads,
                n_blocks=args.n_blocks,
                block_idx_to_verify=args.block_idx_to_verify,
                use_skips=args.use_skips,
                use_layer_norm=args.use_layer_norm,
                hidden_multiplier=args.hidden_multiplier,
                n_steps=args.n_steps,
                param_lr=args.param_lr,
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
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--block_idx_to_verify', type=int, default=0)
    parser.add_argument('--use_skips', type=bool, default=True)
    parser.add_argument('--use_layer_norm', type=bool, default=False)
    parser.add_argument('--hidden_multiplier', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--param_lr', type=float, default=1e-1)
    parser.add_argument('--sweep', type=bool, default=False, 
                        help="Run parameter sweeps instead of a single experiment")
    args = parser.parse_args()
    
    # --- one-block model analysis ---
    sweeps = {
        "n_tasks": [2**i for i in range(14)],
        "seq_len": [50, 250, 1250],
        "input_dim": [2, 20],
        "n_heads": [1, 3],
        "use_layer_norm": [False, True],
    }
    
    if args.sweep:
        run_single_param_sweeps(args, sweeps)
    else:
        save_dir = get_save_dir(
            args.save_dir,
            n_tasks=args.n_tasks,
            seq_len=args.seq_len,
            input_dim=args.input_dim,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            block_idx_to_verify=args.block_idx_to_verify,
            use_skips=args.use_skips,
            use_layer_norm=args.use_layer_norm,
            hidden_multiplier=args.hidden_multiplier,
            n_steps=args.n_steps,
            param_lr=args.param_lr,
            seed=args.seed
        )
        args.save_dir = save_dir
        main(**vars(args))
