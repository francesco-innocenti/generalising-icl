import copy
import argparse
import numpy as np

import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

from utils import set_seed, get_save_dir
from data import generate_linear_tasks
from model import Transformer, apply_mlp_weight_update, train_step
from analytical import compute_ΔW


def main(
        seed: jr.PRNGKey,
        n_tasks: int,
        seq_len: int,
        input_dim: int,
        n_embed: int,
        n_heads: int,
        n_blocks: int,
        block_idx: int,
        use_layer_norm: bool,
        n_steps: int,
        param_lr: float,
        save_dir: str
):
    set_seed(seed)
    key = jr.PRNGKey(seed)
    data_key, model_key = jr.split(key, 2)

    # data
    train_key, test_key = jr.split(data_key, 2)
    C_x_train, y_train = generate_linear_tasks(
        n_tasks=n_tasks,
        seq_len=seq_len,
        dim=input_dim,
        key=train_key
    )
    C_x_test, y_test = generate_linear_tasks(
        n_tasks=n_tasks,
        seq_len=seq_len,
        dim=input_dim,
        key=test_key
    )

    # model and optims
    model = Transformer(
        n_embed=n_embed,
        n_heads=n_heads,
        n_blocks=n_blocks,
        key=model_key,
        use_layer_norm=use_layer_norm
    )
    optim = optax.adam(param_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    train_losses, test_losses = [], []
    theory_test_losses = [] if block_idx+1 == n_blocks else None
    preds_diffs = []
    for t in range(n_steps):

        # test
        preds, block_preds = model(C_x_test, return_activations=True)
        test_loss = 0.5 * jnp.mean((y_test - preds) ** 2)

        # ΔW model
        all_new_C_x_test = [C_x_test] + block_preds
        new_C_x_test = all_new_C_x_test[block_idx]
        new_x_test = new_C_x_test[:, -1:]
        model_copies = [copy.deepcopy(model) for _ in range(seq_len+1)]
        ΔWs = compute_ΔW(
            model=model_copies[0], 
            C_x=new_C_x_test, 
            x=new_x_test,
            block_idx=block_idx
        )
        theory_block_preds = np.zeros((n_tasks, seq_len+1, n_embed))
        for i in range(seq_len+1):
            model_copies[i] = apply_mlp_weight_update(
                model_copies[i], 
                ΔWs[i],
                block_idx=block_idx
            )
            theory_block_preds[:, i] = model_copies[i].blocks[block_idx](new_x_test)
        
        if block_idx+1 == n_blocks:
            theory_test_loss = 0.5 * jnp.mean(
                (y_test - theory_block_preds[:, -1, -1]) ** 2)
            theory_test_losses.append(theory_test_loss)

        # train
        model, opt_state, train_loss = train_step(
            model,
            opt_state,
            C_x_train,
            y_train,
            optim
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        preds_diffs.append(
            (block_preds[block_idx] - theory_block_preds).sum()
        )

        if t % 100 == 0:
            print(f"Step {t} | Loss: {train_loss:.4f}")

    np.save(f"{save_dir}/train_losses.npy", train_losses)
    np.save(f"{save_dir}/test_losses.npy", test_losses)
    np.save(f"{save_dir}/theory_test_losses.npy", theory_test_losses)
    np.save(f"{save_dir}/preds_diffs.npy", preds_diffs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="results")
    parser.add_argument('--seed', type=int, default=53093)
    parser.add_argument('--n_tasks', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--n_embed', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--block_idx', type=int, default=2)
    parser.add_argument('--use_layer_norm', action="store_true")  # false by default
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--param_lr', type=float, default=5e-3)
    args = parser.parse_args()
    assert args.n_embed == args.input_dim + 1
    assert args.block_idx < args.n_blocks

    save_dir = get_save_dir(**vars(args))
    args.save_dir = save_dir
    main(**vars(args))
