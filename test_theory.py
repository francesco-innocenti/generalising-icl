import numpy as np
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax

import copy
import argparse
from utils import set_seed, setup_experiment
from data import generate_linear_data
from model import Transformer, apply_weight_update
from analytical import compute_ΔW


def main(
        seed: jr.PRNGKey,
        batch_size: int,
        seq_len: int,
        input_dim: int,
        n_embed: int,
        n_heads: int,
        n_blocks: int,
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
    C_x_train, y_train = generate_linear_data(
        batch_size=batch_size,
        seq_len=seq_len,
        dim=input_dim,
        key=train_key
    )
    C_x_test, y_test = generate_linear_data(
        batch_size=batch_size,
        seq_len=seq_len,
        dim=input_dim,
        key=test_key
    )
    x_test = C_x_test[:, -1:]

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

    train_losses = []
    test_losses, theory_test_losses = [], []
    preds_diffs = []
    for t in range(n_steps):
        
        # test
        preds = model(C_x_test, all_idxs=True)
        test_loss = 0.5 * jnp.mean((y_test - preds[:, -1]) ** 2)

        # ΔW model
        model_copy = copy.deepcopy(model)
        ΔW = compute_ΔW(model=model_copy, C_x=C_x_test, x=x_test)
        model_updated = apply_weight_update(model_copy, ΔW)
        theory_preds = model_updated(x_test, all_idxs=True)
        theory_test_loss = 0.5 * jnp.mean((y_test - theory_preds[:, -1]) ** 2)
        
        # train
        model, opt_state, train_loss = model.train_step(
            model,
            opt_state,
            C_x_train,
            y_train,
            optim
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        theory_test_losses.append(theory_test_loss)
        preds_diffs.append(sum(preds - theory_preds))

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--n_embed', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--use_layer_norm', action="store_true")  # false by default
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--param_lr', type=float, default=5e-3)
    args = parser.parse_args()

    save_dir = setup_experiment(**vars(args))
    args.save_dir = save_dir
    main(**vars(args))
