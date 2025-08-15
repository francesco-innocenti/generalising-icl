import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypeAlias

import equinox as eqx
import equinox.nn as nn

Array: TypeAlias = jnp.ndarray


class TransformerBlock(eqx.Module):
    attn: nn.MultiheadAttention
    mlp: nn.MLP
    norm_attn: nn.LayerNorm
    norm_mlp: nn.LayerNorm
    use_layer_norm: bool
    use_skips: bool

    def __init__(
            self,
            n_embed: int, 
            n_heads: int, 
            *, 
            key: jr.PRNGKey,
            use_skips: bool = False,
            use_layer_norm: bool = False,
            hidden_multiplier: int = 4
        ):
        k1, k2 = jax.random.split(key, 2)
        self.attn = nn.MultiheadAttention(
            query_size=n_embed,
            num_heads=n_heads,
            key=k1,
        )
        self.mlp = nn.MLP(
            in_size=n_embed,
            out_size=n_embed,
            width_size=hidden_multiplier * n_embed,
            depth=2,
            activation=jax.nn.gelu,
            key=k2,
        )
        self.norm_attn = nn.LayerNorm(n_embed)
        self.norm_mlp = nn.LayerNorm(n_embed)
        self.use_layer_norm = use_layer_norm
        self.use_skips = use_skips

    def attention_layer(self, x):
        if self.use_layer_norm:
            x = jax.vmap(jax.vmap(self.norm_attn))(x)
        return jax.vmap(self.attn)(x, x, x)

    def mlp_layer(self, x):
        if self.use_layer_norm:
            x = jax.vmap(jax.vmap(self.norm_mlp))(x)
        return jax.vmap(jax.vmap(self.mlp))(x)

    def __call__(self, x):
        assert x.ndim == 3, (
            f"Expected input shape (batch, seq_len, dim), got {x.shape}."
        )
        if self.use_skips:
            attn_out = x + self.attention_layer(x)
            mlp_out = attn_out + self.mlp_layer(attn_out)
        else:
            attn_out = self.attention_layer(x)
            mlp_out = self.mlp_layer(attn_out)

        return mlp_out


class Transformer(eqx.Module):
    blocks: list
    n_blocks: int

    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        n_blocks: int,
        *,
        key: jr.PRNGKey,
        use_skips: bool = False,
        use_layer_norm: bool = False,
        hidden_multiplier: int = 4
    ):
        keys = jax.random.split(key, n_blocks)
        self.blocks = [
            TransformerBlock(
                n_embed=n_embed, 
                n_heads=n_heads, 
                key=k,
                use_skips=use_skips,
                use_layer_norm=use_layer_norm,
                hidden_multiplier=hidden_multiplier
            )
            for k in keys
        ]
        self.n_blocks = n_blocks

    def __call__(self, x, return_activations=False):        
        activations = [] if return_activations else None
        for block in self.blocks:
            x = block(x)
            if return_activations:
                activations.append(x)
        
        out = x[:, -1, -1]
        return (out, activations) if return_activations else out
    
    
@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    pred = model(x)
    return 0.5 * jnp.mean((y - pred) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, x, y, optim):
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def apply_icl_updates(model, ΔW, Δb, block_idx=0):
    """
    Applies updates to the first MLP layer and second bias of a transformer block.
    """
    weight_path = lambda m: m.blocks[block_idx].mlp.layers[0].weight
    bias_path = lambda m: m.blocks[block_idx].mlp.layers[-1].bias
    
    W = weight_path(model)
    b = bias_path(model)

    assert W.shape == ΔW.shape, f"ΔW shape {ΔW.shape} != W shape {W.shape}"
    assert b.shape == Δb.shape, f"Δb shape {Δb.shape} != b shape {b.shape}"

    def where_fn(m):
        return (weight_path(m), bias_path(m))
    
    model = eqx.tree_at(
        where_fn,
        model,
        (W + ΔW, b + Δb)
    )
    return model


def _single_block_pass(
        ΔW, 
        Δb, 
        single_x, 
        model, 
        block_idx, 
        apply_updates_fn
    ):
    """
    Performs a single forward pass for one update on one task.
    This function is designed to be vmapped.
    """
    block_input = single_x[None, :, :]  # (1, 1, D)
    updated_model = apply_updates_fn(model, ΔW, Δb, block_idx)
    output = updated_model.blocks[block_idx](block_input)  # (1, 1, D)
    return jnp.squeeze(output, axis=(0, 1))  # (D,)


@eqx.filter_jit
def compute_vectorised_theory_preds(base_model, x, ΔWs, Δbs, block_idx):
    """
    Vectorises theoretical block predictions for different updates over batches 
    and token positions.

    Args:
        base_model: equinox Transformer model
        x: batched D-dimensional input query token (B, 1, D)
        ΔWs: Weight updates unique for each batch and token position (B, N+1, H, D)
        Δbs: Bias updates unique for each batch and token position (B, N+1, D)
        block_idx: index of the transformer block to update

    Returns:
        Prediction array of shape (B, N+1, D)

    """
    # vmap over the sequence length (N) updates for a *single task*
    # NOTE: for a single task, `single_x` is constant across all updates
    vmap_over_updates = jax.vmap(
        _single_block_pass, 
        in_axes=(0, 0, None, None, None, None)  # vmap over Δw, Δb
    )

    # vmap the above function over the task/batch (B) dimension
    # now we provide an axis for `single_x` because each task has its own input
    vmap_over_tasks = jax.vmap(
        vmap_over_updates, 
        in_axes=(0, 0, 0, None, None, None)  # vmap over ΔWs, Δbs, x
    )

    # we pass apply_icl_updates as an argument to make the function pure 
    # and JIT-compatible
    return vmap_over_tasks(ΔWs, Δbs, x, base_model, block_idx, apply_icl_updates)
