import jax
import jax.numpy as jnp
import jax.random as jr
from typing import TypeAlias

import equinox as eqx
import equinox.nn as nn
from utils import check_input_dim

Array: TypeAlias = jnp.ndarray


class TransformerBlock(eqx.Module):
    attn: nn.MultiheadAttention
    mlp: nn.MLP
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    use_layer_norm: bool

    def __init__(
            self,
            n_embed: int, 
            n_heads: int, 
            *, 
            key: jr.PRNGKey, 
            use_layer_norm: bool = False
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
            width_size=4*n_embed,
            depth=2,
            activation=jax.nn.gelu,
            key=k2,
        )
        self.use_layer_norm = use_layer_norm
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def attention_layer(self, x):
        if self.use_layer_norm:
            x = jax.vmap(jax.vmap(self.norm1))(x)
        return jax.vmap(self.attn)(x, x, x)

    def __call__(self, x, *, idx=-1):
        assert x.ndim == 3, (
            f"Expected input shape (batch, seq_len, dim), got {x.shape}."
        )
        x = self.attention_layer(x)
        if self.use_layer_norm:
            x = jax.vmap(jax.vmap(self.norm2))(x)

        x = jax.vmap(jax.vmap(self.mlp))(x)
        return x


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
        use_layer_norm: bool = False,
    ):
        keys = jax.random.split(key, n_blocks)
        self.blocks = [
            TransformerBlock(
                n_embed=n_embed,
                n_heads=n_heads,
                key=k,
                use_layer_norm=use_layer_norm,
            )
            for k in keys
        ]
        self.n_blocks = n_blocks

    def __call__(self, x: Array, *, all_idxs: bool = False) -> Array:        
        for block in self.blocks:
            x = block(x)
        
        return x[:, :, -1] if all_idxs else x[:, -1, -1]
    
    

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


def apply_weight_update(model, ΔW):
    return eqx.tree_at(
        lambda m: m.mlp.layers[0].weight,
        model,
        model.mlp.layers[0].weight + ΔW
    )
