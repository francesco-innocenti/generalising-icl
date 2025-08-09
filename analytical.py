import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TypeAlias

Array: TypeAlias = jnp.ndarray


@eqx.filter_jit
def compute_ΔW(model: eqx.Module, C_x: Array, x: Array, block_idx: int = 0):
    """Computes ΔW according to 
        
        ΔW(C) = (W * ΔA) * A(x)^T / ||A(x)||²
    
    where ΔA = A(C, x) - A(x). Note that this is a slight generalisation of 
    Equation 8 in https://arxiv.org/abs/2507.16003 for all output tokens, not 
    just the last.

    Args:
        model: equinox model.
        C_x: input with context and query (B, N+1, D).
        x: input with only query (and no context) (B, 1, D).
        block_idx: transformer block index for which to compute ΔW.

    Returns:   
        ΔW for all output tokens (N hidden_dim, D).

    """
    # A(C, x): attention output with full context → (B, N+1, D)
    A_C_x = model.blocks[block_idx].attention_layer(C_x)

    # A(x): attention output with query only (no context) → (B, 1, D)
    A_x = model.blocks[block_idx].attention_layer(x)

    # broadcast A_x to match sequence length N of A_C_x → (B, N+1, D)
    A_x = jnp.broadcast_to(A_x, A_C_x.shape)

    # ΔA = A(C,x) - A(x) → (B, N, D)
    ΔA = A_C_x - A_x

    # Get the weight matrix W from first MLP layer → (hidden_dim, D)
    W = model.blocks[block_idx].mlp.layers[0].weight

    def compute_single_ΔW(ΔA_i, A_x_i):
        """ΔA_i: (N, D,), A_x_i: (N, D,) → ΔW_i: (N, hidden_dim, D)"""
        W_ΔA = W @ ΔA_i  # (hidden_dim,)
        numerator = W_ΔA[:, None] @ A_x_i[None, :]  # (hidden_dim, D)
        denominator = jnp.linalg.norm(A_x_i) ** 2
        return numerator / denominator

    # vmap over sequence positions and batch
    compute_over_seq = jax.vmap(jax.vmap(compute_single_ΔW))

    # (B, N, hidden_dim, D)
    ΔWs = compute_over_seq(ΔA, A_x)

    # Average across batch → (N, hidden_dim, D)
    return ΔWs.mean(axis=0)
