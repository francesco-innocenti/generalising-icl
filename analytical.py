import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TypeAlias

Array: TypeAlias = jnp.ndarray


@eqx.filter_jit
def compute_implicit_icl_updates(
        model: eqx.Module, 
        C_x: Array, 
        x: Array, 
        block_idx: int = 0,
        use_skips: bool = False
    ):
    """Computes in-context learning updates according to 
        
        ΔW_i(C) = (W * (ΔA_i + Δz_i)) * (A(x) + x)^T / ||A(x) + x||²,
        Δb_i(C) = ΔA_i + Δz_i,
    
    where ΔA_i = A(C, x)_i - A(x) and Δz_i = (C, x)_i - x. This is a 
    generalisation of Equation 8 in https://arxiv.org/abs/2507.16003 for all 
    output tokens, the "correct" type of skip connections for Pre-LN 
    architectures, and any transformer block. Without skips, the update reduces
    to

        ΔW_i(C) = (W * ΔA_i) * A(x)^T / ||A(x)||².

    Args:
        model: equinox Transformer model
        C_x: input with context and query (B, N+1, D)
        x: input with only query (and no context) (B, 1, D)
        block_idx: index of the transformer block to update
        use_skips: whether to assume residual or skip connections

    Returns:   
        ΔW and Δb for all batches and token positions (B, N+1, H, D)

    """
    # A(C, x): attention output with full context → (B, N+1, D)
    A_C_x = model.blocks[block_idx].attention_layer(C_x)
    
    # A(x): attention output with query only (no context) → (B, 1, D)
    A_x = model.blocks[block_idx].attention_layer(x)
    
    # broadcast A_x to match sequence length of A_C_x → (B, N+1, D)
    A_x_broadcasted = jnp.broadcast_to(A_x, A_C_x.shape)

    # ΔA = A(C,x) - A(x) → (B, N+1, D)
    ΔA = A_C_x - A_x_broadcasted

    # Δz = (C,x) - x → (B, N+1, D)
    Δz = C_x - x if use_skips else jnp.zeros_like(ΔA)

    # Get the weight matrix W from first MLP layer → (H, D)
    W = model.blocks[block_idx].mlp.layers[0].weight

    def compute_single_ΔW(ΔA_i, A_x_i, Δz_i):
        """Computes updates for a single data point and token position."""
        W_ΔA = W @ (ΔA_i + Δz_i) # (H,)
        numerator = W_ΔA[:, None] @ A_x_i[None, :]  # (H, D)
        denominator = jnp.linalg.norm(A_x_i) ** 2
        ΔW_i = numerator / (denominator + 1e-8)
        return ΔW_i

    compute_over_seq = jax.vmap(jax.vmap(compute_single_ΔW))
    A_x = A_x_broadcasted + x if use_skips else A_x_broadcasted

    # (B, N+1, H, D)
    ΔWs = compute_over_seq(ΔA, A_x, Δz)

    # (B, N+1, D)
    Δbs = ΔA + Δz if use_skips else jnp.zeros_like(ΔA)

    return ΔWs, Δbs
