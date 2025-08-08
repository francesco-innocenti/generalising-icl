import jax
import jax.numpy as jnp
import equinox as eqx
from typing import TypeAlias

Array: TypeAlias = jnp.ndarray


@eqx.filter_jit
def compute_ΔW(
        model: eqx.Module, 
        input_prompt: Array, 
        input_query: Array, 
        output_idx: int = -1
    ):
    """Computes ΔW according to 
        
        ΔW(C)[i] = (W * ΔA[i]) * A(x)[i]^T / ||A(x)[i]||²
    
    where ΔA[i] = A(C, x)[i] - A(x)[i]. Note that this is a slight 
    generalisation of Equation 8 in https://arxiv.org/abs/2507.16003 for any 
    output token i, not just the last.

    Args:
        model: equinox model.
        input_prompt: input with context and query.
        input_query: input with only query (and no context).
        output_idx: output token index for which to compute ΔW.

    Returns:   
        ΔW[i] in the shape of W.

    """
    # A(C, x): attention output with full context
    attn_full = model.attention_layer(input_prompt)  # (B, N+1, D)
    A_C_x = attn_full[:, output_idx, :]  # (B, D)

    # A(x): attention output with query only (no context)
    attn_query_only = model.attention_layer(input_query)  # (B, 1, D)
    A_x = attn_query_only[:, output_idx, :]  # (B, D)

    # ΔA = A(C,x) - A(x)
    ΔA = A_C_x - A_x  # (B, D)

    # Get the weight matrix W from first MLP layer
    W = model.mlp.layers[0].weight  # (hidden_dim, D)

    def compute_single_ΔW(ΔA_i, A_x_i):
        """Computes ΔW for a single example in the batch."""
        # W * ΔA (column vector)
        W_ΔA = W @ ΔA_i  # (hidden_dim,)

        # A(x)^T (row vector)
        A_x_T = A_x_i  # (D,)

        # Outer product: (W * ΔA) * A(x)^T
        numerator = W_ΔA[:, None] @ A_x_T[None, :]  # (hidden_dim, D)

        # ||A(x)||²
        denominator = jnp.linalg.norm(A_x_i) ** 2 #+ 1e-8  # optional epsilon for stability

        return numerator / denominator

    # Compute ΔW for each example in batch
    ΔWs = jax.vmap(compute_single_ΔW)(ΔA, A_x)  # (B, hidden_dim, D)

    # Average across batch
    return ΔWs.mean(axis=0)
