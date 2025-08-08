import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, TypeAlias

Array: TypeAlias = jnp.ndarray


def generate_linear_tasks(
        n_tasks: int, 
        seq_len: int, 
        dim: int, 
        key: jr.PRNGKey
    ) -> Tuple[Array, Array]:
    """Generates linear function tasks for in-context learning.

    Args:
        n_tasks: number of tasks.
        seq_len: number of context points.
        dim: dimension of the input space.
        key: random key for sampling.

    Returns:   
        (B, N, D) context inputs and (B,) target outputs.
        
    """
    B, N, D = n_tasks, seq_len, dim
    keys = jr.split(key, 3)

    W = jr.normal(keys[0], (B, D))               # one weight vector per task/batch (B, D)
    X = jr.normal(keys[1], (B, N, D))            # context inputs  (B, N, D)
    y = jnp.einsum('bd,bnd->bn', W, X)           # context outputs (B, N)

    x_query = jr.normal(keys[2], (B, D))         # query input (B, D)
    y_query = jnp.einsum('bd,bd->b', W, x_query) # query output (B,)

    return (
        create_input_matrix(X, y, x_query), 
        y_query
    )


def create_input_matrix(X: Array, y: Array, x_query: Array) -> Array:
    """Creates the input matrix E_τ as described in https://arxiv.org/abs/2507.16003.

    Args:
        X: (B, N, D) - context inputs.
        y: (B, N) - context outputs.
        x_query: (B, D) - query input.

    Returns: 
        (B, N+1, D+1) input matrix.

    """
    B = X.shape[0]

    context = jnp.concatenate([X, y[..., None]], axis=-1)  # (B, N, D+1)
    query = jnp.concatenate([
        x_query[:, None, :],        # (B, 1, D)
        jnp.zeros((B, 1, 1))        # (B, 1, 1) - placeholder for output
    ], axis=-1)                     # (B, 1, D+1)

    return jnp.concatenate([context, query], axis=1)  # (B, N+1, D+1)
