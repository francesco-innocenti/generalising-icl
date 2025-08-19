import jax.numpy as jnp
import jax.random as jr
from equinox import filter_jit
from typing import Tuple, TypeAlias

Array: TypeAlias = jnp.ndarray


@filter_jit
def generate_linear_tasks(
        n_tasks: int, 
        seq_len: int, 
        dim: int, 
        key: jr.PRNGKey
    ) -> Tuple[Array, Array]:
    """Generates linear function tasks for in-context learning.

    Args:
        n_tasks: number of tasks
        seq_len: number of context points
        dim: dimension of the input space
        key: random key for sampling

    Returns:   
        Context inputs (B, N, D) and target outputs (B,)
        
    """
    B, N, D = n_tasks, seq_len, dim
    keys = jr.split(key, 3)

    W = jr.normal(keys[0], (B, D))               # one weight vector per task/batch (B, D)
    x = jr.normal(keys[1], (B, N, D))            # context inputs  (B, N, D)
    y = jnp.einsum('bd,bnd->bn', W, x)           # context outputs (B, N)

    x_query = jr.normal(keys[2], (B, D))         # query input (B, D)
    y_query = jnp.einsum('bd,bd->b', W, x_query) # query output (B,)

    return (
        create_input_matrix(x, y, x_query), 
        y_query
    )


def create_input_matrix(x: Array, y: Array, x_query: Array) -> Array:
    """Creates the input matrix E_τ as described in https://arxiv.org/abs/2507.16003.

    Args:
        x: context inputs (B, N, D)
        y: context outputs (B, N)
        x_query: query input (B, D)

    Returns: 
        Input matrix (B, N+1, D+1)

    """
    B = x.shape[0]

    context = jnp.concatenate([x, y[..., None]], axis=-1)  # (B, N, D+1)
    query = jnp.concatenate([
        x_query[:, None, :],        # (B, 1, D)
        jnp.zeros((B, 1, 1))        # (B, 1, 1) - placeholder for output
    ], axis=-1)                     # (B, 1, D+1)

    return jnp.concatenate([context, query], axis=1)  # (B, N+1, D+1)
