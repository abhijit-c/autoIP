from __future__ import annotations
import jax
import jax.numpy as jnp
from autoip.notation import LinearOperator
from jax import Array


def LinearOperator2Matrix(F: LinearOperator, domain_dim: int) -> Array:
    """Convert a linear operator to a matrix.

    Args:
        F: The linear operator.
        domain_dim: Dimension of the domain of the operator.


    Returns:
        The matrix representation of the linear operator.
    """
    return jax.vmap(F, 1, 1)(jnp.eye(domain_dim))
