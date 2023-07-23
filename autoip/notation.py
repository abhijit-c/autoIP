from __future__ import annotations

from typing import TypeAlias, Callable, Union
import jax
from jax import Array
from jax.typing import ArrayLike

# Some useful type aliases. NOTE: All of these must be duplicated in the conf.py file.

Operator: TypeAlias = Callable[[ArrayLike], Array]
r"""Represents an operator from :math:`\mathbb{R}^n \to \mathbb{R}^m`."""

LinearOperator: TypeAlias = Operator
r"""Represents a linear operator from :math:`\mathbb{R}^n \to \mathbb{R}^m`."""

PRNGKey: TypeAlias = Union[jax.random.KeyArray, Array]
"""Represents a PRNG key passed to JAX random functions."""
