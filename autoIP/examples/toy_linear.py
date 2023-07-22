import os
import sys

autoip_path = os.path.abspath("../../")
if not autoip_path in sys.path:
    sys.path.append(autoip_path)

import autoip
import jax
import jax.numpy as jnp
from jax.tree_util import Partial


def general_toy_linear(A, b, x):
    return A @ x + b


A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
b = jnp.array([1.0, 2.0])
toy_linear = Partial(general_toy_linear, A, b)
