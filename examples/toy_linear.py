try:
    import autoip
except ModuleNotFoundError:
    import os
    import sys

    autoip_path = os.path.abspath("../")
    if not autoip_path in sys.path:
        sys.path.append(autoip_path)
    import autoip

import jax
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import Partial

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

seed = random.PRNGKey(0)


def general_toy_linear(A, x):
    return A @ x


A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
toy_linear = Partial(general_toy_linear, A)

true_x = jnp.array([0.1, 0.2])

mu_prior = jnp.array([0.0, 0.0])
G_prior = jnp.eye(2)
P_prior = tfd.MultivariateNormalFullCovariance(loc=mu_prior, covariance_matrix=G_prior)

mu_obs_err = jnp.array([0.0, 0.0])
G_obs_err = jnp.eye(2) * 0.01
P_obs_err = tfd.MultivariateNormalFullCovariance(
    loc=mu_obs_err, covariance_matrix=G_obs_err
)

key, subkey = random.split(seed)
y = toy_linear(true_x) + P_obs_err.sample(seed=subkey)

from autoip.inverse_problem.TIIP import solve_linear_inverse_problem

P_post = solve_linear_inverse_problem(P_obs_err, P_prior, toy_linear, y)
