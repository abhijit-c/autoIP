import chex
import jax
import jax.numpy as jnp
import jaxopt
from autoip.utils.notation import Operator, LinearOperator, MVN
from jax import Array
from jax.tree_util import Partial
from jax.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def linear_Hessian(
    G_prior_inv: Operator,
    G_obs_inv: Operator,
    y: ArrayLike,
    F: LinearOperator,
    Ft: LinearOperator,
    x: ArrayLike,
) -> Array:
    return Ft(G_obs_inv(F(x))) + G_prior_inv(x)


def linear_MAP(
    mu_obs: MVN, mu_prior: MVN, F: LinearOperator, Ft: LinearOperator, y: ArrayLike
) -> Array:
    G_obs = mu_obs.covariance()
    G_prior = mu_prior.covariance()
    G_post = jnp.linalg.inv(G_prior + Ft(G_obs) @ F(G_obs))
    mu_post = G_post @ (Ft(G_obs) @ y + G_prior @ mu_prior.mean())
    return mu_post


def solve_linearized_inverse_problem(
    mu_obs: MVN,
    mu_prior: MVN,
    F: LinearOperator,
    y: ArrayLike,
):
    raise NotImplementedError("TODO: implement solve_linearized_inverse_problem")


def J(
    mu_obs: MVN, mu_prior: MVN, F: LinearOperator, y: ArrayLike, theta: ArrayLike
) -> float:
    return mu_obs.log_prob(F(theta) - y) + mu_prior.log_prob(theta)


def solve_nonlinear_inverse_problem(
    mu_obs: MVN,
    mu_prior: MVN,
    F: LinearOperator,
    y: ArrayLike,
):
    raise NotImplementedError("TODO: implement solve_nonlinear_inverse_problem")
    Jred = Partial(J, mu_obs, mu_prior, F, y)
