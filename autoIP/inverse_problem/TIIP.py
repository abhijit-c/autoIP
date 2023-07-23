import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
from autoip.utils.notation import Operator, LinearOperator, Gaussian
from autoip.utils.gaussian import Gaussian, gaussian_un_logpdf
from jax import Array
from jax.tree_util import Partial
from jax.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def linear_Hessian(
    P_prior: Gaussian,
    P_obs: Gaussian,
    F: LinearOperator,
    Ft: LinearOperator,
    x: ArrayLike,
) -> Array:
    G_obs_inv = lambda x: jsp.linalg.cho_solve((P_obs.L, True), x)
    G_prior_inv = lambda x: jsp.linalg.cho_solve((P_prior.L, True), x)
    return Ft(G_obs_inv(F(x))) + G_prior_inv(x)


def linear_MAP(
    P_obs: Gaussian,
    P_prior: Gaussian,
    F: LinearOperator,
    Ft: LinearOperator,
    y: ArrayLike,
) -> Array:
    G_obs_inv = lambda x: jsp.linalg.cho_solve((P_obs.L, True), x)
    G_prior_inv = lambda x: jsp.linalg.cho_solve((P_prior.L, True), x)

    rhs = Ft(G_obs_inv(y)) + G_prior_inv(P_prior.mean)

    Hv = Partial(linear_Hessian, P_prior, P_obs, F, Ft)
    MAP, info = jsp.sparse.linalg.cg(Hv, rhs)
    return MAP


def solve_linearized_inverse_problem(
    mu_obs: Gaussian,
    mu_prior: Gaussian,
    F: LinearOperator,
    y: ArrayLike,
):
    raise NotImplementedError("TODO: implement solve_linearized_inverse_problem")


def J(
    mu_obs: Gaussian,
    mu_prior: Gaussian,
    F: LinearOperator,
    y: ArrayLike,
    theta: ArrayLike,
) -> float:
    innov = F(theta) - y
    return gaussian_un_logpdf(mu_obs, innov) + gaussian_un_logpdf(mu_prior, theta)


def solve_nonlinear_inverse_problem(
    mu_obs: Gaussian,
    mu_prior: Gaussian,
    F: LinearOperator,
    y: ArrayLike,
):
    raise NotImplementedError("TODO: implement solve_nonlinear_inverse_problem")
    Jred = Partial(J, mu_obs, mu_prior, F, y)
