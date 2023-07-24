from __future__ import annotations
import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
from autoip.notation import Operator, LinearOperator
from autoip.gaussian import Gaussian, un_logpdf, precision_action
from jax import Array
from jax.tree_util import Partial
from jax.typing import ArrayLike


def linear_Hessian(
    P_prior: Gaussian,
    P_obs: Gaussian,
    F: LinearOperator,
    Ft: LinearOperator,
    x: ArrayLike,
) -> Array:
    """Compute the Hessian action of the cost functional for a linear
    parameter-to-observable map.

    This is given by the expression

    .. math::

        H(x) = F^T \\Sigma_{\\rm obs}^{-1} F(x) + \\Sigma_{\\rm prior}^{-1} x

    where :math:`\\Sigma_{\\rm obs}` and :math:`\\Sigma_{\\rm prior}` are the covariance
    matrices of the observation and prior distributions respectively, :math:`F` is the
    linear parameter-to-observable map.

    Args:
        P_prior: The prior distribution.
        P_obs: The observation distribution.
        F: The linear parameter-to-observable map.
        Ft: The adjoint of the linear parameter-to-observable map.
        x: The point at which to evaluate the Hessian action.

    Returns:
        The Hessian action at the given point.
    """
    return Ft(precision_action(P_obs, F(x))) + precision_action(P_prior, x)


def linear_MAP(
    P_obs: Gaussian,
    P_prior: Gaussian,
    F: LinearOperator,
    Ft: LinearOperator,
    y: ArrayLike,
    **kwargs,
) -> Array:
    """Compute the maximum a posteriori (MAP) estimate for a linear inverse problem.

    This is given by the closed form expression

    .. math::

        \\hat{x} = H^{-1} \\left( F^T \\Sigma_{\\rm obs}^{-1} y +
        \\Sigma_{\\rm prior}^{-1} \\mu_{\\rm prior} \\right)

    where :math:`\\mathcal{H}` is the Hessian of the cost functional :math:`J`,
    :math:`\\Sigma_{\\rm obs}` and :math:`\\Sigma_{\\rm prior}` are the covariance
    matrices of the observation and prior distributions respectively, :math:`\\mu_{\\rm
    prior}` is the mean of the prior distribution, and :math:`F` is the linear
    parameter-to-observable map,

    Internally, this function uses a conjugate gradient solver to solve the linear
    system :math:`H \\hat{x} = b` where :math:`b = F^T \\Sigma_{\\rm obs}^{-1} y +
    \\Sigma_{\\rm prior}^{-1} \\mu_{\\rm prior}`, as we're guaranteed that the Hessian
    is symmetric positive definite in this case. Practically, this is implemented using
    Jax's :func:`jax.scipy.sparse.linalg.cg` function which, by default, impmements
    derivatives via implicit differentiation as opposed to differentiating through
    the linear solver.

    Args:
        P_obs: The observation distribution.
        P_prior: The prior distribution.
        F: The linear parameter-to-observable map.
        Ft: The adjoint of the linear parameter-to-observable map.
        y: The observation.

    Keyword Args:
        **kwargs: Keyword arguments to pass to :func:`jax.scipy.sparse.linalg.cg`. Note:
            There is absolutely no error checking done here, so if you pass in an
            invalid keyword argument, you'll get an error from Jax.

    Returns:
        The MAP estimate.

    """
    rhs = Ft(precision_action(P_obs, y)) + precision_action(P_prior, P_prior.mean)
    Hv = Partial(linear_Hessian, P_prior, P_obs, F, Ft)
    MAP, info = jsp.sparse.linalg.cg(Hv, rhs, **kwargs)
    return MAP


def J(
    mu_obs: Gaussian,
    mu_prior: Gaussian,
    F: Operator,
    y: ArrayLike,
    theta: ArrayLike,
) -> float:
    """Compute the cost functional for an inverse problem.

    This is given by the expression

    .. math::
        \\hat{J}(\\theta) =
        ||F(\\theta) - y||_{\\Sigma_{\\rm obs}^{-1}}^2
        + ||\\theta - \\mu_{\\rm prior}||_{\\Sigma_{\\rm prior}^{-1}}^2


    where :math:`\\Sigma_{\\rm obs}` and :math:`\\Sigma_{\\rm prior}` are the covariance
    matrices of the observation and prior distributions respectively, :math:`F` is the
    parameter-to-observable map, :math:`y` is the observation, and :math:`\\mu_{\\rm
    prior}` is the mean of the prior distribution.

    Args:
        mu_obs: The observation distribution.
        mu_prior: The prior distribution.
        F: The parameter-to-observable map.
        y: The observation.
        theta: The point at which to evaluate the cost functional.
    """
    return un_logpdf(mu_obs, F(theta) - y) + un_logpdf(mu_prior, theta)
