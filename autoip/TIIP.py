from __future__ import annotations
import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
from autoip.notation import Operator, LinearOperator
from autoip.gaussian import Gaussian, logpdf, precision_action, sample
from autoip.utils import LinearOperator2Matrix
from jax import Array
from jax.tree_util import Partial
from jax.typing import ArrayLike


@chex.dataclass
class TIIP:
    """Dataclass representing a Time Independent Inverse Problem.

    Args:
        P_prior: The prior distribution.
        P_obs: The observation distribution.
        F: The parameter-to-observable map.
        y: The observation.
    """

    P_prior: Gaussian
    P_obs: Gaussian
    F: Operator
    y: chex.ArrayDevice


def linear_Hessian(
    ip: TIIP,
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
        ip: The inverse problem represented by a :class:`TIIP` dataclass.
        x: Point to act on.

    Returns:
        The Hessian action at the given point.
    """
    P_prior, P_obs, F = ip.P_prior, ip.P_obs, ip.F
    # TODO: Inefficient to recompute this every time.
    Ft_tup = jax.linear_transpose(F, x)
    Ft = lambda x: Ft_tup(x)[0]
    return Ft(precision_action(P_obs, F(x))) + precision_action(P_prior, x)


def linear_MAP(
    ip: TIIP,
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
        ip: The inverse problem represented by a :class:`TIIP` dataclass.

    Keyword Args:
        **kwargs: Keyword arguments to pass to :func:`jax.scipy.sparse.linalg.cg`.

    Returns:
        The MAP estimate.

    """
    P_prior, P_obs, F, y = ip.P_prior, ip.P_obs, ip.F, ip.y
    Ft_tup = jax.linear_transpose(F, P_prior.mean)
    Ft = lambda x: Ft_tup(x)[0]
    rhs = Ft(precision_action(P_obs, y)) + precision_action(P_prior, P_prior.mean)
    Hv = Partial(linear_Hessian, ip)
    MAP, info = jsp.sparse.linalg.cg(Hv, rhs, **kwargs)
    return MAP


def assemble_linear_posterior(ip: TIIP, **kwargs) -> Gaussian:
    """Assemble the posterior distribution for a linear inverse problem.

    This is glue code that assembles the posterior distribution for a linear inverse
    problem from the MAP estimate and by explicitly computing the posterior covariance
    and its Cholesky factorization from the linear Hessian of the cost functional, i.e.
    :math:`\\Sigma_{\\rm post} = H^{-1}`.

    .. warning::
        Constructing the posterior distribution in this way is not recommended for
        even medium sized problems, as it involes :math:`N` Hessian solves for
        :math:`H \\in \\mathbb{R}^{N \\times N}`.

    Args:
        ip: The inverse problem represented by a :class:`TIIP` dataclass.

    Keyword Args:
        **kwargs: Keyword arguments to pass to :func:`jax.scipy.sparse.linalg.cg` for
            both the MAP estimate and in the construction of the posterior covariance
            from linear Hessian solves.
    """
    mean = linear_MAP(ip, **kwargs)
    Hess_mv = Partial(linear_Hessian, ip)
    Hess_inv_mv = lambda x: jsp.sparse.linalg.cg(Hess_mv, x, **kwargs)[0]
    cov = LinearOperator2Matrix(Hess_inv_mv, mean.shape[0])
    L = jsp.linalg.cholesky(cov, lower=True)
    return Gaussian(mean=mean, cov=cov, L=L)


def IPCost(
    ip: TIIP,
    theta: ArrayLike,
) -> float:
    """Compute the cost functional for an inverse problem.

    This is given by the expression

    .. math::
        \\hat{C}(\\theta) =
        -\\log \\hat{p}(y | F(\\theta)) - \\log \\hat{p}(\\theta)

    where :math:`\\hat{p}(y | F(\\theta))` is the unnormalized likelihood of the
    observation given the parameter-to-observable map evaluated at :math:`\\theta` and
    :math:`\\hat{p}(\\theta)` is the unnormalized prior distribution.

    Args:
        ip: The inverse problem represented by a :class:`TIIP` dataclass.
        theta: The point at which to evaluate the cost functional.
    """
    P_prior, P_obs, F, y = ip.P_prior, ip.P_obs, ip.F, ip.y
    return -logpdf(P_obs, F(theta) - y) + -logpdf(P_prior, theta)


# TODO: Register this as a custom derivative to IPCost.
def gradIPCost(
    ip: TIIP,
    theta: ArrayLike,
):
    """Compute the gradient of the cost functional for an inverse problem.

    This is given by the expression

    .. math::
        \\nabla \\hat{C}(\\theta) =
        J^T \\Sigma_{\\rm obs}^{-1} \\left( F(\\theta) - y \\right) +
        \\Sigma_{\\rm prior}^{-1} \\left( \\theta - \\mu_{\\rm prior} \\right)

    where :math:`\\Sigma_{\\rm obs}` and :math:`\\Sigma_{\\rm prior}` are the covariance
    matrices of the observation and prior distributions respectively, :math:`F` is the
    parameter-to-observable map, :math:`J` is the Jacobian of the
    parameter-to-observable map and particularily :math:`J^T` is the adjoint of the
    parameter-to-observable map, :math:`y` is the observation, and :math:`\\mu_{\\rm
    prior}` is the mean of the prior distribution.

    Args:
        ip: The inverse problem represented by a :class:`TIIP` dataclass.
        theta: The point at which to evaluate the gradient of the cost functional.
    """
    P_prior, P_obs, F, y = ip.P_prior, ip.P_obs, ip.F, ip.y
    Jt = jax.vjp(F, theta)[1]
    return Jt(precision_action(P_obs, F(theta) - y)) + precision_action(
        P_prior, theta - P_prior.mean
    )


def nonlinear_MAP(
    ip: TIIP,
):
    """Compute the maximum a posteriori (MAP) estimate for a nonlinear inverse problem.

    This is given by solving the optimization problem

    .. math::
        \\hat{\\theta} = \\arg\\min_{\\theta} \\hat{C}(\\theta)

    where :math:`\\hat{C}(\\theta)` is the cost functional for the inverse problem. This
    is done using Jax's :func:`jaxopt.LBFGS` function.

    Args:
        ip: The inverse problem represented by a :class:`TIIP` dataclass.

    Returns:
        Returns (params, state) where params is the MAP estimate and state is the state
        of the optimizer.
    """
    F = lambda t: IPCost(ip, t)
    dF = lambda t: gradIPCost(ip, t)
    F_vag = lambda t: (F(t), dF(t))
    solver = jaxopt.LBFGS(F_vag, value_and_grad=True)
    return solver.run(ip.P_prior.mean)
