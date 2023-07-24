from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import chex

from jax.typing import ArrayLike
from jax import Array
from autoip.notation import PRNGKey


@chex.dataclass
class Gaussian:
    """Dataclass representing a Gaussian distribution.

    In addition to the mean and covariance, the Cholesky factorization of the
    covariance is also stored, i.e. :math:`\Sigma = LL^T`. This is done for
    reasons of performance. It must be the case that :math:`LL^T \\approx \\Sigma` and
    that :math:`L` is lower triangular.

    Attributes:
        mean: The mean of the Gaussian distribution.
        cov: The covariance of the Gaussian distribution.
        L: The Cholesky factorization of the covariance.
    """

    mean: chex.ArrayDevice
    cov: chex.ArrayDevice
    L: chex.ArrayDevice


def precision_action(G: Gaussian, x: ArrayLike) -> Array:
    """Compute the action of the precision matrix (covariance inverse) of the given
    Gaussian distribution on a point.

    As each Gaussian is assumed to have an accompanying Cholesky factor, this action
    is computed using a Cholesky accelerated triangular solve via the Jax function
    :func:`jax.scipy.linalg.cho_solve`.

    Args:
        G: The Gaussian distribution.
        x: The point at which to evaluate the precision action.
    """
    return jsp.linalg.cho_solve((G.L, True), x)


def sample(G: Gaussian, key: PRNGKey) -> Array:
    """Sample from the given Gaussian distribution.

    This is given by the formula :math:`\mu + L\epsilon` where :math:`\epsilon
    \sim \mathcal{N}(0, I)`.

    Args:
        G: The Gaussian to sample from.
        key: The PRNG key to use for sampling.
    """

    return G.mean + G.L @ jax.random.normal(key, G.mean.shape)


def logpdf(G: Gaussian, x: ArrayLike, normalized=False) -> float:
    """Compute the log-probability of the given Gaussian distribution at a point.

    This is given by the expression

    .. math::
        \\log \\p(x) = -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu)
        - \\frac{k}{2} \\log (2\\pi |\\Sigma|)

    Args:
        G: The Gaussian distribution.
        x: The point at which to evaluate.
        normalized: Whether to return the normalized log-probability.
    """
    innov = x - G.mean
    p = -0.5 * jnp.dot(innov, jsp.linalg.cho_solve((G.L, True), innov))
    if normalized:
        p -= G.L.shape[0]*jnp.log(jnp.prod(jnp.diag(G.L)))
    return p