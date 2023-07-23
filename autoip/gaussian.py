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
    reasons of performance. It must be the case that :math:`LL^T = \Sigma` and
    that :math:`L` is lower triangular.

    Attributes:
        mean: The mean of the Gaussian distribution.
        cov: The covariance of the Gaussian distribution.
        L: The Cholesky factorization of the covariance.
    """

    mean: chex.ArrayDevice
    cov: chex.ArrayDevice
    L: chex.ArrayDevice


def gaussian_sample(G: Gaussian, key: PRNGKey) -> Array:
    """Sample from the given Gaussian distribution.

    This is given by the formula :math:`\mu + L\epsilon` where :math:`\epsilon
    \sim \mathcal{N}(0, I)`.

    Args:
        G: The Gaussian to sample from.
        key: The PRNG key to use for sampling.
    """

    return G.mean + G.L @ jax.random.normal(key, G.mean.shape)


def gaussian_un_logpdf(G: Gaussian, x: ArrayLike) -> float:
    """Compute the unnormalized log-probability of the given Gaussian distribution at a
    point.

    This is given by the expression

    .. math::
        \\log \\hat{p}(x) = -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu)

    Args:
        G: The Gaussian distribution.
        x: The point at which to evaluate the unnormalized log-probability.
    """
    innov = x - G.mean
    return -0.5 * jnp.dot(innov, jsp.linalg.cho_solve((G.L, True), innov))
