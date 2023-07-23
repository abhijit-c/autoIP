import jax
import jax.numpy as jnp
import jax.scipy as jsp
import chex

from jax.typing import ArrayLike
from jax import Array


@chex.dataclass
class Gaussian:
    r"""
    Dataclass representing a Gaussian distribution, ensuring that the covariance
    additionally is provided in Cholesky factorized form, i.e.

    .. math::
        \\mathcal{N}(\\mu, \\Sigma) = \\mathcal{N}(\\mu, LL^T)
    """
    mean: chex.ArrayDevice
    cov: chex.ArrayDevice
    L: chex.ArrayDevice


def gaussian_sample(key: ArrayLike, G: Gaussian) -> Array:
    r"""
    Sample from the given Gaussian distribution :math:`\\mu + L \\epsilon` where
    :math:`\\epsilon \\sim \\mathcal{N}(0, I)`.
    """

    return G.mean + G.L @ jax.random.normal(key, G.mean.shape)


def gaussian_un_logpdf(G: Gaussian, x: ArrayLike) -> float:
    r"""
    Compute the unnormalized log-probability of the given Gaussian distribution
    at the given point. This is given by

    .. math::
        \\log \\hat{p}(x) = -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu)
    """
    innov = x - G.mean
    return -0.5 * jnp.dot(innov, jsp.linalg.cho_solve((G.L, True), innov))
