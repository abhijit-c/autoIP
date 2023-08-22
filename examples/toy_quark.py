try:
    import autoip
except ModuleNotFoundError:
    import os
    import sys

    autoip_path = os.path.abspath("../")
    if not autoip_path in sys.path:
        sys.path.append(autoip_path)
    import autoip

from autoip.TIIP import TIIP, assemble_linear_posterior
from autoip.gaussian import Gaussian, sample, logpdf

import jax

jax.config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import Partial

import blackjax

import matplotlib
import matplotlib.pyplot as plt


plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
import corner
import numpy as np

key = random.PRNGKey(0)


@jax.jit
def toy_nuclear(p, x):
    N_1, alpha_1, beta_1 = p[0], p[1], p[2]
    N_2, alpha_2, beta_2 = p[3], p[4], p[5]
    return (
        N_1 * x**alpha_1 * (1 - x) ** beta_1 + N_2 * x**alpha_2 * (1 - x) ** beta_2
    )


p_true = jnp.array([1.0, 0.5, 2.5, 0.25, 0.1, 3.0])
F = Partial(toy_nuclear, x=jnp.linspace(0.1, 0.9, 100))

mu_prior = jnp.zeros_like(p_true)
G_prior = jnp.eye(p_true.shape[0])
L_prior = jsp.linalg.cholesky(G_prior, lower=True)
P_prior = Gaussian(mean=mu_prior, cov=G_prior, L=L_prior)

mu_obs_err = jnp.zeros_like(F(p_true))
G_obs_err = jnp.diag(F(p_true) * 0.01)
L_obs_err = jsp.linalg.cholesky(G_obs_err, lower=True)
P_obs_err = Gaussian(mean=mu_obs_err, cov=G_obs_err, L=L_obs_err)

key, subkey = random.split(key)
obs = F(p_true) + sample(P_obs_err, subkey)

logdensity = lambda theta: logpdf(P_obs_err, F(theta) - obs) + logpdf(P_prior, theta)

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.eye(p_true.shape[0])
nuts = blackjax.nuts(logdensity, step_size, inverse_mass_matrix)

# Initialize the state
key, subkey = random.split(key)
init_pos = sample(P_prior, subkey)
state = nuts.init(init_pos)


# iterate
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


key, subkey = random.split(key)
states = inference_loop(subkey, jax.jit(nuts.step), state, 1000)


# Plot the results
labels = [
    r"$N_1$",
    r"$\alpha_1$",
    r"$\beta_1$",
    r"$N_2$",
    r"$\alpha_2$",
    r"$\beta_2$",
]
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()
for i in range(6):
    axes[i].plot(states.position[:, i])
    axes[i].axhline(p_true[i], color="black", linestyle="--")
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(labels[i])
fig.savefig("plots/toy_nuclear_trace.png", dpi=512, bbox_inches="tight")

fig = corner.corner(np.array(states.position), labels=labels)
fig.savefig("plots/toy_nuclear_corner.png", dpi=512, bbox_inches="tight")
