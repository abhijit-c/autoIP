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
from autoip.gaussian import Gaussian, sample, kl_divergence

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.random as random
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import Partial

seed = random.PRNGKey(0)


@jax.jit
def kld_ip(p, q):
    def general_toy_linear(x, p=0.25, q=0.25):
        return jnp.array([p * x[0] + q * x[1], q * x[0] + (1 - p) * x[1]])

    F = lambda x: general_toy_linear(x, p=p, q=q)

    true_x = jnp.array([0.1, 0.2])

    mu_prior = jnp.array([0.0, 0.0])
    G_prior = jnp.eye(2)
    L_prior = jsp.linalg.cholesky(G_prior, lower=True)
    P_prior = Gaussian(mean=mu_prior, cov=G_prior, L=L_prior)

    mu_obs_err = jnp.array([0.0, 0.0])
    G_obs_err = jnp.eye(2) * jnp.linalg.norm(F(true_x)) * 0.01
    L_obs_err = jsp.linalg.cholesky(G_obs_err, lower=True)
    P_ops_err = Gaussian(mean=mu_obs_err, cov=G_obs_err, L=L_obs_err)

    key, subkey = random.split(seed)
    y = F(true_x) + sample(P_ops_err, subkey)

    ip = TIIP(P_prior=P_prior, P_obs=P_ops_err, F=F, y=y)
    P_post = assemble_linear_posterior(ip)
    return kl_divergence(P_prior, P_post)


p_vals = jnp.linspace(0.0, 1.0, 22)[1:-1]
q_vals = jnp.linspace(0.0, 1.0, 22)[1:-1]
p_grid, q_grid = jnp.meshgrid(p_vals, q_vals)

kld_vals = jnp.zeros_like(p_grid)
for i in range(p_grid.shape[0]):
    for j in range(p_grid.shape[1]):
        kld_vals.at[i, j].set(kld_ip(p_grid[i, j], q_grid[i, j]))

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

fig, ax = plt.subplots()
C = ax.contourf(p_grid, q_grid, kld_vals, 100, cmap="viridis")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$q$")
ax.set_title(
    r"$D_{\mathrm{KL}}(\mathcal{N}(\mu_{\mathrm{prior}}, \Sigma_{\mathrm{prior}}) ||"
    r" \mathcal{N}(\mu_{\mathrm{post}}, \Sigma_{\mathrm{post}}))$"
)
fig.colorbar(C)
plt.show()
