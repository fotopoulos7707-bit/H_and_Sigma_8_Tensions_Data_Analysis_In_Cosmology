# likelihood.py
"""
Distance-modulus theory curve and the combined log-posterior used by emcee.
Kept separate from main.py so the sampler's objective function can be
imported/tested independently of the run configuration and plotting code.
"""
import numpy as np
from scipy.linalg import cho_solve

import data_loader as dl


def mu_theory(z_arr, theta_cosmo, H0, active_E2, n_grid=600):
    """Theoretical SNe Ia distance modulus mu(z) for a given E(z)^2 model."""
    z_max = z_arr.max() * 1.05
    z_grid = np.linspace(0.0, z_max, n_grid)
    inv_E = 1.0 / np.sqrt(np.array([active_E2(zi, theta_cosmo) for zi in z_grid]))
    dz = np.diff(z_grid)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (inv_E[:-1] + inv_E[1:]) * dz)]) * (dl.C_LIGHT / H0)
    return 5.0 * np.log10((1.0 + z_arr) * np.interp(z_arr, z_grid, chi)) + 25.0


def resolve_active_E2(chosen_model, E2_func, fixed_delta, fixed_alpha):
    """
    Returns the E2 function actually used in the likelihood, accounting for
    models that fix one or more shape parameters (delta, alpha) rather than
    sampling them.
    """
    if chosen_model == "Custom_Universal":
        return lambda z, t: E2_func(z, t, (fixed_delta, fixed_alpha))
    elif chosen_model == "param_Hubble_fixed_delta":
        return lambda z, t: E2_func(z, t, (fixed_delta,))
    elif chosen_model == "param_Hubble_fixed_alpha":
        return lambda z, t: E2_func(z, t, (fixed_alpha,))
    else:
        return E2_func


def make_log_posterior(chosen_model, E2_func, prior_func, fixed_delta, fixed_alpha):
    """
    Builds a log_posterior(params, ...) closure bound to a specific model
    configuration, so main.py doesn't need to pass CHOSEN_MODEL/E2_func/etc.
    through emcee's args on every call.
    """
    active_E2 = resolve_active_E2(chosen_model, E2_func, fixed_delta, fixed_alpha)

    def log_posterior(params, z_cmb, mu_obs, C_fac, Cinv_ones, scalar_denom, hz_datasets):
        H0 = params[0]
        theta_cosmo = tuple(params[1:])

        lp = prior_func(H0, theta_cosmo)
        if not np.isfinite(lp):
            return -np.inf

        ll = 0.0

        # SNIa Pantheon+SH0ES Marginalized Likelihood
        if mu_obs is not None:
            residual = mu_obs - mu_theory(z_cmb, theta_cosmo, H0, active_E2)
            Cinv_r = cho_solve(C_fac, residual, check_finite=False)
            a = float(residual @ Cinv_r)
            b = float(Cinv_ones @ residual)
            ll += -0.5 * (a - b * b / scalar_denom)

        # H(z) Cosmic Chronometers, BAO, & Clustering Likelihoods
        for (z_arr, H_obs, sigma_H) in hz_datasets:
            H_th = H0 * np.sqrt(np.array([active_E2(z, theta_cosmo) for z in z_arr]))
            ll += -0.5 * np.sum(((H_obs - H_th) / sigma_H) ** 2)

        return lp + ll

    return log_posterior