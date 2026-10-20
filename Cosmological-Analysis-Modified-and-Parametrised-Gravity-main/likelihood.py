# likelihood.py
"""
Distance-modulus theory curve and the combined log-posterior used by emcee.
Kept separate from main.py so the sampler's objective function can be
imported/tested independently of the run configuration and plotting code.

Parameter vector layout (emcee):  [H0, *theta_parameterisation, *theta_fR]
    H0      : present-day Hubble constant (E^2(0) = 1 always; see fR.py)
    theta_p : parameterisation parameters (parameterisations.py); in f(R) runs
              Omega_m is excluded (held at FR_OMEGA_M_ANCHOR, see fR.py)
    theta_fR: extra f(R) parameters (fR.py); empty for FR_MODEL = "GR"
"""
import numpy as np
from scipy.linalg import cho_solve

import data_loader as dl
import parameterisations as P
import fR


def mu_theory(z_arr, E2_of_z, H0, n_grid=600):
    """
    Theoretical SNe Ia distance modulus mu(z) for a vectorised E^2(z).
    Returns None if E^2 is non-finite or non-positive anywhere on the grid.
    """
    z_max = z_arr.max() * 1.05
    z_grid = np.linspace(0.0, z_max, n_grid)
    E2 = np.asarray(E2_of_z(z_grid), dtype=float)
    if not (np.all(np.isfinite(E2)) and np.all(E2 > 0.0)):
        return None
    inv_E = 1.0 / np.sqrt(E2)
    dz = np.diff(z_grid)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (inv_E[:-1] + inv_E[1:]) * dz)]) * (dl.C_LIGHT / H0)
    return 5.0 * np.log10((1.0 + z_arr) * np.interp(z_arr, z_grid, chi)) + 25.0


def resolve_active_E2(chosen_model, E2_func, fixed_delta, fixed_alpha):
    """
    Returns the E2(z, theta) function actually used for a parameterisation,
    accounting for models that fix one or more shape parameters (delta, alpha)
    rather than sampling them.
    """
    if chosen_model == "Custom_Universal":
        return lambda z, t: E2_func(z, t, (fixed_delta, fixed_alpha))
    elif chosen_model == "param_Hubble_fixed_delta":
        return lambda z, t: E2_func(z, t, (fixed_delta,))
    elif chosen_model == "param_Hubble_fixed_alpha":
        return lambda z, t: E2_func(z, t, (fixed_alpha,))
    else:
        return E2_func


def _get_fr_config(fr_model):
    if fr_model not in fR.FR_REGISTRY:
        raise ValueError(f"Unknown f(R) model '{fr_model}'. Options: {list(fR.FR_REGISTRY)}")
    return fR.FR_REGISTRY[fr_model]


def make_theta_expander(parameterisation, fr_model, omega_m_anchor):
    """
    Returns (expand, n_sampled): expand(theta_sampled) -> full theta_cosmo tuple.
    In f(R) runs Omega_m is not sampled (gauge-fixed, see fR.py); expand() puts
    FR_OMEGA_M_ANCHOR back at the position the parameterisation expects.
    """
    pcfg = P.PARAMETERISATION_REGISTRY[parameterisation]
    n_full = len(pcfg["labels"]) - 1
    idx = pcfg["omega_m_index"]
    if fr_model == "GR" or idx is None:
        return (lambda th: tuple(th)), n_full
    return (lambda th: tuple(th[:idx]) + (omega_m_anchor,) + tuple(th[idx:])), n_full - 1


def make_background_builder(parameterisation, fr_model, fixed_delta, fixed_alpha, fr_z_init,
                            omega_m_anchor=0.3, fr_fixed=None):
    """
    Returns build(theta_p, theta_fR) -> fR.Background (or None if the f(R)
    integration fails).  theta_p / theta_fR = SAMPLED parameters only (fr_fixed
    parameters are inserted here).  Shared by
    the likelihood and the plotting scripts so the background used for plots is
    always the one used in the fit.
    """
    pcfg = P.PARAMETERISATION_REGISTRY[parameterisation]
    active_E2 = resolve_active_E2(parameterisation, pcfg["E2_func"], fixed_delta, fixed_alpha)
    fcfg = _get_fr_config(fr_model)
    derivs_of = fcfg["derivs"]
    omega_m_func = pcfg["omega_m_func"]
    expand, _ = make_theta_expander(parameterisation, fr_model, omega_m_anchor)
    expand_fr, _ = fR.make_fr_expander(fr_model, fr_fixed)

    if derivs_of is None:                       # GR: parameterisation IS the background
        def build(theta_p, theta_fr=()):
            theta = expand(theta_p)
            return fR.Background(lambda z: active_E2(z, theta), 1.0)
        return build

    if omega_m_func is None:
        raise ValueError(f"Parameterisation '{parameterisation}' has no conserved-matter "
                         f"Omega_m and cannot be combined with f(R) model '{fr_model}'.")

    def build(theta_p, theta_fr):
        theta = expand(theta_p)
        return fR.solve_background(lambda z: active_E2(z, theta), omega_m_func(theta),
                                   derivs_of(expand_fr(theta_fr)), fr_z_init)
    return build


def make_log_posterior(parameterisation, fr_model, fixed_delta, fixed_alpha, fr_z_init,
                       omega_m_anchor=0.3, fr_fixed=None):
    """
    Builds a log_posterior(params, ...) closure bound to a specific
    (parameterisation, f(R) model) pair, so main.py doesn't need to pass them
    through emcee's args on every call.
    """
    pcfg = P.PARAMETERISATION_REGISTRY[parameterisation]
    fcfg = _get_fr_config(fr_model)
    prior_func = pcfg["prior_func"]
    fr_prior = fcfg["prior_func"]
    expand, n_p = make_theta_expander(parameterisation, fr_model, omega_m_anchor)
    expand_fr, _ = fR.make_fr_expander(fr_model, fr_fixed)
    build = make_background_builder(parameterisation, fr_model, fixed_delta, fixed_alpha, fr_z_init,
                                    omega_m_anchor, fr_fixed)

    def log_posterior(params, z_cmb, mu_obs, C_fac, Cinv_ones, scalar_denom, hz_datasets):
        H0 = params[0]
        theta_p = tuple(params[1:1 + n_p])
        theta_fr = tuple(params[1 + n_p:])

        lp = prior_func(H0, expand(theta_p)) + fr_prior(expand_fr(theta_fr))
        if not np.isfinite(lp):
            return -np.inf

        bg = build(theta_p, theta_fr)
        if bg is None:
            return -np.inf
        E2_of_z = bg.E2

        ll = 0.0

        # SNIa Pantheon+SH0ES Marginalized Likelihood
        if mu_obs is not None:
            mu_th = mu_theory(z_cmb, E2_of_z, H0)
            if mu_th is None:
                return -np.inf
            residual = mu_obs - mu_th
            Cinv_r = cho_solve(C_fac, residual, check_finite=False)
            a = float(residual @ Cinv_r)
            b = float(Cinv_ones @ residual)
            ll += -0.5 * (a - b * b / scalar_denom)

        # H(z) Cosmic Chronometers, BAO, & Clustering Likelihoods
        for (z_arr, H_obs, sigma_H) in hz_datasets:
            E2 = np.asarray(E2_of_z(z_arr), dtype=float)
            if not (np.all(np.isfinite(E2)) and np.all(E2 > 0.0)):
                return -np.inf
            ll += -0.5 * np.sum(((H_obs - H0 * np.sqrt(E2)) / sigma_H) ** 2)

        return lp + ll

    return log_posterior


def suggest_alpha_start(parameterisation, fr_model, fr_fixed, fixed_delta, fixed_alpha, fr_z_init,
                        omega_m_anchor, theta_p_center, n_scan=60):
    """
    Starting value for alpha when other f(R) parameters (e.g. n) are pinned: the
    alpha, on the sign branch where f_RR > 0, whose background has u(0) closest to 1
    (physical Omega_m = Omega_m/u(0) ~ the anchor, i.e. a LCDM-like expansion).
    Only a starting point for the walkers, not a fit.  Returns None if no scanned
    alpha (|alpha| in 1e-3..10) gives a background with u(0) within a factor 2 of 1,
    i.e. the pinned parameters admit no LCDM-like matter-only expansion.
    """
    names = fR.FR_REGISTRY[fr_model]["param_names"]
    if "alpha" not in names or "alpha" in (fr_fixed or {}):
        return None
    if fr_model == "power_law":
        n = (fr_fixed or {}).get("n")
        if n is None:
            return None
        sign = np.sign(n * (n - 1.0))
        if sign == 0:
            return None
    else:
        sign = 1.0
    build = make_background_builder(parameterisation, fr_model, fixed_delta, fixed_alpha, fr_z_init,
                                    omega_m_anchor, fr_fixed)
    best, best_gap = None, np.inf
    for mag in np.logspace(-3, 1, n_scan):
        alpha = sign * mag
        bg = build(tuple(theta_p_center), (alpha,))
        if bg is not None and abs(np.log(bg.u0)) < best_gap:
            best, best_gap = alpha, abs(np.log(bg.u0))
    return best if best_gap < np.log(2.0) else None


def suggest_n_start(parameterisation, fr_fixed, fixed_delta, fixed_alpha, fr_z_init,
                    omega_m_anchor, theta_p_center, n_scan=60):
    """
    power_law with alpha pinned and n sampled: n on the stable branch (alpha<0 -> 0<n<1,
    alpha>0 -> 1<n<2) whose background has u(0) closest to 1.  Start point only.
    Returns None if no scanned n gives u(0) within a factor 2 of 1.
    """
    alpha = (fr_fixed or {}).get("alpha")
    if alpha is None or alpha == 0:
        return None
    lo, hi = fR.FR_PRIOR_BOUNDS["n_power"]
    cand = np.linspace(max(lo, 0.0) + 0.01, 0.99, n_scan) if alpha < 0 else np.linspace(1.01, hi - 0.01, n_scan)
    build = make_background_builder(parameterisation, "power_law", fixed_delta, fixed_alpha, fr_z_init,
                                    omega_m_anchor, fr_fixed)
    best, best_gap = None, np.inf
    for n in cand:
        bg = build(tuple(theta_p_center), (n,))
        if bg is not None and abs(np.log(bg.u0)) < best_gap:
            best, best_gap = n, abs(np.log(bg.u0))
    return best if best_gap < np.log(2.0) else None
