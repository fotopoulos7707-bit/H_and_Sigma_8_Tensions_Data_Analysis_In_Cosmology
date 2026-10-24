# main.py
import os
# One BLAS/OpenMP thread per process.  Must be set BEFORE numpy is imported: with a process pool
# (N_PROCESSES > 1) multi-threaded BLAS in every worker would oversubscribe the CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing
import sys
import numpy as np
import emcee

import data_loader as dl
import fR
from run_config import get_run_config
from likelihood import (make_log_posterior, make_background_builder, make_theta_expander,
                        suggest_alpha_start, suggest_n_start)
from plot_selector import PlotContext, resolve_keys, run_plots
from results_io import save_histogram_data, write_posterior_summary

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD CONFIGURATION — SWITCH SCENARIOS HERE
# ════════════════════════════════════════════════════════════════════════════
PARAMETERISATION = "wCDM"   # Options: "LCDM", "wCDM", "wCDM_fixed_Omega_m", "wowaCDM", "CPL", "JBP", "HDE", "IDE",
                         #          "param_Hubble_fixed_delta", "param_Hubble_fixed_alpha", "Custom_Universal"
FR_MODEL         = "R_lambda_R2"  # f(R) gravity model (fR.py).  Options:
                         #   "GR"         f = R                   -> the parameterisation IS the background (old behaviour)
                         #   "power_law"  f = R + alpha R^n       -> f(R) ODE gives H(z); parameterisation sets ICs + Omega_m
                         #   "R_log_R"    f = R (1 + alpha ln R)  -> same
                         # Modified models need a parameterisation with conserved matter (not IDE / param_Hubble* / Custom_Universal).
FR_FIXED_PARAMS  = {"alpha": 1e-9}    # pin f(R) parameters instead of sampling them, e.g. {"n": 0.5} (power_law) or {"alpha": 0.4}.
                         # Names: power_law -> "alpha", "n";  R_log_R -> "alpha".  {} = sample all of them.
                         # For power_law, f_RR > 0 needs alpha<0 if 0<n<1 and alpha>0 if n>1 (n=1 is just GR): this is enforced.
FR_Z_INIT        = None  # None -> model default (40 for the ODE models; 2.5 for R_lambda_R2).  ODE models: redshift where the
                         # parameterisation sets the initial conditions (> max data z ~2.4; larger = more converged but slower;
                         # a convergence test is printed for f(R) runs).  R_lambda_R2: highest redshift where the perturbative
                         # solution is trusted (must exceed ~2.4).
FR_OMEGA_M_ANCHOR = 0.3  # f(R) runs only: Omega_m is NOT sampled (degenerate with alpha, see fR.py); this is the gauge value.
                         # The physical present-day Omega_m is reported as a derived quantity.

# ── Dataset Toggles ─────────────────────────────────────────────────────────
# Set each to True/False to include or exclude that dataset from the likelihood.
# At least one must be True, or the run will abort.
USE_SNIA    = True   # Pantheon+SH0ES Type Ia Supernovae (full covariance, marginalized M_B)
USE_CC      = True   # Cosmic Chronometers H(z)
USE_BAO     = True   # Baryon Acoustic Oscillations H(z)
USE_CLUSTER = True   # Galaxy Clustering H(z)
# ────────────────────────────────────────────────────────────────────────────

# ── Graphs made at the end of the run ───────────────────────────────────────
# Per-model plots: "corner", "traces", "potential", "omega_components", or "model" for all four.
# Plots that do not apply to the chosen model (e.g. "potential" for an f(R) run) are skipped with a message.
# [] = no graphs (numbers and chain are still saved).  Comparison plots (violin, box, hubble, contours, ...)
# need several chains: make them afterwards with  python make_plots.py  (interactive menu or --plots).
PLOTS        = ["corner", "potential", "omega_components"]   # default = what this script always did
SHOW_FIGURES = True      # open the figure windows when finished; False = only write the PDFs
# ────────────────────────────────────────────────────────────────────────────

# MANUAL ASSIGNMENTS FOR CONSTANT PARAMETERS ACROSS CUSTOM SCENARIOS
FIXED_DELTA  = 0.050                      # Used in Custom_Universal and param_Hubble_fixed_delta
FIXED_ALPHA  = 0.170                      # Used in Custom_Universal and param_Hubble_fixed_alpha

# ── Derived run tag (auto-generated — do not edit manually) ─────────────────
_active = []
if USE_SNIA:    _active.append("snia")
if USE_CC:      _active.append("cc")
if USE_BAO:     _active.append("bao")
if USE_CLUSTER: _active.append("clu")
if not _active:
    raise ValueError("All datasets are disabled. Enable at least one USE_* flag.")
RUN_MODE = "+".join(_active)   # e.g. "snia+cc+bao", "cc+bao", "snia"
# ────────────────────────────────────────────────────────────────────────────

N_WALKERS, N_BURNIN, N_STEPS = 64, 20, 100
N_PROCESSES = min(8, os.cpu_count() or 1)   # worker processes for emcee; 1 = serial.  Ryzen 7 9700X: 8 physical cores.
                                            # emcee updates half the walkers at a time, so useful workers <= N_WALKERS/2.
RANDOM_SEED = None                          # int -> reproducible chain (same result for any N_PROCESSES); None -> random

# ════════════════════════════════════════════════════════════════════════════
# VALIDATE SELECTION & INITIALIZE ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
if FR_Z_INIT is None:
    FR_Z_INIT = fR.FR_REGISTRY[FR_MODEL].get("z_init_default", 40.0)
model_config = get_run_config(PARAMETERISATION, FR_MODEL, FR_FIXED_PARAMS, FR_OMEGA_M_ANCHOR)
PLOTS = resolve_keys(PLOTS, scope="model")   # typo in PLOTS fails now, not after the MCMC run
RUN_TAG = model_config["tag"]            # names the chain file and graphs/ subfolder
n_param_theta = model_config["n_param_theta"]

log_posterior = make_log_posterior(PARAMETERISATION, FR_MODEL, FIXED_DELTA, FIXED_ALPHA, FR_Z_INIT,
                                 FR_OMEGA_M_ANCHOR, FR_FIXED_PARAMS)

# Picklable, module-level target for the process pool (log_posterior is a closure, which cannot be
# pickled).  Workers are forked after _PARALLEL_ARGS is set, so they inherit the loaded datasets.
_PARALLEL_ARGS = None
def _log_prob(params):
    return log_posterior(params, *_PARALLEL_ARGS)

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION ROUTINE
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Initializing MCMC Target: parameterisation [{PARAMETERISATION}] | f(R) model [{FR_MODEL}] "
          f"({model_config['fR_description']}) | Active datasets: [{RUN_MODE}]")

    # Load Pantheon data only if SNIa is active (Cholesky is expensive and unnecessary otherwise)
    if USE_SNIA:
        z_cmb, mu_obs, C, C_fac, Cinv_ones, scalar_denom = dl.load_pantheon_data()
    else:
        z_cmb = mu_obs = C = C_fac = Cinv_ones = scalar_denom = None

    hz_datasets = []
    if USE_CC:      hz_datasets.append(dl.load_cc_data())
    if USE_BAO:     hz_datasets.append(dl.load_bao_data())
    if USE_CLUSTER: hz_datasets.append(dl.load_clustering_data())

    ndim = len(model_config["labels"])
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    # With pinned f(R) parameters (e.g. n) the stored alpha start (tuned for the default n) may be
    # invalid, so choose alpha so that the background is LCDM-like (u(0) ~ 1).  Start point only.
    if FR_FIXED_PARAMS and FR_MODEL in ("power_law", "R_log_R"):
        free_names = [nm for nm in fR.FR_REGISTRY[FR_MODEL]["param_names"] if nm not in FR_FIXED_PARAMS]
        if "alpha" in free_names:
            col = 1 + n_param_theta + free_names.index("alpha")
            a0 = suggest_alpha_start(PARAMETERISATION, FR_MODEL, FR_FIXED_PARAMS, FIXED_DELTA, FIXED_ALPHA,
                                     FR_Z_INIT, FR_OMEGA_M_ANCHOR, model_config["p0_center"][1:1 + n_param_theta])
            if a0 is None:
                raise ValueError(f"No valid background found for FR_FIXED_PARAMS={FR_FIXED_PARAMS}; "
                                 f"this f(R) model/parameter choice gives no viable expansion history.")
            model_config["p0_center"][col] = a0
            model_config["p0_scatter"][col] = 0.05 * abs(a0)
            print(f"Starting alpha for fixed {FR_FIXED_PARAMS}: {a0:.4g} (u(0) closest to 1; start point only)")
        if FR_MODEL == "power_law" and "alpha" in FR_FIXED_PARAMS and "n" in free_names:
            col = 1 + n_param_theta + free_names.index("n")
            n0 = suggest_n_start(PARAMETERISATION, FR_FIXED_PARAMS, FIXED_DELTA, FIXED_ALPHA, FR_Z_INIT,
                                 FR_OMEGA_M_ANCHOR, model_config["p0_center"][1:1 + n_param_theta])
            if n0 is None:
                raise ValueError(f"No n gives a viable background for FR_FIXED_PARAMS={FR_FIXED_PARAMS}; "
                                 f"choose a different pinned alpha.")
            model_config["p0_center"][col] = n0
            model_config["p0_scatter"][col] = 0.02
            print(f"Starting n for fixed {FR_FIXED_PARAMS}: {n0:.4g} (u(0) closest to 1; start point only)")

    p0 = model_config["p0_center"] + model_config["p0_scatter"] * np.random.standard_normal((N_WALKERS, ndim))

    # Fail early with a useful message if the start point has zero probability (e.g. pinned f(R)
    # parameters that admit no valid expansion history), instead of emcee's generic error.
    _args = (z_cmb, mu_obs, C_fac, Cinv_ones, scalar_denom, hz_datasets)
    n_bad = sum(not np.isfinite(log_posterior(w, *_args)) for w in p0)
    if n_bad:
        # Say WHY, using the first bad walker (prior/stability, or the f(R) solver's own reason).
        w = next(w for w in p0 if not np.isfinite(log_posterior(w, *_args)))
        th_p, th_fr = tuple(w[1:1 + n_param_theta]), tuple(w[1 + n_param_theta:])
        expand_p, _ = make_theta_expander(PARAMETERISATION, FR_MODEL, FR_OMEGA_M_ANCHOR)
        expand_fr, _ = fR.make_fr_expander(FR_MODEL, FR_FIXED_PARAMS)
        prior_ok = np.isfinite(model_config["prior_func"](w[0], expand_p(th_p))) and \
                   np.isfinite(fR.FR_REGISTRY[FR_MODEL]["prior_func"](expand_fr(th_fr)))
        if not prior_ok:
            reason = "the starting point is outside the prior / violates the f(R) stability condition"
        else:
            _, reason = make_background_builder(PARAMETERISATION, FR_MODEL, FIXED_DELTA, FIXED_ALPHA, FR_Z_INIT,
                                                FR_OMEGA_M_ANCHOR, FR_FIXED_PARAMS)(th_p, th_fr, return_reason=True)
            reason = reason or "log-likelihood is not finite (non-positive or non-finite E^2(z))"
        raise ValueError(f"{n_bad}/{N_WALKERS} initial walkers have zero probability for "
                         f"PARAMETERISATION={PARAMETERISATION}, FR_MODEL={FR_MODEL}, FR_FIXED_PARAMS={FR_FIXED_PARAMS}.\n"
                         f"Reason (first bad walker): {reason}.")

    n_proc = N_PROCESSES
    if n_proc > 1 and "fork" not in multiprocessing.get_all_start_methods():
        print("Process pool needs the 'fork' start method (Linux/macOS); running serially.")
        n_proc = 1
    if n_proc > 1 and N_WALKERS // 2 < n_proc:
        print(f"Note: only {N_WALKERS // 2} walkers are updated at a time; {n_proc} processes will not all be busy.")

    _PARALLEL_ARGS = _args
    pool = multiprocessing.get_context("fork").Pool(n_proc) if n_proc > 1 else None
    if pool is not None:
        sampler = emcee.EnsembleSampler(N_WALKERS, ndim, _log_prob, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(N_WALKERS, ndim, log_posterior, args=_args)
    if RANDOM_SEED is not None:
        sampler.random_state = np.random.RandomState(RANDOM_SEED).get_state()
    print(f"Sampler: {'serial' if pool is None else f'{n_proc} worker processes'}")

    try:
        print(f"Running Burn-in Loops ({N_BURNIN} steps)...")
        state = sampler.run_mcmc(p0, N_BURNIN, progress=True)
        sampler.reset()

        print(f"Running Production Chains ({N_STEPS} steps)...")
        sampler.run_mcmc(state, N_STEPS, progress=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ════════════════════════════════════════════════════════════════════════════
    # DYNAMIC DIRECTORY SETUP FOR INDIVIDUAL MODELS
    # ════════════════════════════════════════════════════════════════════════════
    base_graph_dir = "graphs"
    model_output_dir = os.path.join(base_graph_dir, RUN_TAG)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"\nTarget output path successfully mapped to: {model_output_dir}")

    flat_chain = sampler.get_chain(flat=True)
    np.save(f"chain_{RUN_TAG}_{RUN_MODE}.npy", flat_chain)

    # ════════════════════════════════════════════════════════════════════════════
    # EXPORT NUMERICAL RESULTS TO TEXT FILE & PRINT POSTERIOR SUMMARY
    # ════════════════════════════════════════════════════════════════════════════
    extra_lines = [f"Parameterisation : {PARAMETERISATION}",
                   f"f(R) model       : {FR_MODEL}  ({model_config['fR_description']})"]
    if FR_MODEL != "GR":
        print("\nRe-solving the f(R) background for a subset of samples (derived quantities)...")
        build_bg = make_background_builder(PARAMETERISATION, FR_MODEL, FIXED_DELTA, FIXED_ALPHA, FR_Z_INIT,
                                         FR_OMEGA_M_ANCHOR, FR_FIXED_PARAMS)
        derived_lines, _ = fR.derived_summary(flat_chain, build_bg, n_param_theta)
        anchor_txt = (f"{model_config['omega_m_anchor']}  (gauge value, NOT sampled; alpha is in anchor units)"
                      if model_config["omega_m_anchor"] is not None else "0.3 (hard-wired in this parameterisation)")
        if FR_FIXED_PARAMS:
            extra_lines += ["f(R) pinned params: " + ", ".join(f"{k} = {v:g}" for k, v in FR_FIXED_PARAMS.items()) + "  (NOT sampled)"]
        extra_lines += [f"FR_Z_INIT        : {FR_Z_INIT}",
                        f"Omega_m anchor   : {anchor_txt}",
                        "Derived (H0 above is the true present-day value):"]
        extra_lines += ["  " + ln for ln in derived_lines]

        med = np.median(flat_chain, axis=0)
        if fR.FR_REGISTRY[FR_MODEL].get("analytic_background") is not None:
            # Perturbative model: no initial conditions; report the truncation-error estimate instead
            bg_med = build_bg(tuple(med[1:1 + n_param_theta]), tuple(med[1 + n_param_theta:]))
            if bg_med is None:
                extra_lines += ["Perturbative truncation test: background invalid at the posterior median"]
            else:
                extra_lines += [f"Perturbative truncation error estimate at the posterior median (z <= {FR_Z_INIT:g}): "
                                f"{bg_med.truncation_error:.1e}  (rejected above {fR.PERTURBATIVE_TOL:.0e})",
                                "  Scalaron-free attractor, 2nd order in alpha; the model is almost degenerate with LCDM for background data."]
            sens = "skip"
        else:
            sens = None
        # Initial-condition convergence test at the posterior median: z_init vs 2 z_init
        build_2x = make_background_builder(PARAMETERISATION, FR_MODEL, FIXED_DELTA, FIXED_ALPHA,
                                           2.0 * FR_Z_INIT, FR_OMEGA_M_ANCHOR, FR_FIXED_PARAMS)
        if sens != "skip":
            sens = fR.ic_sensitivity(build_bg, build_2x, tuple(med[1:1 + n_param_theta]), tuple(med[1 + n_param_theta:]))
        if sens == "skip":
            pass
        elif sens is None:
            extra_lines += ["IC convergence test: background could not be solved at the posterior median"]
        else:
            extra_lines += [f"IC convergence test: max|E2(z_init={FR_Z_INIT:g})/E2(z_init={2*FR_Z_INIT:g}) - 1| = {sens:.2e} on z in [0, 2.4]"]
            if sens > 1e-3:
                extra_lines += ["  WARNING: background is NOT converged w.r.t. FR_Z_INIT -> results depend on the arbitrary",
                                "           starting redshift (no GR-like early attractor for this model/parameters). Do not trust them."]
    write_posterior_summary(flat_chain, model_config, RUN_TAG, RUN_MODE, model_output_dir,
                             USE_SNIA, USE_CC, USE_BAO, USE_CLUSTER, FIXED_DELTA, FIXED_ALPHA,
                             extra_lines=extra_lines)

    # ════════════════════════════════════════════════════════════════════════════
    # EXECUTE POSTERIOR HISTOGRAM DATA WRITER
    # ════════════════════════════════════════════════════════════════════════════
    save_histogram_data(flat_chain, model_config["labels"], model_output_dir, RUN_MODE, RUN_TAG, bins=50)

    # ════════════════════════════════════════════════════════════════════════════
    # GENERATE THE SELECTED GRAPHS (PLOTS switchboard above)
    # ════════════════════════════════════════════════════════════════════════════
    plot_ctx = PlotContext.from_main(sys.modules[__name__], graphs_dir=base_graph_dir, show=SHOW_FIGURES)
    if PLOTS:
        run_plots(PLOTS, plot_ctx, {RUN_TAG: flat_chain})
