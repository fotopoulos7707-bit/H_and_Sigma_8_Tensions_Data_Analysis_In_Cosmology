# main.py
import numpy as np
import emcee
import matplotlib.pyplot as plt
import os

import data_loader as dl
import models
from models import MODEL_REGISTRY
from likelihood import make_log_posterior
from plotting import plot_corner, generate_model_specific_plots
from results_io import save_histogram_data, write_posterior_summary

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD CONFIGURATION — SWITCH SCENARIOS HERE
# ════════════════════════════════════════════════════════════════════════════
CHOSEN_MODEL = "wCDM_fixed_Omega_m"   # Options: "LCDM", "wCDM", "wowaCDM", "CPL", "JBP", "HDE", "IDE",
                         #          "param_Hubble_fixed_delta", "param_Hubble_fixed_alpha", "Custom_Universal"

# ── Dataset Toggles ─────────────────────────────────────────────────────────
# Set each to True/False to include or exclude that dataset from the likelihood.
# At least one must be True, or the run will abort.
USE_SNIA    = True   # Pantheon+SH0ES Type Ia Supernovae (full covariance, marginalized M_B)
USE_CC      = True   # Cosmic Chronometers H(z)
USE_BAO     = True   # Baryon Acoustic Oscillations H(z)
USE_CLUSTER = True   # Galaxy Clustering H(z)
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

# ════════════════════════════════════════════════════════════════════════════
# VALIDATE SELECTION & INITIALIZE ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
model_config = MODEL_REGISTRY[CHOSEN_MODEL]
E2_func = model_config["E2_func"]
prior_func = model_config["prior_func"]

log_posterior = make_log_posterior(CHOSEN_MODEL, E2_func, prior_func, FIXED_DELTA, FIXED_ALPHA)

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION ROUTINE
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Initializing MCMC Target: [{CHOSEN_MODEL}] | Active datasets: [{RUN_MODE}]")

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
    p0 = model_config["p0_center"] + model_config["p0_scatter"] * np.random.standard_normal((N_WALKERS, ndim))

    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, log_posterior,
                                    args=(z_cmb, mu_obs, C_fac, Cinv_ones, scalar_denom, hz_datasets))

    print(f"Running Burn-in Loops ({N_BURNIN} steps)...")
    state = sampler.run_mcmc(p0, N_BURNIN, progress=True)
    sampler.reset()

    print(f"Running Production Chains ({N_STEPS} steps)...")
    sampler.run_mcmc(state, N_STEPS, progress=True)

    # ════════════════════════════════════════════════════════════════════════════
    # DYNAMIC DIRECTORY SETUP FOR INDIVIDUAL MODELS
    # ════════════════════════════════════════════════════════════════════════════
    base_graph_dir = "graphs"
    model_output_dir = os.path.join(base_graph_dir, CHOSEN_MODEL)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"\nTarget output path successfully mapped to: {model_output_dir}")

    flat_chain = sampler.get_chain(flat=True)
    np.save(f"chain_{CHOSEN_MODEL}_{RUN_MODE}.npy", flat_chain)

    # ════════════════════════════════════════════════════════════════════════════
    # EXPORT NUMERICAL RESULTS TO TEXT FILE & PRINT POSTERIOR SUMMARY
    # ════════════════════════════════════════════════════════════════════════════
    write_posterior_summary(flat_chain, model_config, CHOSEN_MODEL, RUN_MODE, model_output_dir,
                             USE_SNIA, USE_CC, USE_BAO, USE_CLUSTER, FIXED_DELTA, FIXED_ALPHA)

    # ════════════════════════════════════════════════════════════════════════════
    # EXECUTE POSTERIOR HISTOGRAM DATA WRITER
    # ════════════════════════════════════════════════════════════════════════════
    save_histogram_data(flat_chain, model_config["labels"], model_output_dir, RUN_MODE, CHOSEN_MODEL, bins=50)

    # ════════════════════════════════════════════════════════════════════════════
    # GENERATE CORNER PLOTS
    # ════════════════════════════════════════════════════════════════════════════
    plot_corner(flat_chain, model_config["labels"], CHOSEN_MODEL, model_output_dir, RUN_MODE)

    # ════════════════════════════════════════════════════════════════════════════
    # GENERATE COMPONENT OR POTENTIAL TRACES
    # ════════════════════════════════════════════════════════════════════════════
    generate_model_specific_plots(flat_chain, CHOSEN_MODEL, model_config, model_output_dir, RUN_MODE,
                                   FIXED_DELTA, FIXED_ALPHA)

    print("\nOpening active interactive figure windows... Close graphics window to exit pipeline.")
    plt.show()