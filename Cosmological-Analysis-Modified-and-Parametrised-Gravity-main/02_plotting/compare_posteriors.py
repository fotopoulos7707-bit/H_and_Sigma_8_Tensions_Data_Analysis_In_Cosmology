# compare_posteriors.py
"""
Model-comparison plots (graphs/Model_Comparisons/).  The drawing code now lives in comparison_plots.py;
this script keeps the old entry point + switchboard.  To pick plots/models on the fly use make_plots.py.
"""
import numpy as np

import main
from plot_selector import PlotContext, find_chains, run_plots

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD CONFIGURATION — CHOOSE MODELS AND PLOTS
# ════════════════════════════════════════════════════════════════════════════
# Entries are run tags: "<parameterisation>" for FR_MODEL="GR", or "<parameterisation>__<fR_model>[__param=value...]"
# (e.g. "LCDM__power_law__alpha=-2__n=0.5").  The run mode (datasets) is main.RUN_MODE.
MODELS_TO_COMPARE = ["LCDM", "LCDM__power_law__alpha=-0.01__n=0.5"]

# Any of: "hist", "box", "violin", "contour_vs_ref", "contour_all", "hubble"  (or "comparison" for all of them).
PLOTS = ["hist", "box", "contour_vs_ref", "contour_all"]

# Parameters compared in hist/box/violin and on the contour axes.  Names: H0, Omega_m, Omega_0, w0, wa,
# alpha_fR, n_fR, Lambda_fR, ... and "matter" (= Omega_m or Omega_0).  f(R) runs have no Omega_m (anchored):
# compare them in e.g. ["H0", "alpha_fR"].  A model lacking a parameter is left out of that panel.
PARAMS = ["H0", "matter"]
PAIR = ["H0", "matter"]
REFERENCE = None     # run tag drawn against every other model in contour_vs_ref; None -> "LCDM" if listed, else the first

# SMOOTHNESS ADJUSTMENT: decrease (e.g. 100, 150 -> fewer bins) to make the histogram curves smoother
HIST_BINS = 16
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ctx = PlotContext.from_main(main, hist_bins=HIST_BINS, params=tuple(PARAMS), pair=tuple(PAIR), reference=REFERENCE)
    found = find_chains(ctx.run_mode, ctx.root)
    print(f"--- Loading MCMC Chains for Run Mode: [{ctx.run_mode}] ---")
    chains = {}
    for tag in MODELS_TO_COMPARE:
        if tag not in found:
            print(f"❌ Warning: no chain file for '{tag}' (run mode {ctx.run_mode}). Skipping.  Found: {list(found)}")
            continue
        chains[tag] = np.load(found[tag])
        print(f"   Loaded {len(chains[tag])} samples from {found[tag]}")
    if not chains:
        raise SystemExit("❌ Critical Error: No datasets were loaded. Check your .npy file sources.")
    run_plots(PLOTS, ctx, chains)
