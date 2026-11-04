# plot_hubble_diagram.py
"""
Best-fit mu(z) and H(z) curves of the chosen models over the SNIa and cosmic-chronometer data
(graphs/Model_Comparisons/).  The drawing code now lives in comparison_plots.plot_hubble; this script
keeps the old entry point + switchboard.  To pick plots/models on the fly use make_plots.py.
"""
import numpy as np

import main
from plot_selector import PlotContext, find_chains, run_plots

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD CONFIGURATION — CHOOSE MODELS TO VISUALIZE
# ════════════════════════════════════════════════════════════════════════════
# Entries are run tags: "<parameterisation>" or "<parameterisation>__<fR_model>" (e.g. "LCDM__power_law__alpha=-2__n=0.5").
MODELS_TO_PLOT = ["LCDM", "wCDM", "CPL", "HDE"]
# FIXED_DELTA, FIXED_ALPHA, FR_Z_INIT, FR_OMEGA_M_ANCHOR and the run mode are taken from main.py, so this
# script cannot drift out of sync with the values main.py used for the MCMC run.
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ctx = PlotContext.from_main(main)
    found = find_chains(ctx.run_mode, ctx.root)
    chains = {}
    for tag in MODELS_TO_PLOT:
        if tag not in found:
            print(f"❌ Warning: no chain file for '{tag}' (run mode {ctx.run_mode}). Skipping.  Found: {list(found)}")
            continue
        chains[tag] = np.load(found[tag])
    if not chains:
        raise SystemExit("❌ No chains found for MODELS_TO_PLOT.")
    run_plots(["hubble"], ctx, chains)
