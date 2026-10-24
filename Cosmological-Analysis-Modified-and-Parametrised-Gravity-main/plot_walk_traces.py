# plot_walk_traces.py
"""
Walker trace plots of one saved chain (graphs/<tag>/).  The drawing code now lives in plotting.plot_traces;
this script keeps the old entry point + switchboard.  To pick plots/models on the fly use make_plots.py.
"""
import numpy as np

import main
from plot_selector import PlotContext, find_chains, run_plots

CHOSEN_MODEL = "LCDM"   # run tag: "<parameterisation>" or "<parameterisation>__<fR_model>"
# Run mode (datasets) and the walker count (needed to un-flatten the chain) come from main.py.

if __name__ == "__main__":
    ctx = PlotContext.from_main(main)
    found = find_chains(ctx.run_mode, ctx.root)
    if CHOSEN_MODEL not in found:
        raise SystemExit(f"No chain for '{CHOSEN_MODEL}' with run mode '{ctx.run_mode}'. Found: {list(found)}")
    run_plots(["traces"], ctx, {CHOSEN_MODEL: np.load(found[CHOSEN_MODEL])})
