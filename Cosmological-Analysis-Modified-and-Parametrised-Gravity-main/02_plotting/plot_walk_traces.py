# plot_walk_traces.py
"""
Walker trace plots of one saved chain (graphs/<tag>/).  The drawing code now lives in plotting.plot_traces;
this script keeps the old entry point + switchboard.  To pick plots/models on the fly use make_plots.py.
"""

# Resolve the project root so scripts work after being moved into workflow folders.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve()
while not (_PROJECT_ROOT / "project_paths.py").is_file() and _PROJECT_ROOT != _PROJECT_ROOT.parent:
    _PROJECT_ROOT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import configure_import_paths
PROJECT_ROOT = configure_import_paths()

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
