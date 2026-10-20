# plot_walk_traces.py
import os
import numpy as np
import matplotlib.pyplot as plt
from run_config import get_run_config, split_tag

CHOSEN_MODEL = "LCDM"   # run tag: "<parameterisation>" or "<parameterisation>__<fR_model>"
RUN_MODE = "joint"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

npy_file = os.path.join(SCRIPT_DIR, f"chain_{CHOSEN_MODEL}_{RUN_MODE}.npy")
if os.path.exists(npy_file):
    chain = np.load(npy_file)
    labels = get_run_config(*split_tag(CHOSEN_MODEL))["labels"]
    ndim = len(labels)
    
    # Reconstruct separate walker paths
    n_walkers = 64
    n_steps = len(chain) // n_walkers
    reshaped = chain.reshape(n_steps, n_walkers, ndim)
    
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 2 * ndim), sharex=True)
    if ndim == 1: axes = [axes]
    
    for i in range(ndim):
        axes[i].plot(reshaped[:, :, i], color="black", alpha=0.15)
        axes[i].set_ylabel(labels[i], fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("MCMC Step Index", fontsize=12)
    plt.suptitle(f"Walker Diagnostics: {CHOSEN_MODEL}", y=0.98)
    
    out_path = os.path.join(SCRIPT_DIR, "graphs", CHOSEN_MODEL, f"traces_{CHOSEN_MODEL}.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Diagnostics traces generated at: {out_path}")
    plt.show()