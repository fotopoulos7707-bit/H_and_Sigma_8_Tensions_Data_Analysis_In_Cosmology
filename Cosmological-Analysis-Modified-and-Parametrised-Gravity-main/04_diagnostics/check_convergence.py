# check_convergence.py
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

import os
import numpy as np
from run_config import get_run_config, split_tag

RUN_MODE = "joint"
# run tags: "<parameterisation>" or "<parameterisation>__<fR_model>" (e.g. "LCDM__power_law")
MODELS_TO_CHECK = ["LCDM"]

SCRIPT_DIR = str(PROJECT_ROOT)
CHAIN_DIR = os.path.join(SCRIPT_DIR, "chains")

print("=== MCMC CONVERGENCE ANALYSIS (Gelman-Rubin R-hat) ===")
for model_name in MODELS_TO_CHECK:
    filename = f"chain_{model_name}_{RUN_MODE}.npy"
    npy_file = os.path.join(CHAIN_DIR, filename)
    if not os.path.exists(npy_file):
        npy_file = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(npy_file):
        continue
        
    chain = np.load(npy_file)
    labels = get_run_config(*split_tag(model_name))["labels"]
    
    print(f"\nModel: {model_name}")
    print("-" * 40)
    
    # emcee folds walkers into a flat chain. To calculate R-hat precisely, 
    # we split the chain back into virtual independent sub-sequences.
    n_walkers = 64
    n_steps = len(chain) // n_walkers
    reshaped_chain = chain.reshape(n_steps, n_walkers, -1)
    
    for i, label in enumerate(labels):
        # Extract chains for parameter i: shape is (steps, walkers)
        param_chains = reshaped_chain[:, :, i]
        
        # Calculate Between-Chain Variance (B) and Within-Chain Variance (W)
        walker_means = np.mean(param_chains, axis=0)
        global_mean = np.mean(walker_means)
        B = n_steps * np.var(walker_means, ddof=1)
        W = np.mean(np.var(param_chains, axis=0, ddof=1))
        
        # Calculate R-hat
        var_plus = ((n_steps - 1) / n_steps) * W + (1 / n_steps) * B
        r_hat = np.sqrt(var_plus / W) if W > 0 else 1.0
        
        status = "✅ Converged" if r_hat < 1.05 else "❌ Warning: Unconverged"
        print(f"  {label:12s} : R-hat = {r_hat:.4f}   ({status})")