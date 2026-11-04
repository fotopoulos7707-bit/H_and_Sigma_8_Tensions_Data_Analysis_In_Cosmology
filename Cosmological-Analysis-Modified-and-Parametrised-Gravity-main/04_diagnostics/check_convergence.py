# check_convergence.py
import os
import numpy as np
from run_config import get_run_config, split_tag

RUN_MODE = "joint"
# run tags: "<parameterisation>" or "<parameterisation>__<fR_model>" (e.g. "LCDM__power_law")
MODELS_TO_CHECK = ["LCDM"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=== MCMC CONVERGENCE ANALYSIS (Gelman-Rubin R-hat) ===")
for model_name in MODELS_TO_CHECK:
    npy_file = os.path.join(SCRIPT_DIR, f"chain_{model_name}_{RUN_MODE}.npy")
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