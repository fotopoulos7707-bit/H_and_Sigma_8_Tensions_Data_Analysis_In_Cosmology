# plot_hubble_diagram.py
import os
import numpy as np
import matplotlib.pyplot as plt

import data_loader as dl
from likelihood import make_background_builder
from run_config import get_run_config, split_tag
from main import FIXED_DELTA, FIXED_ALPHA, RUN_MODE, FR_Z_INIT, FR_OMEGA_M_ANCHOR

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD CONFIGURATION — CHOOSE MODELS TO VISUALIZE
# ════════════════════════════════════════════════════════════════════════════


# Add the models you want to compare on the physical distance curves.
# Entries are run tags: "<parameterisation>" or "<parameterisation>__<fR_model>" (e.g. "LCDM__power_law").
MODELS_TO_PLOT = ["LCDM", "wCDM", "CPL","HDE"]

# Background parameters for custom frameworks (FIXED_DELTA, FIXED_ALPHA) are
# now imported directly from main.py so this script can never silently drift
# out of sync with whatever values main.py actually used during the MCMC run.

# Plot formatting choices
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graphs", "Model_Comparisons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION THEORETICAL CALCULATORS
# ════════════════════════════════════════════════════════════════════════════
def get_theory_curves(model_name, median_params, z_grid):
    """
    Computes theoretical H(z) and mu(z) curves given a model's best-fit parameters.
    """
    param_name, fr_model, fr_fixed = split_tag(model_name)
    n_p = get_run_config(param_name, fr_model, fr_fixed, FR_OMEGA_M_ANCHOR)["n_param_theta"]
    H0_fit = median_params[0]
    theta_p = tuple(median_params[1:1 + n_p])
    theta_fr = tuple(median_params[1 + n_p:])

    # Same background builder as the likelihood (GR: parameterisation; f(R): ODE solution)
    bg = make_background_builder(param_name, fr_model, FIXED_DELTA, FIXED_ALPHA, FR_Z_INIT,
                                 FR_OMEGA_M_ANCHOR, fr_fixed)(theta_p, theta_fr)
    if bg is None:
        raise ValueError(f"f(R) background could not be solved for {model_name} at the median parameters.")
    E2_func = bg.E2

    # Calculate H(z) evolution curve
    H_curve = H0_fit * np.sqrt(E2_func(z_grid))

    # Calculate Luminosity Distance Modulus mu(z) curve
    # Using a dense numerical integration array for exact distance calculation
    fine_z = np.linspace(0.0, max(z_grid) * 1.05, 500)
    inv_E = 1.0 / np.sqrt(E2_func(fine_z))
    dz = np.diff(fine_z)
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (inv_E[:-1] + inv_E[1:]) * dz)]) * (dl.C_LIGHT / H0_fit)
    
    # Interpolate from integration grid to our plotting grid
    mu_curve = 5.0 * np.log10((1.0 + z_grid) * np.interp(z_grid, fine_z, chi)) + 25.0
    # Fix the standard boundary value log issue at exactly z=0
    mu_curve[z_grid == 0.0] = 0.0 
    
    return H_curve, mu_curve

# ════════════════════════════════════════════════════════════════════════════
# EXECUTION & PLOTTING PIPELINE
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("--- Loading Observational Dataset Profiles ---")
    # Load raw Supernovae points
    z_sn, mu_sn, _, _, _, _ = dl.load_pantheon_data()
    # Load Cosmic Chronometers points
    z_cc, H_cc, err_cc = dl.load_cc_data()
    
    # Setup plotting mesh
    z_plot_grid = np.linspace(0.01, 2.3, 200)
    
    # Initialize the composite multi-panel plot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    
    # 1. Draw Observational Data Points onto the layouts
    print("Plotting real experimental observations...")
    # Scatter plot for Pantheon Supernovae (using thin alpha to cleanly manage data density)
    ax1.scatter(z_sn, mu_sn, color="gray", s=4, alpha=0.25, label="Pantheon+ SNIa Dataset", zorder=1)
    # Error bars for Cosmic Chronometers
    ax2.errorbar(z_cc, H_cc, yerr=err_cc, fmt="o", color="black", ecolor="dimgray", 
                 elinewidth=1.5, capsize=3, ms=5, label="Cosmic Chronometers (CC)", zorder=2)
    
    # 2. Extract chains and project best-fit model lines
    print("\nExtracting MCMC best-fit states and mapping curves...")
    for idx, model_name in enumerate(MODELS_TO_PLOT):
        npy_file = os.path.join(SCRIPT_DIR, f"chain_{model_name}_{RUN_MODE}.npy")
        
        if not os.path.exists(npy_file):
            print(f"❌ Warning: Chain file '{npy_file}' not found. Skipping model {model_name}.")
            continue
            
        chain = np.load(npy_file)
        # Compute the median (best-fit) parameter array from the distribution
        median_parameters = np.median(chain, axis=0)
        
        # Calculate theoretical data values along our plotting grid
        H_theory, mu_theory_vals = get_theory_curves(model_name, median_parameters, z_plot_grid)
        
        color = COLORS[idx % len(COLORS)]
        # Add smooth curves to both panel plots
        ax1.plot(z_plot_grid, mu_theory_vals, color=color, lw=2.5, label=f"Best-fit {model_name}", zorder=3)
        ax2.plot(z_plot_grid, H_theory, color=color, lw=2.5, label=f"Best-fit {model_name}", zorder=3)
        
        print(f"   -> Plotted best-fit parameters line for model: {model_name}")
        
    # ════════════════════════════════════════════════════════════════════════════
    # AESTHETICS & LAYOUT POLISHING
    # ════════════════════════════════════════════════════════════════════════════
    # Top Panel Configuration (Distance Modulus)
    ax1.set_ylabel(r"Distance Modulus $\mu(z)$", fontsize=13)
    ax1.set_title("Cosmological Distance Curves & Expansion History Profiles", fontsize=14, pad=15)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax1.set_ylim(32, 47)
    
    # Bottom Panel Configuration (Hubble Parameter)
    ax2.set_xlabel(r"Redshift ($z$)", fontsize=13)
    ax2.set_ylabel(r"Hubble Parameter $H(z)$ $\rm [km/s/Mpc]$", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax2.set_xlim(0, 2.3)
    ax2.set_ylim(50, 250)
    
    # Prevent layout label overlapping
    plt.tight_layout()
    
    # Save compilation layout vector file
    save_path = os.path.join(OUTPUT_DIR, f"hubble_diagram_comparison_{RUN_MODE}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"\n✅ Complete comparison diagram saved to: {save_path}")
    
    plt.show()