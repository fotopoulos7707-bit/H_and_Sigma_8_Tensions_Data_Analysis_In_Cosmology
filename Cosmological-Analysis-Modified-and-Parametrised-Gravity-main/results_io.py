# results_io.py
"""
Numerical output: posterior summary text file and 1D histogram data export.
No plotting or sampling logic lives here, only reporting.
"""
import os
import numpy as np


def save_histogram_data(flat_chain, labels, output_dir, run_mode, chosen_model, bins=50):
    """
    Extracts the numerical 1D histogram coordinates (bin midpoints and counts)
    for every sampled posterior distribution parameter and saves them as text files.
    """
    print("\nExporting 1D Posterior Histogram data files...")
    clean_names = {r"$H_0$": "H0", r"$\Omega_m$": "Omega_m", r"$\Omega_0$": "Omega_0",
                   r"$w_0$": "w0", r"$w_a$": "wa", r"$\alpha$": "alpha", r"$\delta$": "delta",
                   r"$\alpha_{f(R)}$": "alpha_fR", r"$n_{f(R)}$": "n_fR"}

    for i, lbl in enumerate(labels):
        col_data = flat_chain[:, i]

        counts, bin_edges = np.histogram(col_data, bins=bins, density=True)
        bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        param_name = clean_names.get(lbl, f"param_{i}")
        txt_filename = os.path.join(output_dir, f"histogram_data_{param_name}_{run_mode}.txt")

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"# 1D Posterior Probability Distribution Histogram Data\n")
            f.write(f"# Model Framework    : {chosen_model}\n")
            f.write(f"# Target Parameter   : {lbl}\n")
            f.write(f"# Data Run Mode      : {run_mode}\n")
            f.write(f"# -----------------------------------------\n")
            f.write(f"# {'Bin_Midpoint':15s}\t{'Probability_Density'}\n")
            f.write(f"# -----------------------------------------\n")
            for mid, cnt in zip(bin_midpoints, counts):
                f.write(f"{mid:15.6f}\t{cnt:.6e}\n")

        print(f"    -> Exported data track for {lbl} into: {txt_filename}")


def write_posterior_summary(flat_chain, model_config, chosen_model, run_mode, output_dir,
                             use_snia, use_cc, use_bao, use_cluster,
                             fixed_delta, fixed_alpha, extra_lines=None):
    """
    Prints the posterior summary table to stdout and writes the same
    information to posterior_results.txt in output_dir.
    """
    txt_filename = os.path.join(output_dir, "posterior_results.txt")

    print(f"\n── Posterior summary for {chosen_model} | datasets: {run_mode} ──")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"==========================================================\n")
        f.write(f"MCMC POSTERIOR STATISTICAL RESULTS\n")
        f.write(f"Model Frame    : {chosen_model}\n")
        f.write(f"Active Datasets: {run_mode}\n")
        f.write(f"  SNIa (Pantheon+SH0ES) : {'ON' if use_snia    else 'OFF'}\n")
        f.write(f"  CC   (Cosmic Chron.)  : {'ON' if use_cc      else 'OFF'}\n")
        f.write(f"  BAO                   : {'ON' if use_bao     else 'OFF'}\n")
        f.write(f"  Clustering            : {'ON' if use_cluster else 'OFF'}\n")
        f.write(f"==========================================================\n")
        f.write(f"{'Parameter':15s} | {'Median':10s} | {'Upper Error (+1sigma)':22s} | {'Lower Error (-1sigma)':22s}\n")
        f.write(f"----------------------------------------------------------\n")

        for i, lbl in enumerate(model_config["labels"]):
            col = flat_chain[:, i]
            med = np.median(col)
            lo = med - np.percentile(col, 16)
            hi = np.percentile(col, 84) - med

            print(f"    {lbl:14s}  {med:.4f}  +{hi:.4f} / -{lo:.4f}")
            f.write(f"{lbl:15s} | {med:10.4f} | {hi:22.4f} | {lo:22.4f}\n")

        if chosen_model in ["Custom_Universal", "param_Hubble_fixed_delta", "param_Hubble_fixed_alpha"]:
            f.write(f"----------------------------------------------------------\n")
            f.write(f"Background Fixed Constraints Set for This Run:\n")
            f.write(f"  FIXED_DELTA = {fixed_delta:.4f}\n")
            f.write(f"  FIXED_ALPHA = {fixed_alpha:.4f}\n")
        if extra_lines:
            f.write(f"----------------------------------------------------------\n")
            for ln in extra_lines:
                f.write(ln + "\n")
                print(ln)
        f.write(f"==========================================================\n")

    print(f"Numerical analysis metrics saved successfully to: {txt_filename}")
    return txt_filename