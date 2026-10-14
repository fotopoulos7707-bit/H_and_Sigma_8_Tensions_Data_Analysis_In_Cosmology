# compare_posteriors.py
import os
import numpy as np
import matplotlib.pyplot as plt
from models import MODEL_REGISTRY
from main import FIXED_ALPHA
from main import FIXED_DELTA, RUN_MODE

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD CONFIGURATION — CHOOSE MODELS AND MODES TO OVERLAY
# ════════════════════════════════════════════════════════════════════════════
#RUN_MODE = "snia+cc+bao+clu"  # "snia" or "joint"

# Add or remove any models you wish to compare here! (Supports 1 to all 6)
MODELS_TO_COMPARE = ["LCDM", "wCDM"]

# SMOOTHNESS ADJUSTMENT: Decrease this number (e.g., 100, 150) to make curves smoother
HIST_BINS = 16  



# Plot layout aesthetic styles
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graphs", "Model_Comparisons")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# CORE DATA EXTRACTION & DENSITY RESOLUTION
# ════════════════════════════════════════════════════════════════════════════
extracted_data = {}

print(f"--- Loading MCMC Chains for Run Mode: [{RUN_MODE}] ---")
for model_name in MODELS_TO_COMPARE:
    npy_file = os.path.join(SCRIPT_DIR, "chains", f"chain_{model_name}_{RUN_MODE}.npy")
    
    if not os.path.exists(npy_file):
        print(f"❌ Warning: Chain file '{npy_file}' not found! Skip model {model_name}.")
        continue
        
    # Load raw flattened chain matrix [N_samples, N_dimensions]
    chain = np.load(npy_file)
    
    # H0 is universally tracked at column index 0 across all pipelines
    h0_samples = chain[:, 0]
    
    # Resolve the matter parameter column.
    # Every model in MODEL_REGISTRY tracks its "matter-like" parameter at
    # chain column 1 EXCEPT the three custom parametrized-Hubble models,
    # which track Omega_0 there instead. Default to Omega_m and only switch
    # to Omega_0 for the explicit custom-model exceptions, so any current or
    # future standard model (CPL, JBP, HDE, IDE, ...) is covered automatically
    # instead of needing to be added to an allowlist by hand.
    custom_hubble_models = ["param_Hubble_fixed_delta", "param_Hubble_fixed_alpha", "Custom_Universal"]
    if model_name in custom_hubble_models:
        matter_samples = chain[:, 1]
        param_label = r"$\Omega_0$ [$\Omega_1(z=0)$]"
    else:
        matter_samples = chain[:, 1]
        param_label = r"$\Omega_m$"

    extracted_data[model_name] = {
        "H0": h0_samples,
        "Matter": matter_samples,
        "Label": param_label
    }
    print(f"   Successfully loaded {len(h0_samples)} samples from {model_name}")

# ════════════════════════════════════════════════════════════════════════════
# PLOTTING ROUTINES
# ════════════════════════════════════════════════════════════════════════════
if not extracted_data:
    print("❌ Critical Error: No datasets were loaded. Check your .npy file sources.")
    exit()

# ── PLOT 1: H0 HISTOGRAM POSTERIOR COMPARISON ──
print(f"\nGenerating H0 Poster Distributions Overlay (Bins={HIST_BINS})...")
plt.figure(figsize=(8, 5.5))

for idx, (model_name, data) in enumerate(extracted_data.items()):
    color = COLORS[idx % len(COLORS)]
    # Use hist type='step' to create elegant overlapping scientific contours
    plt.hist(data["H0"], bins=HIST_BINS, density=True, histtype="step", lw=2.5,
             color=color, label=f"{model_name}")
    # Optional alpha filling to make overlaps easily recognizable
    plt.hist(data["H0"], bins=HIST_BINS, density=True, histtype="stepfilled", 
             color=color, alpha=0.04)

plt.xlabel(r"Hubble Constant $H_0$ $\rm [km/s/Mpc]$", fontsize=13)
plt.ylabel(r"Probability Density $P(H_0 | \rm Data)$", fontsize=13)
plt.title(f"Hubble Parameter Constraint Discrepancy Analysis ({RUN_MODE})", fontsize=13, pad=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper right", fontsize=10, framealpha=0.9)

h0_save_path = os.path.join(OUTPUT_DIR, f"comparison_H0_{RUN_MODE}.pdf")
plt.savefig(h0_save_path, bbox_inches="tight")
print(f"   -> Saved H0 comparison plot to: {h0_save_path}")


# ── PLOT 2: MATTER PARAMETER HISTOGRAM OVERLAY ──
print(f"Generating Matter Component Posterior Distributions Overlay (Bins={HIST_BINS})...")
plt.figure(figsize=(8, 5.5))

for idx, (model_name, data) in enumerate(extracted_data.items()):
    if data["Matter"] is None:
        continue
    color = COLORS[idx % len(COLORS)]
    
    plt.hist(data["Matter"], bins=HIST_BINS, density=True, histtype="step", lw=2.5,
             color=color, label=f"{model_name}")
    plt.hist(data["Matter"], bins=HIST_BINS, density=True, histtype="stepfilled", 
             color=color, alpha=0.04)

plt.xlabel(r"Matter / Base Density SCALE $\Omega_x(z=0)$", fontsize=13)
plt.ylabel(r"Probability Density $P(\Omega_x | \rm Data)$", fontsize=13)
plt.title(f"Matter Density vs Custom $\\Omega_1(z=0)$ Posteriors ({RUN_MODE})", fontsize=13, pad=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper right", fontsize=10, framealpha=0.9)

matter_save_path = os.path.join(OUTPUT_DIR, f"comparison_Matter_{RUN_MODE}.pdf")
plt.savefig(matter_save_path, bbox_inches="tight")
print(f"   -> Saved Matter parameter comparison plot to: {matter_save_path}")


# ── PLOT 3: MEDIAN ± 1σ BOX PLOT ACROSS MODELS ──
print(f"Generating Median/Error Box Plot Comparison...")

def _custom_box_stats(samples, label):
    """
    Builds a single matplotlib bxp() stats dict using the 16/50/84 percentile
    convention used everywhere else in this codebase (1-sigma equivalent),
    rather than matplotlib's default 25/50/75 quartile box.
    """
    med = np.median(samples)
    lo  = np.percentile(samples, 16)
    hi  = np.percentile(samples, 84)
    return {
        "label": label,
        "med": med,
        "q1": lo,        # bottom of box -> 16th percentile
        "q3": hi,        # top of box    -> 84th percentile
        "whislo": lo,    # no extra whisker beyond the 1-sigma box
        "whishi": hi,
        "fliers": [],
    }

model_names = list(extracted_data.keys())
fig, (axb1, axb2) = plt.subplots(1, 2, figsize=(11, 5.5))

# H0 panel
h0_stats = [_custom_box_stats(extracted_data[m]["H0"], m) for m in model_names]
bp1 = axb1.bxp(h0_stats, showfliers=False, patch_artist=True, widths=0.5)
for idx, box in enumerate(bp1["boxes"]):
    box.set_facecolor(COLORS[idx % len(COLORS)])
    box.set_alpha(0.5)
axb1.set_ylabel(r"Hubble Constant $H_0$ $\rm [km/s/Mpc]$", fontsize=12)
axb1.set_title(r"$H_0$ Median $\pm 1\sigma$", fontsize=12)
axb1.grid(True, linestyle="--", alpha=0.4, axis="y")

# Matter/Omega_0 panel (skip models with no matter samples, same as Plot 2)
matter_models = [m for m in model_names if extracted_data[m]["Matter"] is not None]
matter_stats = [_custom_box_stats(extracted_data[m]["Matter"], m) for m in matter_models]
if matter_stats:
    bp2 = axb2.bxp(matter_stats, showfliers=False, patch_artist=True, widths=0.5)
    for idx, box in enumerate(bp2["boxes"]):
        box.set_facecolor(COLORS[idx % len(COLORS)])
        box.set_alpha(0.5)
axb2.set_ylabel(r"Matter / Base Density Scale $\Omega_x(z=0)$", fontsize=12)
axb2.set_title(r"$\Omega_m$ / $\Omega_0$ Median $\pm 1\sigma$", fontsize=12)
axb2.grid(True, linestyle="--", alpha=0.4, axis="y")

fig.suptitle(f"Posterior Median & 1σ Spread Across Models ({RUN_MODE})", fontsize=13, y=1.02)
plt.tight_layout()

box_save_path = os.path.join(OUTPUT_DIR, f"comparison_BoxPlot_{RUN_MODE}.pdf")
plt.savefig(box_save_path, bbox_inches="tight")
print(f"   -> Saved median/error box plot to: {box_save_path}")





# ── PLOT 4: PAIRWISE 2D CONTOUR (BANANA/ELLIPSE) OVERLAYS AGAINST LCDM ──
# Different models carry different numbers of free parameters (LCDM has 2,
# wCDM has 3, CPL/JBP have 4, ...), so a full corner plot can't overlay all
# of them in one figure. What IS common to every model is the (H0, Matter)
# pair already extracted above, so each non-LCDM model gets its own 2D
# contour overlay against LCDM in that shared subspace.
#
# Uses corner.hist2d() directly rather than corner.corner(), since
# corner.corner() always draws the 1D marginal histograms on the diagonal.
# hist2d() draws ONLY the 2D contour (the banana/ellipse shape itself) with
# no marginal panels.
import corner

if "LCDM" not in extracted_data:
    print("\n⚠ Skipping pairwise corner comparisons: LCDM chain not loaded.")
else:
    lcdm_data = extracted_data["LCDM"]
    if lcdm_data["Matter"] is None:
        print("\n⚠ Skipping pairwise corner comparisons: LCDM has no Matter samples.")
    else:
        lcdm_H0, lcdm_Matter = lcdm_data["H0"], lcdm_data["Matter"]

        other_models = [m for m in model_names if m != "LCDM" and extracted_data[m]["Matter"] is not None]
        if not other_models:
            print("\n⚠ No other models with Matter samples available for pairwise contour comparison.")

        for idx, model_name in enumerate(other_models, start=1):
            print(f"Generating pairwise 2D contour overlay: LCDM vs {model_name}...")
            other_data = extracted_data[model_name]

            lcdm_color = COLORS[0]
            other_color = COLORS[idx % len(COLORS)]
            if other_color == lcdm_color:
                other_color = COLORS[(idx + 1) % len(COLORS)]

            fig, ax = plt.subplots(figsize=(6.5, 6))

            corner.hist2d(lcdm_H0, lcdm_Matter, ax=ax, color=lcdm_color,
                          levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),  # 1-sigma, 2-sigma
                          plot_datapoints=False, plot_density=False, fill_contours=True)
            corner.hist2d(other_data["H0"], other_data["Matter"], ax=ax, color=other_color,
                          levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
                          plot_datapoints=False, plot_density=False, fill_contours=True)

            ax.set_xlabel(r"$H_0$ $\rm [km/s/Mpc]$", fontsize=13)
            ax.set_ylabel(r"Matter Density $\Omega_x$", fontsize=13)
            ax.set_title(f"LCDM vs {model_name} ({RUN_MODE})", fontsize=13, pad=12)

            legend_handles = [
                plt.Line2D([0], [0], color=lcdm_color, lw=2.5, label="LCDM"),
                plt.Line2D([0], [0], color=other_color, lw=2.5, label=model_name),
            ]
            ax.legend(handles=legend_handles, loc="best", fontsize=11, framealpha=0.9)

            pair_save_path = os.path.join(OUTPUT_DIR, f"comparison_contour_LCDM_vs_{model_name}_{RUN_MODE}.pdf")
            fig.savefig(pair_save_path, bbox_inches="tight")
            
            print(f"   -> Saved pairwise 2D contour overlay to: {pair_save_path}")

print("\nProcessing complete. Displaying generated figures...")
plt.show()


# ── PLOT 5: ALL-MODELS COMBINED 2D CONTOUR OVERLAY ──
# Same (H0, Matter) shared subspace as Plot 5, but every loaded model is
# drawn together on one panel instead of one-at-a-time against LCDM, so you
# can see how the whole set of degeneracy "bananas" sit relative to each
# other, not just relative to LCDM individually.
all_contour_models = [m for m in model_names if extracted_data[m]["Matter"] is not None]
 
if len(all_contour_models) < 2:
    print("\n⚠ Skipping combined all-models contour plot: need at least 2 models with Matter samples.")
else:
    print(f"\nGenerating combined contour overlay for all models: {all_contour_models}...")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
 
    legend_handles = []
    for idx, model_name in enumerate(all_contour_models):
        color = COLORS[idx % len(COLORS)]
        data = extracted_data[model_name]
        corner.hist2d(data["H0"], data["Matter"], ax=ax, color=color,
                      levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
                      plot_datapoints=False, plot_density=False, fill_contours=True)
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.5, label=model_name))
 
    ax.set_xlabel(r"$H_0$ $\rm [km/s/Mpc]$", fontsize=13)
    ax.set_ylabel(r"Matter Density $\Omega_x$", fontsize=13)
    ax.set_title(f"All Models: $H_0$ vs Matter Density Posterior Contours ({RUN_MODE})", fontsize=13, pad=12)
    ax.legend(handles=legend_handles, loc="best", fontsize=10, framealpha=0.9)
 
    all_contour_save_path = os.path.join(OUTPUT_DIR, f"comparison_contour_AllModels_{RUN_MODE}.pdf")
    fig.savefig(all_contour_save_path, bbox_inches="tight")
    
    print(f"   -> Saved combined all-models contour overlay to: {all_contour_save_path}")
 
print("\nProcessing complete. Displaying generated figures...")
plt.show()
