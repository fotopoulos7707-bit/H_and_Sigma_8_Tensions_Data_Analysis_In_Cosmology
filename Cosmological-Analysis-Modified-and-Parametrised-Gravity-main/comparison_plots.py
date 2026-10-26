# comparison_plots.py
"""
Multi-model figures (everything that ends up in graphs/Model_Comparisons/).

Any set of runs can be compared -- GR parameterisations, f(R) runs, or a mix -- in any parameters they
share.  Each plot is a function  f(ctx, data)  where
    ctx  : plot_selector.PlotContext  (run mode, output dir, which parameters, reference model, ...)
    data : {run_tag: {"params": {name: (latex_label, samples)}, "chain": full chain}}  from extract_summary(),
           in the order the models were requested.
Parameters are addressed by name (see PARAM_NAMES: H0, Omega_m, w0, wa, alpha_fR, n_fR, Lambda_fR, ...).
"matter" is an alias for Omega_m / Omega_0.  A model that does not sample a requested parameter is left out
of that panel/figure (f(R) runs anchor Omega_m, so they have no Omega_m), and the plot says so.
The functions only draw and save; showing / closing figures is done by plot_selector.run_plots.
A function raises PlotSkipped when the selection cannot produce that plot.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import corner

import data_loader as dl
import fR
from likelihood import make_background_builder
from run_config import get_run_config, split_tag

COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
          "#e6ab02", "#a6761d", "#666666", "#1b9e77", "#e41a1c"]
CONTOUR_LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))     # 1 sigma, 2 sigma

# chain label -> parameter name
PARAM_NAMES = {r"$H_0$": "H0", r"$\Omega_m$": "Omega_m", r"$\Omega_0$": "Omega_0", r"$w_0$": "w0", r"$w_a$": "wa",
               r"$\alpha$": "alpha", r"$\delta$": "delta", r"$c$": "c", r"$\xi$": "xi",
               r"$\alpha_{f(R)}$": "alpha_fR", r"$n_{f(R)}$": "n_fR", r"$\Lambda_{f(R)}$": "Lambda_fR", r"$b_{f(R)}$": "b_fR"}
MATTER_ALIASES = ("Omega_m", "Omega_0")                  # what "matter" resolves to, per model
DEFAULT_PARAMS = ("H0", "matter")
DEFAULT_PAIR = ("H0", "matter")


class PlotSkipped(Exception):
    """The plot is not applicable to the selected models (not an error)."""


def param_name(label):
    return PARAM_NAMES.get(label, label.strip("$").replace("\\", "").replace("{", "").replace("}", ""))


def extract_summary(tag, chain):
    """{"params": {name: (label, samples)}, "chain": chain}; also adds the "matter" alias when present."""
    labels = get_run_config(*split_tag(tag))["labels"]
    if chain.ndim != 2 or chain.shape[1] != len(labels):
        raise ValueError(f"chain for '{tag}' has shape {chain.shape} but the model defines "
                         f"{len(labels)} parameters {labels}; chain from an older version of the code?")
    params = {param_name(lb): (lb, chain[:, i]) for i, lb in enumerate(labels)}
    for m in MATTER_ALIASES:
        if m in params:
            params["matter"] = params[m]
            break
    return {"params": params, "chain": chain}


def parameters_of(tag):
    """Parameter names a run tag samples (for menus / --list); no chain needed."""
    names = [param_name(lb) for lb in get_run_config(*split_tag(tag))["labels"]]
    return names + (["matter"] if any(m in names for m in MATTER_ALIASES) else [])


# ── helpers ─────────────────────────────────────────────────────────────────
def _color(data, tag):
    return COLORS[list(data).index(tag) % len(COLORS)]       # a model keeps its colour in every figure


def _having(data, *names):
    """Models that sample all of the named parameters."""
    return [t for t, d in data.items() if all(n in d["params"] for n in names)]


def _axis_label(data, models, name):
    if name == "matter":
        return r"Matter density $\Omega_x(z=0)$  ($\Omega_m$ or $\Omega_0$)"
    return data[models[0]]["params"][name][0]


def _note_missing(data, name, models):
    gone = [t for t in data if t not in models]
    if gone:
        print(f"   (no sampled '{name}' in: {', '.join(gone)})")


def _check_names(data, names):
    known = sorted({n for d in data.values() for n in d["params"]})
    bad = [n for n in names if n not in known]
    if bad:
        raise PlotSkipped(f"parameter(s) {bad} not sampled by any selected model (available: {known})")


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    print(f"   -> saved: {path}")


def _rotate_ticks(ax):
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_ha("right")


def _fname(prefix, name, run_mode):
    legacy = {"H0": "H0", "matter": "Matter"}                  # keep the historical file names
    return f"{prefix}_{legacy.get(name, name)}_{run_mode}.pdf"


# ════════════════════════════════════════════════════════════════════════════
# 1D POSTERIOR OVERLAYS  (one figure per chosen parameter)
# ════════════════════════════════════════════════════════════════════════════
def plot_hist(ctx, data):
    _check_names(data, ctx.params)
    figs = []
    for name in ctx.params:
        models = _having(data, name)
        if not models:
            print(f"   ! '{name}': no selected model samples it -- skipped")
            continue
        _note_missing(data, name, models)
        fig = plt.figure(figsize=(8, 5.5))
        for t in models:
            s = data[t]["params"][name][1]
            c = _color(data, t)
            plt.hist(s, bins=ctx.hist_bins, density=True, histtype="step", lw=2.5, color=c, label=t)
            plt.hist(s, bins=ctx.hist_bins, density=True, histtype="stepfilled", color=c, alpha=0.04)
        xl = _axis_label(data, models, name)
        plt.xlabel(xl, fontsize=13)
        plt.ylabel("Probability Density", fontsize=13)
        plt.title(f"Posterior of {xl} ({ctx.run_mode})", fontsize=13, pad=12)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="best", fontsize=10, framealpha=0.9)
        _save(fig, os.path.join(ctx.comparison_dir(), _fname("comparison", name, ctx.run_mode)))
        figs.append(fig)
    if not figs:
        raise PlotSkipped("no figure could be made")
    return figs


# ════════════════════════════════════════════════════════════════════════════
# BOX / VIOLIN SUMMARIES  (median +- 1 sigma, 16/50/84 percentiles as everywhere else)
# One panel per chosen parameter.
# ════════════════════════════════════════════════════════════════════════════
def _box_stats(samples, label):
    lo, med, hi = np.percentile(samples, [16, 50, 84])
    return {"label": label, "med": med, "q1": lo, "q3": hi, "whislo": lo, "whishi": hi, "fliers": []}


def _panels(ctx, data, draw, fname, suptitle):
    _check_names(data, ctx.params)
    fig, axes = plt.subplots(1, len(ctx.params), figsize=(5.5 * len(ctx.params), 5.5), squeeze=False)
    for ax, name in zip(axes[0], ctx.params):
        models = _having(data, name)
        if models:
            _note_missing(data, name, models)
            draw(ax, data, models, name)
            ax.set_ylabel(_axis_label(data, models, name), fontsize=12)
            ax.set_title(name, fontsize=12)
            _rotate_ticks(ax)
        else:
            ax.text(0.5, 0.5, f"no selected model samples '{name}'", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.suptitle(f"{suptitle} ({ctx.run_mode})", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(ctx.comparison_dir(), f"{fname}_{ctx.run_mode}.pdf"))
    return fig


def plot_box(ctx, data):
    def draw(ax, data, models, name):
        bp = ax.bxp([_box_stats(data[t]["params"][name][1], t) for t in models],
                    showfliers=False, patch_artist=True, widths=0.5)
        for box, t in zip(bp["boxes"], models):
            box.set_facecolor(_color(data, t)); box.set_alpha(0.5)
    return _panels(ctx, data, draw, "comparison_BoxPlot", "Posterior Median & 1σ Spread Across Models")


def plot_violin(ctx, data):
    def draw(ax, data, models, name):
        samples = [data[t]["params"][name][1] for t in models]
        parts = ax.violinplot(samples, positions=range(1, len(models) + 1), widths=0.8,
                              showextrema=False, showmedians=False)
        for body, t in zip(parts["bodies"], models):
            body.set_facecolor(_color(data, t)); body.set_edgecolor("black"); body.set_alpha(0.5)
        for pos, s in enumerate(samples, start=1):
            lo, med, hi = np.percentile(s, [16, 50, 84])
            ax.vlines(pos, lo, hi, color="black", lw=3, zorder=3)                       # 16-84 % range
            ax.scatter([pos], [med], color="white", edgecolor="black", zorder=4, s=28)  # median
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models)
    return _panels(ctx, data, draw, "comparison_Violin", "Posterior Distributions Across Models; bar = 16-84 %, dot = median")


# ════════════════════════════════════════════════════════════════════════════
# 2D CONTOURS in a chosen parameter pair (ctx.pair, default H0 vs matter).
# corner.hist2d draws the contour only (corner.corner would add 1D marginals).
# ════════════════════════════════════════════════════════════════════════════
def _contour(ax, d, pair, color):
    corner.hist2d(d["params"][pair[0]][1], d["params"][pair[1]][1], ax=ax, color=color, levels=CONTOUR_LEVELS,
                  plot_datapoints=False, plot_density=False, fill_contours=True)


def _pair_suffix(pair):
    return "" if tuple(pair) == DEFAULT_PAIR else f"_{pair[0]}-{pair[1]}"


def _contour_axes(ax, data, models, pair):
    ax.set_xlabel(_axis_label(data, models, pair[0]), fontsize=13)
    ax.set_ylabel(_axis_label(data, models, pair[1]), fontsize=13)


def _pair(ctx, data):
    pair = tuple(ctx.pair)
    _check_names(data, pair)
    return pair


def plot_contour_vs_ref(ctx, data):
    """Each model's contour overlaid on a reference model's (default: LCDM if selected, else the first model)."""
    pair = _pair(ctx, data)
    ref = ctx.reference or ("LCDM" if "LCDM" in data else next(iter(data)))
    if ref not in data:
        raise PlotSkipped(f"reference '{ref}' is not among the selected models {list(data)}")
    if ref not in _having(data, *pair):
        raise PlotSkipped(f"reference '{ref}' does not sample both {pair[0]} and {pair[1]}")
    others = [t for t in _having(data, *pair) if t != ref]
    if not others:
        raise PlotSkipped(f"no other selected model samples both {pair[0]} and {pair[1]}")
    figs = []
    for t in others:
        print(f"   {ref} vs {t} ...")
        rc, oc = _color(data, ref), _color(data, t)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        _contour(ax, data[ref], pair, rc)
        _contour(ax, data[t], pair, oc)
        _contour_axes(ax, data, [ref], pair)
        ax.set_title(f"{ref} vs {t} ({ctx.run_mode})", fontsize=11, pad=12)
        ax.legend(handles=[plt.Line2D([0], [0], color=rc, lw=2.5, label=ref),
                           plt.Line2D([0], [0], color=oc, lw=2.5, label=t)], loc="best", fontsize=10, framealpha=0.9)
        _save(fig, os.path.join(ctx.comparison_dir(),
                                f"comparison_contour_{ref}_vs_{t}{_pair_suffix(pair)}_{ctx.run_mode}.pdf"))
        figs.append(fig)
    return figs


def plot_contour_all(ctx, data):
    pair = _pair(ctx, data)
    models = _having(data, *pair)
    if len(models) < 2:
        raise PlotSkipped(f"needs at least 2 selected models sampling both {pair[0]} and {pair[1]} "
                          f"(have: {models or 'none'})")
    _note_missing(data, f"{pair[0]}+{pair[1]}", models)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    handles = []
    for t in models:
        c = _color(data, t)
        _contour(ax, data[t], pair, c)
        handles.append(plt.Line2D([0], [0], color=c, lw=2.5, label=t))
    _contour_axes(ax, data, models, pair)
    ax.set_title(f"All Models: {pair[0]} vs {pair[1]} Posterior Contours ({ctx.run_mode})", fontsize=12, pad=12)
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    _save(fig, os.path.join(ctx.comparison_dir(), f"comparison_contour_AllModels{_pair_suffix(pair)}_{ctx.run_mode}.pdf"))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# HUBBLE DIAGRAM: best-fit (posterior-median) mu(z) and H(z) against SNIa and CC data
# ════════════════════════════════════════════════════════════════════════════
def _theory_curves(ctx, tag, median_params, z_grid):
    param_name, fr_model, fr_fixed = split_tag(tag)
    n_p = get_run_config(param_name, fr_model, fr_fixed, ctx.fr_omega_m_anchor)["n_param_theta"]
    H0_fit = median_params[0]
    theta_p, theta_fr = tuple(median_params[1:1 + n_p]), tuple(median_params[1 + n_p:])

    # z_init is a per-f(R)-model default: only reuse main.py's value if it belongs to the same f(R) model
    z_init = ctx.fr_z_init if fr_model == ctx.fr_model else fR.FR_REGISTRY[fr_model].get("z_init_default", 40.0)
    bg = make_background_builder(param_name, fr_model, ctx.fixed_delta, ctx.fixed_alpha, z_init,
                                 ctx.fr_omega_m_anchor, fr_fixed)(theta_p, theta_fr)
    if bg is None:
        raise ValueError(f"background could not be solved for '{tag}' at the posterior median")

    H_curve = H0_fit * np.sqrt(bg.E2(z_grid))
    fine_z = np.linspace(0.0, max(z_grid) * 1.05, 500)               # trapezoidal comoving distance
    inv_E = 1.0 / np.sqrt(bg.E2(fine_z))
    chi = np.concatenate([[0.0], np.cumsum(0.5 * (inv_E[:-1] + inv_E[1:]) * np.diff(fine_z))]) * (dl.C_LIGHT / H0_fit)
    mu_curve = 5.0 * np.log10((1.0 + z_grid) * np.interp(z_grid, fine_z, chi)) + 25.0
    return H_curve, mu_curve


def plot_hubble(ctx, data):
    z_sn, mu_sn, *_ = dl.load_pantheon_data()
    z_cc, H_cc, err_cc = dl.load_cc_data()
    z_grid = np.linspace(0.01, 2.3, 200)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    ax1.scatter(z_sn, mu_sn, color="gray", s=4, alpha=0.25, label="Pantheon+ SNIa Dataset", zorder=1)
    ax2.errorbar(z_cc, H_cc, yerr=err_cc, fmt="o", color="black", ecolor="dimgray",
                 elinewidth=1.5, capsize=3, ms=5, label="Cosmic Chronometers (CC)", zorder=2)

    n_drawn = 0
    for i, (m, d) in enumerate(data.items()):
        try:
            H_th, mu_th = _theory_curves(ctx, m, np.median(d["chain"], axis=0), z_grid)
        except ValueError as exc:
            print(f"   ! {exc}; model left out of the Hubble diagram")
            continue
        ax1.plot(z_grid, mu_th, color=_color(i), lw=2.5, label=f"Best-fit {m}", zorder=3)
        ax2.plot(z_grid, H_th, color=_color(i), lw=2.5, label=f"Best-fit {m}", zorder=3)
        n_drawn += 1
    if n_drawn == 0:
        plt.close(fig)
        raise PlotSkipped("no model background could be evaluated")

    ax1.set_ylabel(r"Distance Modulus $\mu(z)$", fontsize=13)
    ax1.set_title("Cosmological Distance Curves & Expansion History Profiles", fontsize=14, pad=15)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax1.set_ylim(32, 47)
    ax2.set_xlabel(r"Redshift ($z$)", fontsize=13)
    ax2.set_ylabel(r"Hubble Parameter $H(z)$ $\rm [km/s/Mpc]$", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax2.set_xlim(0, 2.3)
    ax2.set_ylim(50, 250)
    fig.tight_layout()
    _save(fig, os.path.join(ctx.comparison_dir(), f"hubble_diagram_comparison_{ctx.run_mode}.pdf"))
    return fig
