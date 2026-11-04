# plot_selector.py
"""
Registry of every graph the pipeline can make, and the dispatcher that runs a chosen subset.

Two kinds of plot (scope):
    "model"       one figure set per run tag, from that run's own chain  -> graphs/<tag>/
    "comparison"  one figure over several run tags                       -> graphs/Model_Comparisons/

Used by
    main.py        PLOTS = [...] in the switchboard (model-scope plots, made right after sampling)
    make_plots.py  command line / interactive menu, from chains already on disk
    compare_posteriors.py, plot_hubble_diagram.py, plot_walk_traces.py  (thin wrappers, old entry points)

This module must NOT import main.py (main imports this); settings travel in a PlotContext.
"""
import os
import re
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt

import plotting
import comparison_plots as cp
from comparison_plots import PlotSkipped
from run_config import get_run_config, split_tag

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_HUBBLE_MODELS = ("Custom_Universal", "param_Hubble_fixed_delta", "param_Hubble_fixed_alpha")


# ════════════════════════════════════════════════════════════════════════════
# CONTEXT
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PlotContext:
    run_mode: str                       # e.g. "snia+cc+bao+clu"; part of every chain / figure file name
    fixed_delta: float = 0.050          # main.FIXED_DELTA / FIXED_ALPHA (custom Hubble models)
    fixed_alpha: float = 0.170
    fr_model: str = "GR"                # main.FR_MODEL / FR_Z_INIT / FR_OMEGA_M_ANCHOR: the chain files do not
    fr_z_init: float = 40.0             # record these, so hubble-diagram backgrounds for f(R) chains assume them
    fr_omega_m_anchor: float = 0.3
    n_walkers: int = 64                 # needed to un-flatten chains for the trace plot (not stored in the .npy)
    hist_bins: int = 16                 # 1D histogram overlays: fewer bins = smoother
    params: tuple = cp.DEFAULT_PARAMS   # comparison plots (hist, box, violin): parameters to compare, by name
    pair: tuple = cp.DEFAULT_PAIR       # contour plots: (x, y) parameter names
    reference: Optional[str] = None     # contour_vs_ref: reference run tag (None -> LCDM if selected, else first model)
    root: str = SCRIPT_DIR              # where chain_*.npy live (also searched: <root>/chains/)
    graphs_dir: Optional[str] = None    # default <root>/graphs
    show: bool = True                   # open the figure windows at the end

    @classmethod
    def from_main(cls, m, **overrides):
        """Build a context from the switchboard constants of the main module."""
        kw = dict(run_mode=m.RUN_MODE, fixed_delta=m.FIXED_DELTA, fixed_alpha=m.FIXED_ALPHA,
                  fr_model=m.FR_MODEL, fr_z_init=m.FR_Z_INIT, fr_omega_m_anchor=m.FR_OMEGA_M_ANCHOR,
                  n_walkers=m.N_WALKERS)
        kw.update(overrides)
        return cls(**kw)

    def _graphs(self):
        return self.graphs_dir or os.path.join(self.root, "graphs")

    def model_dir(self, tag):
        path = os.path.join(self._graphs(), tag)
        os.makedirs(path, exist_ok=True)
        return path

    def comparison_dir(self):
        path = os.path.join(self._graphs(), "Model_Comparisons")
        os.makedirs(path, exist_ok=True)
        return path


# ════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PlotSpec:
    key: str
    scope: str                          # "model" | "comparison"
    title: str
    func: Callable
    applies: Optional[Callable] = None  # model scope: (tag, cfg) -> None if OK, else reason string


def _needs_potential(tag, cfg):
    if cfg["potential_func"] is None:
        return "no potential reconstruction for this model (none defined, or an f(R) run)"


def _needs_custom_hubble(tag, cfg):
    if cfg["parameterisation"] not in CUSTOM_HUBBLE_MODELS or cfg["fR_model"] != "GR":
        return f"only defined for {', '.join(CUSTOM_HUBBLE_MODELS)}"


PLOTS = {s.key: s for s in [
    PlotSpec("corner", "model", "Corner plot of all sampled parameters",
             lambda ctx, tag, chain, cfg: plotting.plot_corner(chain, cfg["labels"], tag, ctx.model_dir(tag), ctx.run_mode)),
    PlotSpec("traces", "model", "Walker traces (convergence diagnostic)",
             lambda ctx, tag, chain, cfg: plotting.plot_traces(chain, cfg["labels"], tag, ctx.model_dir(tag),
                                                               ctx.run_mode, ctx.n_walkers)),
    PlotSpec("potential", "model", "Dark-energy potential V(phi) reconstruction",
             lambda ctx, tag, chain, cfg: plotting.plot_dark_energy_potential(chain, tag, cfg["potential_func"],
                                                                              ctx.model_dir(tag), ctx.run_mode),
             applies=_needs_potential),
    PlotSpec("omega_components", "model", "Omega_1(z), Omega_2(z) evolution (custom Hubble models)",
             lambda ctx, tag, chain, cfg: plotting.plot_and_save_omega_components(chain, tag, ctx.model_dir(tag), ctx.run_mode,
                                                                                  ctx.fixed_delta, ctx.fixed_alpha),
             applies=_needs_custom_hubble),
    PlotSpec("hist", "comparison", "1D posterior overlay, one figure per chosen parameter (params)", cp.plot_hist),
    PlotSpec("box", "comparison", "Median +- 1 sigma box plot, one panel per chosen parameter (params)", cp.plot_box),
    PlotSpec("violin", "comparison", "Violin plot, one panel per chosen parameter (params)", cp.plot_violin),
    PlotSpec("contour_vs_ref", "comparison", "2D contours of each model vs a reference model (pair, reference)", cp.plot_contour_vs_ref),
    PlotSpec("contour_all", "comparison", "2D contours of all models together (pair)", cp.plot_contour_all),
    PlotSpec("hubble", "comparison", "Hubble diagram: best-fit mu(z), H(z) vs SNIa + CC data", cp.plot_hubble),
]}
ALIASES = {"contour_vs_lcdm": "contour_vs_ref", "h0_hist": "hist", "matter_hist": "hist"}   # names from the first version
GROUPS = {"all": list(PLOTS),
          "model": [k for k, s in PLOTS.items() if s.scope == "model"],
          "comparison": [k for k, s in PLOTS.items() if s.scope == "comparison"]}


def resolve_keys(requested, scope=None):
    """
    Expand group names ('all', 'model', 'comparison'), drop duplicates, validate.
    scope='model' additionally rejects comparison plots (used by main.py, which has a single chain).
    """
    keys = []
    for k in requested:
        for kk in GROUPS.get(ALIASES.get(k, k), [ALIASES.get(k, k)]):
            if kk not in PLOTS:
                raise ValueError(f"Unknown plot '{kk}'. Plots: {list(PLOTS)}; groups: {list(GROUPS)}")
            if scope and PLOTS[kk].scope != scope:
                raise ValueError(f"Plot '{kk}' is a {PLOTS[kk].scope}-scope plot and cannot be used here "
                                 f"(allowed: {GROUPS[scope]}). Comparison plots need several chains: use make_plots.py.")
            if kk not in keys:
                keys.append(kk)
    return keys


def describe_plots():
    lines = []
    for scope, head in (("model", "Per-model plots (graphs/<tag>/)"), ("comparison", "Comparison plots (graphs/Model_Comparisons/)")):
        lines.append(head)
        lines += [f"  {k:17s} {PLOTS[k].title}" for k in GROUPS[scope]]
    lines += ["", "Parameters are chosen by name: H0, Omega_m, Omega_0, w0, wa, alpha_fR, n_fR, Lambda_fR, ... and",
              "'matter' (= Omega_m or Omega_0).  Models lacking a chosen parameter are left out of that panel/figure.",
              "Any mix of GR and f(R) runs can be compared; f(R) runs have no Omega_m (anchored), so compare them",
              "in e.g. H0 / alpha_fR / Lambda_fR.  --list shows what each chain samples."]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# CHAIN DISCOVERY
# ════════════════════════════════════════════════════════════════════════════
def find_chains(run_mode, root=SCRIPT_DIR):
    """{run_tag: path} for every chain_<tag>_<run_mode>.npy in <root> and <root>/chains (newest file wins)."""
    suffix, found = f"_{run_mode}.npy", {}
    for folder in (root, os.path.join(root, "chains")):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith("chain_") and name.endswith(suffix):
                path, tag = os.path.join(folder, name), name[len("chain_"):-len(suffix)]
                if tag not in found or os.path.getmtime(path) > os.path.getmtime(found[tag]):
                    found[tag] = path
    return dict(sorted(found.items()))


def available_run_modes(root=SCRIPT_DIR):
    modes = set()
    for folder in (root, os.path.join(root, "chains")):
        if os.path.isdir(folder):
            modes |= {re.sub(r"\.npy$", "", n).rsplit("_", 1)[1] for n in os.listdir(folder)
                      if n.startswith("chain_") and n.endswith(".npy") and "_" in n[6:]}
    return sorted(modes)


# ════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ════════════════════════════════════════════════════════════════════════════
def run_plots(requested, ctx, chains):
    """
    Make the requested plots.
        requested : plot keys / group names
        chains    : {run_tag: flat chain array}, in the order the models should appear in comparisons
    A plot that does not apply (or a model with a broken chain) is skipped with a reason; an exception
    inside one plot is reported and does not stop the others.  Returns [(plot, target, status, detail)],
    status in {"ok", "skipped", "FAILED"}.  Opens the figure windows at the end if ctx.show.
    """
    keys = resolve_keys(requested)
    results = []

    def attempt(key, target, call):
        try:
            call()
            results.append((key, target, "ok", ""))
        except PlotSkipped as exc:
            results.append((key, target, "skipped", str(exc)))
        except Exception as exc:                       # keep going: report at the end
            traceback.print_exc()
            results.append((key, target, "FAILED", f"{type(exc).__name__}: {exc}"))
        if not ctx.show:
            plt.close("all")

    cfgs, good = {}, {}
    for tag, chain in chains.items():
        try:
            cfgs[tag] = get_run_config(*split_tag(tag), ctx.fr_omega_m_anchor)
            if chain.ndim != 2 or chain.shape[1] != len(cfgs[tag]["labels"]):
                raise ValueError(f"chain shape {chain.shape} does not match the {len(cfgs[tag]['labels'])} "
                                 f"parameters {cfgs[tag]['labels']}")
            good[tag] = chain
        except Exception as exc:
            for k in keys:
                results.append((k, tag, "skipped" if PLOTS[k].scope == "model" else "n/a", f"bad chain/tag: {exc}"))
            print(f"! '{tag}' dropped: {exc}")

    for k in (k for k in keys if PLOTS[k].scope == "model"):
        for tag, chain in good.items():
            reason = PLOTS[k].applies(tag, cfgs[tag]) if PLOTS[k].applies else None
            if reason:
                results.append((k, tag, "skipped", reason))
                continue
            print(f"\n[{k}] {tag}")
            attempt(k, tag, lambda k=k, tag=tag, chain=chain: PLOTS[k].func(ctx, tag, chain, cfgs[tag]))

    comparison_keys = [k for k in keys if PLOTS[k].scope == "comparison"]
    if comparison_keys:
        data = {}
        for tag, chain in good.items():
            data[tag] = cp.extract_summary(tag, chain)
        for k in comparison_keys:
            if not data:
                results.append((k, "-", "skipped", "no usable chains"))
                continue
            print(f"\n[{k}] {', '.join(data)}")
            attempt(k, f"{len(data)} models", lambda k=k: PLOTS[k].func(ctx, data))

    print("\n── Plot summary " + "─" * 60)
    grouped = {}                                   # one line per (plot, status, reason), listing its targets
    for key, target, status, detail in results:
        grouped.setdefault((key, status, detail), []).append(target)
    for (key, status, detail), targets in grouped.items():
        print(f"  {status:8s} {key:17s} " + ("; ".join(targets) if status == "ok" else f"{len(targets)}x: {detail}"))
        if status != "ok" and len(targets) <= 3:
            print(f"{'':27s}{'; '.join(targets)}")
    if ctx.show and any(r[2] == "ok" for r in results):
        print("\nOpening interactive figure windows... close them to exit.")
        plt.show()
    return results
