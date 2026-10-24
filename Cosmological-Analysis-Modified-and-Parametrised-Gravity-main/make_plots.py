# make_plots.py
"""
Choose which graphs to make from MCMC chains that are already on disk (no re-sampling).

    python make_plots.py                       # interactive menu
    python make_plots.py --list                # plots + chains available for the run mode
    python make_plots.py --plots corner violin --models LCDM wCDM
    python make_plots.py --plots hist box violin --params H0 alpha_fR --models LCDM__power_law__alpha=-2__n=0.5 wCDM
    python make_plots.py --plots contour_vs_ref --reference wCDM --pair H0 w0     # any model as the reference
    python make_plots.py --plots comparison --no-show
    python make_plots.py --plots traces --walkers 64 --run-mode snia+cc+bao+clu

Plots: corner, traces, potential, omega_components (one set per model);
       hist, box, violin, contour_vs_ref, contour_all, hubble (several models together).
Groups: all, model, comparison.  Any mix of GR and f(R) runs can be compared, in any parameters they share
(--params for hist/box/violin, --pair for contours, --reference for contour_vs_ref).  Plots that do not
apply to a model (e.g. potential for an f(R) run) are skipped with a reason.
Command-line options override the switchboard below.
"""
import argparse
import os
import sys

# ════════════════════════════════════════════════════════════════════════════
# SWITCHBOARD — used when the corresponding command-line option is not given
# ════════════════════════════════════════════════════════════════════════════
PLOTS_TO_MAKE = None   # e.g. ["corner", "violin"] or ["comparison"]; None -> ask in a menu
MODELS        = None   # run tags, e.g. ["LCDM", "wCDM", "LCDM__power_law__alpha=-2__n=0.5"]; None -> all chains found
RUN_MODE      = None   # None -> main.RUN_MODE (set by the USE_* dataset toggles)
PARAMS        = None   # hist/box/violin: parameter names, e.g. ["H0", "w0", "alpha_fR"]; None -> ["H0", "matter"]
PAIR          = None   # contours: [x, y] parameter names, e.g. ["H0", "alpha_fR"]; None -> ["H0", "matter"]
REFERENCE     = None   # contour_vs_ref: reference run tag; None -> LCDM if selected, else the first model
# ════════════════════════════════════════════════════════════════════════════

import numpy as np

import main as cfg_main
from comparison_plots import DEFAULT_PAIR, DEFAULT_PARAMS, parameters_of
from plot_selector import (PLOTS, PlotContext, available_run_modes, describe_plots,
                           find_chains, resolve_keys, run_plots)


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        sys.exit("\nNo input available (not an interactive terminal); pass --plots.")


def _pick(items, prompt):
    """Ask for 1-based numbers / names (space or comma separated), 'a' = all; returns the chosen items."""
    while True:
        raw = _ask(prompt).replace(",", " ").split()
        if not raw:
            continue
        if raw[0].lower() in ("a", "all"):
            return list(items)
        try:
            return [items[int(r) - 1] if r.isdigit() else items[items.index(r)] for r in raw]
        except (IndexError, ValueError):
            print(f"  Not understood: {' '.join(raw)}  (numbers 1-{len(items)}, names, or 'a')")


def choose_plots():
    keys = list(PLOTS)
    print("\nAvailable plots:")
    for i, k in enumerate(keys, 1):
        print(f"  {i:2d}. {k:17s} [{PLOTS[k].scope:10s}] {PLOTS[k].title}")
    return resolve_keys(_pick(keys, "Select plots (numbers/names, 'a' = all): "))


def choose_models(found, needs_models):
    tags = list(found)
    if not needs_models:
        return tags
    print("\nChains found:")
    for i, t in enumerate(tags, 1):
        print(f"  {i:2d}. {t}")
    return _pick(tags, "Select models (numbers/names, 'a' = all): ")


def choose_options(keys, tags, ctx_kw):
    """Menu mode: ask for the comparison parameters / reference / pair (Enter = default)."""
    wants = lambda *ks: any(k in keys for k in ks)
    shared = sorted({n for t in tags for n in parameters_of(t)})
    if not wants("hist", "box", "violin", "contour_vs_ref", "contour_all"):
        return
    print(f"\nParameters sampled by the selected models: {', '.join(shared)}")
    if wants("hist", "box", "violin"):
        raw = _ask(f"Parameters to compare in hist/box/violin [Enter = {' '.join(DEFAULT_PARAMS)}]: ").replace(",", " ").split()
        if raw: ctx_kw["params"] = tuple(raw)
    if wants("contour_vs_ref", "contour_all"):
        raw = _ask(f"Contour axes: two parameters [Enter = {' '.join(DEFAULT_PAIR)}]: ").replace(",", " ").split()
        if len(raw) == 2: ctx_kw["pair"] = tuple(raw)
    if wants("contour_vs_ref") and len(tags) > 1:
        raw = _ask(f"Reference model for contour_vs_ref [Enter = {'LCDM' if 'LCDM' in tags else tags[0]}]: ")
        if raw: ctx_kw["reference"] = raw


def main():
    ap = argparse.ArgumentParser(description="Make a chosen subset of the pipeline's graphs from saved chains.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=describe_plots())
    ap.add_argument("--plots", nargs="+", metavar="PLOT", help="plot keys or groups (all, model, comparison)")
    ap.add_argument("--models", nargs="+", metavar="TAG", help="run tags to use (default: all chains found)")
    ap.add_argument("--params", nargs="+", metavar="NAME", help="parameters for hist/box/violin (default: H0 matter)")
    ap.add_argument("--pair", nargs=2, metavar=("X", "Y"), help="parameters on the contour axes (default: H0 matter)")
    ap.add_argument("--reference", metavar="TAG", help="reference model for contour_vs_ref (default: LCDM if selected, else first)")
    ap.add_argument("--run-mode", help="dataset tag in the chain file names (default: main.RUN_MODE)")
    ap.add_argument("--walkers", type=int, help="walkers the chains were made with (default: main.N_WALKERS)")
    ap.add_argument("--bins", type=int, help="bins for the 1D histogram overlays (default 16)")
    ap.add_argument("--no-show", action="store_true", help="save the figures but do not open windows")
    ap.add_argument("--list", action="store_true", help="list plots and available chains, then exit")
    args = ap.parse_args()

    overrides = {}
    if args.walkers: overrides["n_walkers"] = args.walkers
    if args.bins:    overrides["hist_bins"] = args.bins
    if args.no_show: overrides["show"] = False
    if args.params or PARAMS: overrides["params"] = tuple(args.params or PARAMS)
    if args.pair or PAIR:     overrides["pair"] = tuple(args.pair or PAIR)
    if args.reference or REFERENCE: overrides["reference"] = args.reference or REFERENCE
    run_mode = args.run_mode or RUN_MODE or cfg_main.RUN_MODE
    ctx = PlotContext.from_main(cfg_main, run_mode=run_mode, **overrides)

    found = find_chains(run_mode, ctx.root)
    if args.list:
        print(describe_plots())
        print(f"\nChains for run mode '{run_mode}' ({len(found)}) and the parameters each samples:")
        for t, p in found.items():
            try:
                ps = " ".join(n for n in parameters_of(t))
            except Exception as exc:
                ps = f"?? ({exc})"
            print(f"  {t:50s} {ps}")
        print(f"Run modes with chains on disk: {available_run_modes(ctx.root) or 'none'}")
        return 0
    if not found:
        print(f"No chain_*_{run_mode}.npy files found in {ctx.root} or its chains/ folder.\n"
              f"Run modes with chains on disk: {available_run_modes(ctx.root) or 'none'}  (use --run-mode)")
        return 1

    requested = args.plots or PLOTS_TO_MAKE
    if requested:
        keys = resolve_keys(requested)
    elif sys.stdin.isatty():
        keys = choose_plots()
    else:
        print("No plots selected (not an interactive terminal). Use --plots; --list shows the options.")
        return 2

    wanted = args.models or MODELS
    if wanted is None:
        needs_models = sys.stdin.isatty() and not requested        # menu mode: let the user narrow the models too
        wanted = choose_models(found, needs_models)
    missing = [t for t in wanted if t not in found]
    for t in missing:
        print(f"! No chain for '{t}' with run mode '{run_mode}' -- skipped. (Found: {list(found)})")
    chains = {}
    for t in wanted:
        if t in found:
            print(f"Loading {os.path.relpath(found[t], ctx.root)}")
            chains[t] = np.load(found[t])
    if not chains:
        return 1

    if not requested and sys.stdin.isatty():                      # menu mode
        kw = {}
        choose_options(keys, list(chains), kw)
        for k, v in kw.items():
            setattr(ctx, k, v)
    if ctx.reference and ctx.reference not in chains:
        print(f"! reference '{ctx.reference}' is not among the selected models {list(chains)}; contour_vs_ref will be skipped")

    results = run_plots(keys, ctx, chains)
    return 1 if any(r[2] == "FAILED" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
