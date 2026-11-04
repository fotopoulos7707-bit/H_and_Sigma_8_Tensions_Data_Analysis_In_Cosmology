# run_config.py
"""
Combines a parameterisation (parameterisations.py) and an f(R) model (fR.py)
into the single run configuration used by main.py and the analysis scripts.

Run tag (used in chain / output file names):
    FR_MODEL == "GR"  ->  "<parameterisation>"            (unchanged from the old pipeline)
    otherwise         ->  "<parameterisation>__<fR_model>"
    pinned f(R) parameters are appended: "LCDM__power_law__n=0.5"
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

import parameterisations as P
import fR

TAG_SEP = "__"


def make_tag(parameterisation, fr_model, fr_fixed=None):
    if fr_model == "GR":
        return parameterisation
    tag = f"{parameterisation}{TAG_SEP}{fr_model}"
    for k, v in sorted((fr_fixed or {}).items()):
        tag += f"{TAG_SEP}{k}={v:g}"
    return tag


def split_tag(tag):
    """Inverse of make_tag: returns (parameterisation, fr_model, fr_fixed dict)."""
    parts = tag.split(TAG_SEP)
    if len(parts) == 1:
        return parts[0], "GR", {}
    fixed = {}
    for p in parts[2:]:
        k, v = p.split("=")
        fixed[k] = float(v)
    return parts[0], parts[1], fixed


def get_run_config(parameterisation, fr_model, fr_fixed=None, omega_m_anchor=0.3):
    """
    Validated, combined configuration.  Keys: everything in the parameterisation
    entry, with labels / p0_center / p0_scatter extended by the f(R) parameters, plus
        n_param_theta : number of parameterisation parameters (excluding H0)
        parameterisation, fR_model, tag, fR_description
    """
    if parameterisation not in P.PARAMETERISATION_REGISTRY:
        raise ValueError(f"Unknown PARAMETERISATION '{parameterisation}'. "
                         f"Options: {list(P.PARAMETERISATION_REGISTRY)}")
    if fr_model not in fR.FR_REGISTRY:
        raise ValueError(f"Unknown FR_MODEL '{fr_model}'. Options: {list(fR.FR_REGISTRY)}")

    pcfg = P.PARAMETERISATION_REGISTRY[parameterisation]
    fcfg = fR.FR_REGISTRY[fr_model]
    fr_fixed = dict(fr_fixed or {})
    _, fr_sampled_idx = fR.make_fr_expander(fr_model, fr_fixed)      # validates the names
    for nm, val in fr_fixed.items():
        if fr_model == "power_law" and nm == "n" and not (fR.FR_PRIOR_BOUNDS["n_power"][0] <= val <= fR.FR_PRIOR_BOUNDS["n_power"][1] and val != 1.0):
            raise ValueError(f"Fixed n={val} is outside the prior range {fR.FR_PRIOR_BOUNDS['n_power']} or equals 1 (f = (1+alpha)R, i.e. GR).")

    names = fcfg["param_names"]
    if names and all(nm in fr_fixed for nm in names):
        full = tuple(fr_fixed[nm] for nm in names)
        if not np.isfinite(fcfg["prior_func"](full)):
            rule = {"power_law": "f_RR > 0 needs alpha < 0 for 0 < n < 1 and alpha > 0 for n > 1 (and alpha, n inside their prior bounds)",
                    "R_log_R": "f_RR > 0 needs alpha > 0 (inside the prior bounds)",
                    "R_lambda_R2": "f_RR = 2 alpha > 0 needs alpha > 0, with Lambda and alpha inside their prior bounds",
                    "starobinsky": "f_RR = 2 alpha > 0 needs alpha > 0 (inside the prior bounds)",
                    "hu_sawicki": "f_RR = 4 Lambda^2 b/(R + Lambda b)^3 > 0 needs Lambda > 0 and b > 0, and a de Sitter attractor needs b < 2 (both inside the prior bounds)"}[fcfg.get("base_model", fr_model)]
            raise ValueError(f"FR_FIXED_PARAMS={fr_fixed} violates the f(R) prior/stability condition: {rule}. "
                             f"Bounds: {fR.FR_PRIOR_BOUNDS}")

    if fcfg.get("de_on_top") and parameterisation == "LCDM":
        raise ValueError(f"FR_MODEL '{fr_model}' adds the parameterisation's dark energy on top of Lambda, but LCDM's is a "
                         f"constant, i.e. a second cosmological constant degenerate with Lambda. Use wCDM, CPL, wowaCDM, JBP, "
                         f"HDE or wCDM_fixed_Omega_m.")
    if fr_model != "GR" and pcfg["omega_m_func"] is None:
        ok = [k for k, v in P.PARAMETERISATION_REGISTRY.items() if v["omega_m_func"] is not None]
        raise ValueError(f"PARAMETERISATION '{parameterisation}' has no standard conserved-matter "
                         f"component, so it cannot supply initial conditions for FR_MODEL "
                         f"'{fr_model}'. Compatible parameterisations: {ok}")

    labels = list(pcfg["labels"])
    p0_center, p0_scatter = np.array(pcfg["p0_center"], dtype=float), np.array(pcfg["p0_scatter"], dtype=float)
    idx = pcfg["omega_m_index"]
    if fr_model != "GR" and idx is not None:
        lo, hi = P.PRIOR_BOUNDS["Omega_m"]
        if not lo < omega_m_anchor < hi:
            raise ValueError(f"FR_OMEGA_M_ANCHOR={omega_m_anchor} outside the Omega_m prior ({lo}, {hi}).")
        k = 1 + idx                                  # column 0 is H0
        labels.pop(k); p0_center = np.delete(p0_center, k); p0_scatter = np.delete(p0_scatter, k)

    cfg = dict(pcfg)
    cfg["labels"] = labels + [fcfg["labels"][i] for i in fr_sampled_idx]
    cfg["p0_center"] = np.concatenate([p0_center, fcfg["p0_center"][fr_sampled_idx]])
    cfg["p0_scatter"] = np.concatenate([p0_scatter, fcfg["p0_scatter"][fr_sampled_idx]])
    cfg["fR_fixed"] = fr_fixed
    cfg["n_param_theta"] = len(labels) - 1           # sampled parameterisation parameters
    cfg["omega_m_anchor"] = omega_m_anchor if (fr_model != "GR" and idx is not None) else None
    if fr_model != "GR":
        cfg["potential_func"] = None      # scalar-field potential reconstruction is a GR+DE notion
    cfg.update(parameterisation=parameterisation, fR_model=fr_model,
               tag=make_tag(parameterisation, fr_model, fr_fixed), fR_description=fcfg["description"])
    return cfg
