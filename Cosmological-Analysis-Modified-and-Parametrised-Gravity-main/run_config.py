# run_config.py
"""
Combines a parameterisation (parameterisations.py) and an f(R) model (fR.py)
into the single run configuration used by main.py and the analysis scripts.

Run tag (used in chain / output file names):
    FR_MODEL == "GR"  ->  "<parameterisation>"            (unchanged from the old pipeline)
    otherwise         ->  "<parameterisation>__<fR_model>"
    pinned f(R) parameters are appended: "LCDM__power_law__n=0.5"
"""
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
        if fr_model == "power_law" and nm == "n" and not (fR.FR_PRIOR_BOUNDS["n_power"][0] < val < fR.FR_PRIOR_BOUNDS["n_power"][1] and val != 1.0):
            raise ValueError(f"Fixed n={val} is outside the prior range {fR.FR_PRIOR_BOUNDS['n_power']} or equals 1 (f = (1+alpha)R, i.e. GR).")

    names = fcfg["param_names"]
    if fr_fixed and all(nm in fr_fixed for nm in (names if fr_model == "power_law" else ["alpha"] if fr_model == "R_log_R" else [])):
        full = tuple(fr_fixed[nm] for nm in names)
        if not np.isfinite(fcfg["prior_func"](full)):
            rule = {"power_law": "f_RR > 0 needs alpha < 0 for 0 < n < 1 and alpha > 0 for n > 1 (and alpha, n inside their prior bounds)",
                    "R_log_R": "f_RR > 0 needs alpha > 0 (inside the prior bounds)"}[fr_model]
            raise ValueError(f"FR_FIXED_PARAMS={fr_fixed} violates the f(R) prior/stability condition: {rule}. "
                             f"Bounds: {fR.FR_PRIOR_BOUNDS}")

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
