# fR.py
"""
Metric f(R) gravity: model registry and background solver.

Choose the model with FR_MODEL in main.py.  The three models are the ones worked
out symbolically in Modified_Gravity_Models/fR_friedmann.py:

    "GR"         f = R                  (no modification; the parameterisation
                                         is the background, as in the old pipeline)
    "power_law"  f = R + alpha R^n      free: alpha, n
    "R_log_R"    f = R (1 + alpha ln R) free: alpha

HOW THE f(R) MODEL ENTERS THE LIKELIHOOD
----------------------------------------
For a modified model the background is NOT taken from the parameterisation.
Instead H(z) is obtained by integrating the f(R) field equation with matter only.
The chosen parameterisation only supplies
    * the initial conditions  u_i = E^2(z_init)  and  u'_i = dE^2/dlna  at z_init,
    * the matter density Omega_m (via the registry's omega_m_func).

Equations (flat FLRW, signature -+++, N = ln a, ' = d/dN, u = H^2/H0^2,
all curvatures in units of H0^2, kappa rho_m = 3 Omega_m e^{-3N}):

    R      = 3 u' + 12 u
    F      = f_R(R),   F' = f_RR(R) R'
    kappa rho = 3 F u + (f - R F)/2 + 3 u F'        (tt equation, = kappa rho_m)
    kappa p   = F (R/6 + u) - f/2 - f_RRR u R'^2
                - f_RR (u R'' + u' R'/2) - 2 u f_RR R'   (rr equation, = 0 for matter)

kappa rho = kappa rho_m is solved for R' = (3u'' + 12u'), giving a 2nd-order ODE
for u(N).  kappa p = 0 then holds automatically (Bianchi identity); check_fR.py
verifies this numerically as an independent test of the ODE.

GAUGE FIXING OF Omega_m (important)
-----------------------------------
The equation is covariant under R -> lambda^2 R, which maps (Omega_m, alpha) to
another pair and leaves E^2(z) = u(z)/u(0) unchanged.  For f = R + alpha R^n the
shape depends only on alpha * Omega_m^(n-1) (checked numerically: identical to
<1e-3, the residue being the small dark-energy term in the initial conditions).
Sampling both would give a flat, curved ridge and a meaningless Omega_m posterior.
Therefore, in f(R) runs the parameterisation's Omega_m is NOT sampled: it is held
at FR_OMEGA_M_ANCHOR (main.py) and alpha is expressed in those anchor units.  The
physical present-day matter fraction Omega_m/u(0) is a derived quantity.

NORMALISATION / MEANING OF THE SAMPLED H0
-----------------------------------------
The ODE is integrated in "anchor" units (the parameterisation's H0 = 1).  In a
pure-matter f(R) universe u(N=0) != 1 in general.  The sampled H0 is defined as
the TRUE present-day Hubble constant:  E^2(z) = u(z)/u(0),  so E^2(0) = 1 and
H(z) = H0 sqrt(E^2).  Consequently Omega_m (anchor) is a gauge value; the physical
present-day matter fraction is Omega_m/u(0), reported as a derived quantity after
the run.  alpha is likewise in anchor units (R measured in units of the anchor H0^2).

LIMITATIONS
-----------
* f = R cannot be integrated (f_RR = 0 makes the tt equation a constraint), so
  "GR" means GR + the parameterisation's dark energy fluid.
* The parameterisation's dark-energy parameters (w0, wa, ...) enter only through
  the initial conditions at z_init, where DE is negligible -> they are expected
  to be almost unconstrained in f(R) runs.  LCDM (Omega_m only) is the natural
  companion.
* Stability: f_RR > 0 is imposed in the prior (see each model's prior function).
* Initial-condition dependence.  The parameterisation's (u, u') at FR_Z_INIT are only
  a valid starting point if the f(R) model has a GR-like matter era at early times.
    - "power_law" (0<n<1): alpha R^n / R -> 0 as R -> infinity, so E^2(z) converges
      as FR_Z_INIT grows (changes by ~10x per doubling; FR_Z_INIT = 40 gives ~5e-4).
    - "R_log_R": F = 1 + alpha (ln R + 1) does NOT tend to 1, there is no GR matter-era
      attractor, and E^2(z) does NOT converge with FR_Z_INIT (10-50% changes found).
      Its results depend on the arbitrary starting redshift and must not be trusted.
  main.py prints this convergence test (ic_sensitivity) for every f(R) run.
"""
import numpy as np
from scipy.integrate import solve_ivp


# ════════════════════════════════════════════════════════════════════════════
# PRIOR BOUNDS  (uniform, hard-wall; edit here)
# ════════════════════════════════════════════════════════════════════════════
FR_PRIOR_BOUNDS = {
    "alpha_power": (-10.0, 10.0),
    "n_power"    : (  0.1,  2.0),
    "alpha_log"  : (  0.0,  3.0),
}

def _in(val, key):
    lo, hi = FR_PRIOR_BOUNDS[key]
    return lo <= val <= hi          # closed interval, so a pinned value may sit exactly on a bound (e.g. n = 2)


# ════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS:  derivatives (f, f_R, f_RR, f_RRR) of f(R), R in H0^2 units
# ════════════════════════════════════════════════════════════════════════════
def _derivs_power_law(theta):
    alpha, n = theta
    return (lambda R: R + alpha * R**n,
            lambda R: 1.0 + alpha * n * R**(n - 1.0),
            lambda R: alpha * n * (n - 1.0) * R**(n - 2.0),
            lambda R: alpha * n * (n - 1.0) * (n - 2.0) * R**(n - 3.0))

def _prior_power_law(theta):
    alpha, n = theta
    # f_RR = alpha n (n-1) R^(n-2) > 0 for R > 0  <=>  alpha n (n-1) > 0
    if _in(alpha, "alpha_power") and _in(n, "n_power") and alpha * n * (n - 1.0) > 0.0:
        return 0.0
    return -np.inf


def _derivs_R_log_R(theta):
    alpha, = theta
    return (lambda R: R * (1.0 + alpha * np.log(R)),
            lambda R: 1.0 + alpha * (np.log(R) + 1.0),
            lambda R: alpha / R,
            lambda R: -alpha / R**2)

def _prior_R_log_R(theta):
    alpha, = theta
    # f_RR = alpha / R > 0  <=>  alpha > 0  (alpha_log lower bound is 0)
    return 0.0 if _in(alpha, "alpha_log") else -np.inf


# ════════════════════════════════════════════════════════════════════════════
# f(R) REGISTRY
# ════════════════════════════════════════════════════════════════════════════
#   param_names: names of the f(R) parameters, in order (usable in FR_FIXED_PARAMS)
#   labels     : LaTeX names of the extra MCMC parameters (appended after the
#                parameterisation's labels)
#   p0_center / p0_scatter : walker initialisation for those parameters
#   prior_func : theta_fR -> 0.0 or -inf
#   derivs     : theta_fR -> (f, f_R, f_RR, f_RRR);  None for GR
FR_REGISTRY = {
    "GR": {
        "description": "f(R) = R: General Relativity; dark energy comes from the parameterisation",
        "param_names": [],
        "labels"     : [],
        "p0_center"  : np.array([]),
        "p0_scatter" : np.array([]),
        "prior_func" : lambda theta: 0.0,
        "derivs"     : None,
    },
    "power_law": {
        "description": "f(R) = R + alpha R^n",
        "param_names": ["alpha", "n"],
        "labels"     : [r"$\alpha_{f(R)}$", r"$n_{f(R)}$"],
        "p0_center"  : np.array([-2.7, 0.29]),    # ~best fit to SNIa+CC+BAO+clustering; alpha<0, 0<n<1 is the stable branch
        "p0_scatter" : np.array([ 0.15, 0.01]),
        "prior_func" : _prior_power_law,
        "derivs"     : _derivs_power_law,
    },
    "R_log_R": {
        "description": "f(R) = R (1 + alpha ln R)",
        "param_names": ["alpha"],
        "labels"     : [r"$\alpha_{f(R)}$"],
        "p0_center"  : np.array([0.45]),
        "p0_scatter" : np.array([0.05]),
        "prior_func" : _prior_R_log_R,
        "derivs"     : _derivs_R_log_R,
    },
}


def make_fr_expander(fr_model, fixed=None):
    """
    Returns (expand, sampled_idx).  `fixed` = {param_name: value} pins f(R) parameters
    (FR_FIXED_PARAMS in main.py); they are removed from the MCMC vector.
    expand(theta_sampled) -> full f(R) parameter tuple, in param_names order.
    """
    fixed = dict(fixed or {})
    names = FR_REGISTRY[fr_model]["param_names"]
    unknown = set(fixed) - set(names)
    if unknown:
        raise ValueError(f"FR_FIXED_PARAMS has {sorted(unknown)}, but f(R) model '{fr_model}' "
                         f"only has parameters {names}.")
    sampled_idx = [i for i, nm in enumerate(names) if nm not in fixed]

    def expand(theta_sampled):
        it = iter(theta_sampled)
        return tuple(fixed[nm] if nm in fixed else next(it) for nm in names)
    return expand, sampled_idx


# ════════════════════════════════════════════════════════════════════════════
# kappa rho, kappa p  (closed forms from fR_friedmann.py, flat, N-derivatives)
# ════════════════════════════════════════════════════════════════════════════
def kappa_rho(derivs, R, R_N, u):
    """tt equation: kappa*rho = 3 F u + (f - R F)/2 + 3 u f_RR R'."""
    f, fR, fRR, _ = derivs
    return 3.0 * fR(R) * u + 0.5 * (f(R) - R * fR(R)) + 3.0 * u * fRR(R) * R_N

def kappa_p(derivs, R, R_N, R_NN, u, u_N):
    """rr equation: kappa*p (Rdot = H R', Rddot = u R'' + u' R'/2)."""
    f, fR, fRR, fRRR = derivs
    return (fR(R) * (R / 6.0 + u) - 0.5 * f(R)
            - fRRR(R) * u * R_N**2
            - fRR(R) * (u * R_NN + 0.5 * u_N * R_N)
            - 2.0 * u * fRR(R) * R_N)

def R_N_from_matter(derivs, R, u, kappa_rho_m):
    """Invert kappa_rho(R, R', u) = kappa_rho_m for R' (kappa_rho is linear in R')."""
    fRR = derivs[2]
    rho_at_zero = kappa_rho(derivs, R, 0.0, u)
    return (kappa_rho_m - rho_at_zero) / (3.0 * u * fRR(R))


# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND OBJECT + SOLVER
# ════════════════════════════════════════════════════════════════════════════
class Background:
    """E2(z): vectorised E^2(z) = H^2/H0^2 with E^2(0)=1.  u0: u(N=0) in anchor units.
    omega_m: anchor Omega_m used for the integration (None for GR backgrounds).
    sol: scipy dense solution y(N) = (u, u') in anchor units (None for GR); for diagnostics."""
    def __init__(self, E2, u0=1.0, omega_m=None, sol=None):
        self.E2 = E2
        self.u0 = u0
        self.omega_m = omega_m
        self.sol = sol


MAX_RHS_EVALS = 5000       # typical solve: ~300-600; stiff corners of parameter space blow past this
U0_RANGE = (1e-2, 1e2)     # u(0) outside this <=> physical Omega_m = Omega_m/u(0) outside ~(0.003, 30): runaway


def solve_background(E2_param, Omega_m, derivs, z_init, rtol=1e-8, return_reason=False):
    """
    Integrate the f(R) background from z_init to z=0 (see module docstring).
    Returns a Background, or None if the integration fails or leaves the
    physical domain (u <= 0, R <= 0, blow-up, non-finite output, too many function
    evaluations (stiff), or u(0) outside U0_RANGE (runaway solution)).
    With return_reason=True returns (background_or_None, reason_string_or_None).
    """
    def fail(msg):
        return (None, msg) if return_reason else None

    N_i = -np.log1p(z_init)
    h = 1e-4
    u_i = float(E2_param(z_init))
    v_i = -(1.0 + z_init) * (float(E2_param(z_init + h)) - float(E2_param(z_init - h))) / (2.0 * h)
    R_i = 3.0 * v_i + 12.0 * u_i
    if not (np.isfinite(u_i) and np.isfinite(v_i) and u_i > 0.0 and R_i > 0.0):
        return fail("invalid initial conditions from the parameterisation (u_i or R_i <= 0)")

    R_FLOOR = 1e-12          # keeps rhs finite inside a step; events catch the violation

    class _TooStiff(Exception):
        pass
    n_eval = [0]

    def rhs(N, y):
        n_eval[0] += 1
        if n_eval[0] > MAX_RHS_EVALS:
            raise _TooStiff
        u, v = y
        R = max(3.0 * v + 12.0 * u, R_FLOOR)
        u = max(u, R_FLOOR)
        R_N = R_N_from_matter(derivs, R, u, 3.0 * Omega_m * np.exp(-3.0 * N))
        return [v, R_N / 3.0 - 4.0 * v]          # R' = 3u'' + 12u'

    def leaves_domain(N, y):                      # terminal: u<=0 or R<=0
        return min(y[0], 3.0 * y[1] + 12.0 * y[0]) - 1e-8
    leaves_domain.terminal, leaves_domain.direction = True, -1

    def blows_up(N, y):                           # terminal: |u'/u| huge
        return 1e3 - abs(y[1]) / max(abs(y[0]), R_FLOOR)
    blows_up.terminal, blows_up.direction = True, -1

    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(rhs, (N_i, 0.0), [u_i, v_i], method="LSODA", rtol=rtol, atol=1e-10,
                            dense_output=True, events=(leaves_domain, blows_up))
    except _TooStiff:
        return fail(f"stiff: more than {MAX_RHS_EVALS} right-hand-side evaluations")
    except Exception as exc:
        return fail(f"integrator error: {exc}")
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        z_stop = np.expm1(-sol.t[-1])
        if len(sol.t_events[1]):
            return fail(f"solution blew up at z = {z_stop:.3g} (|u'/u| > 1e3: exponential instability)")
        if len(sol.t_events[0]):
            return fail(f"solution left the physical domain (u <= 0 or R <= 0) at z = {z_stop:.3g}")
        return fail(f"integrator failed at z = {z_stop:.3g}: {sol.message}")
    u0 = float(sol.y[0, -1])
    if not (np.isfinite(u0) and U0_RANGE[0] < u0 < U0_RANGE[1]):
        return fail(f"runaway solution: u(0) = {u0:.3g} outside {U0_RANGE} "
                    f"(physical Omega_m = Omega_m/u(0) = {Omega_m / u0:.2g}); the f(R) correction is not "
                    f"small at the starting redshift, so the GR-like initial conditions are invalid")

    def E2(z):
        z = np.asarray(z, dtype=float)
        if np.any(z > z_init):
            raise ValueError(f"E2(z) requested at z={z.max():.3g} > FR_Z_INIT={z_init}; "
                             f"raise FR_Z_INIT in main.py.")
        out = sol.sol(-np.log1p(z))[0] / u0
        return out if z.ndim else float(out)
    bg = Background(E2, u0, Omega_m, sol)
    return (bg, None) if return_reason else bg


# ════════════════════════════════════════════════════════════════════════════
# DERIVED QUANTITIES FOR THE POSTERIOR SUMMARY
# ════════════════════════════════════════════════════════════════════════════
def derived_summary(flat_chain, build_background, n_param_theta, n_samples=300, seed=0):
    """
    Re-solves the background for a random subset of chain samples and returns
    (lines, derived) where derived = {name: array}.  Reports the physical
    present-day matter fraction Omega_m/u(0) and the anchor Hubble constant
    H0/sqrt(u(0)) (the parameterisation's H0 that sets the initial conditions).
    n_param_theta = number of SAMPLED parameterisation parameters (excl. H0).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat_chain), size=min(n_samples, len(flat_chain)), replace=False)
    Om_phys, H_anchor = [], []
    for i in idx:
        p = flat_chain[i]
        theta_p = tuple(p[1:1 + n_param_theta])
        theta_fr = tuple(p[1 + n_param_theta:])
        bg = build_background(theta_p, theta_fr)
        if bg is None:
            continue
        Om_phys.append(bg.omega_m / bg.u0)
        H_anchor.append(p[0] / np.sqrt(bg.u0))
    derived = {"Omega_m (present, physical) = Omega_m/u(0)": np.array(Om_phys),
               "H0 (anchor, sets ICs) = H0/sqrt(u(0))": np.array(H_anchor)}
    lines = []
    for name, arr in derived.items():
        if len(arr) == 0:
            lines.append(f"{name}: no valid samples")
            continue
        med, lo, hi = np.median(arr), np.percentile(arr, 16), np.percentile(arr, 84)
        lines.append(f"{name}: {med:.4f} +{hi - med:.4f} / -{med - lo:.4f}   ({len(arr)} samples)")
    return lines, derived


def ic_sensitivity(build_a, build_b, theta_p, theta_fr, z_max=2.4, n=100):
    """
    max |E2_a(z)/E2_b(z) - 1| for z in [0, z_max], where a and b are backgrounds built
    with different FR_Z_INIT (e.g. z_init and 2 z_init).  None if either solve fails.
    Small -> the result is independent of the starting redshift.
    """
    a, b = build_a(theta_p, theta_fr), build_b(theta_p, theta_fr)
    if a is None or b is None:
        return None
    z = np.linspace(0.0, z_max, n)
    return float(np.max(np.abs(a.E2(z) / b.E2(z) - 1.0)))
