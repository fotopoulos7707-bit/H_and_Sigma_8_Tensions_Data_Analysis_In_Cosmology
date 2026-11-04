# fR.py
"""
Metric f(R) gravity: model registry and background solver.

Choose the model with FR_MODEL in main.py.  The three models are the ones worked
out symbolically in Modified_Gravity_Models/fR_friedmann.py:

    "GR"         f = R                  (no modification; the parameterisation
                                         is the background, as in the old pipeline)
    "power_law"  f = R + alpha R^n      free: alpha, n
    "R_log_R"    f = R (1 + alpha ln R) free: alpha
    "R_lambda_R2" f = R - 2 Lambda + alpha R^2   free: Lambda, alpha  (not in fR_friedmann.py)
                 Solved NOT by the ODE but on the smooth (scalaron-free) attractor, in closed form
                 to second order in alpha -- see "R_lambda_R2" below.
    "starobinsky" f = R + alpha R^2     free: alpha   (simple Starobinsky model: NO cosmological constant;
                 the "R_lambda_R2" closed form with Lambda = 0 -- see "starobinsky" below)
    "hu_sawicki" f = R - 2 Lambda (1 - 1/(1 + (R/(Lambda b))^p)),  p = 1   free: Lambda, b
                 = R - 2 Lambda R/(R + Lambda b)   (Hu-Sawicki, n = 1).  ODE started on the model's own
                 high-curvature attractor and integrated with an implicit (Radau) method -- see "hu_sawicki" below.

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
* "R_lambda_R2" (f = R - 2 Lambda + alpha R^2).  The ODE is unusable for R^2 gravity: the heavy
  scalaron (mass^2 ~ 1/6alpha) makes it stiff for small alpha (cost ~ alpha^-1/2), and the
  LCDM initial conditions excite scalaron oscillations whose relative amplitude grows as the
  universe expands (R even turns negative).  Instead the tt equation is solved order by order in
  alpha on the smooth attractor.  With x = Omega_m (1+z)^3 (anchor H0 = 1 units):
        u = x + Lambda/3 + (27/2) alpha x^2 - alpha^2 (486 Lambda x^2 + 729 x^3) + O(alpha^3)
  (zeroth order = LCDM with Omega_Lambda = Lambda/3; verified symbolically: the tt residual of
  the first-order form starts at O(alpha^2), and the second-order term follows from it).
  Truncation error is estimated as max|E2_1st/E2_2nd - 1| over 0<=z<=FR_Z_INIT and parameters are
  rejected above PERTURBATIVE_TOL.  Here FR_Z_INIT is the highest redshift where the expansion is
  trusted (default 2.5, just above the data).  This is the scalaron-free attractor: any scalaron
  oscillation is assumed to have decayed.  Practically the estimate scales as alpha^2 and reaches the
  1e-3 tolerance at alpha ~ 9e-5, where the effect of alpha on E^2 at the highest data redshift
  (BAO, z = 2.36) is only ~0.6%, below the data errors: alpha is UNCONSTRAINED by background data
  (its posterior is the prior cut at ~9e-5), and the model is degenerate with LCDM (Lambda plays
  the role of Omega_m).
* "starobinsky" (f = R + alpha R^2, NO Lambda).  Same scalaron-free attractor as "R_lambda_R2" evaluated at Lambda = 0:
        u = x + (27/2) alpha x^2 - 729 alpha^2 x^3 + O(alpha^3),     x = Omega_m (1+z)^3   (anchor H0 = 1 units)
  (verified symbolically: the tt residual of this form starts at O(alpha^3); the first-order form's starts at O(alpha^2)).
  There is NO dark-energy component of any kind here: with alpha <= 1e-4 the R^2 term is a <~2% correction to
  a matter-only (Einstein-de Sitter) expansion at z = 2.4 and vanishes at z = 0, so the model CANNOT produce late-time
  acceleration (q0 ~ 0.5).  It is a minimal reference model, not a competitive dark-energy model; expect a very poor fit
  (H0 pushed against its lower prior bound).  alpha is the only f(R) parameter; the natural companion is LCDM
  (any other parameterisation contributes only Omega_m -- its w0, wa, ... are unused and stay prior-dominated).
* "hu_sawicki" (f = R - 2 Lambda R/(R + c),  c = Lambda b;  the p = 1 case of
  f = R - 2 Lambda [1 - 1/(1 + (R/(Lambda b))^p)]).  Derivatives (verified symbolically):
        f_R   = 1 - 2 Lambda c/(R + c)^2,   f_RR = 4 Lambda c/(R + c)^3,   f_RRR = -12 Lambda c/(R + c)^4
  Limits: R >> c gives f -> R - 2 Lambda (LCDM with Omega_Lambda = Lambda/3 in anchor units), so there IS a
  GR-like matter era; b -> 0 is exactly LCDM.  f_RR > 0 for R > 0 needs Lambda b > 0.  A de Sitter attractor
  (R f_R = 2 f) exists only for b < 2, at R_dS = Lambda (2 - b + sqrt(4 - 2b)) (-> 4 Lambda as b -> 0), so the
  prior is 0 < b < 2.  Mapping to the usual Hu-Sawicki (n = 1) parameters (m^2, c1, c2):
  2 Lambda = m^2 c1/c2,  Lambda b = m^2/c2.  Viable (solar-system / small |f_R0|) models have b << 1.
  STIFFNESS (the essential problem).  f_RR ~ R^-3 is tiny at early times, so the scalaron mode is oscillatory and
  extremely stiff (omega/H ~ R^(3/2)/sqrt(Lambda c) ~ 1e2-1e3 at z ~ 10).  The default LSODA stays in its explicit
  mode and needs ~1e3-1e6 right-hand-side evaluations (growing like (1 + z_init)^4.5 and as b -> 0), so the generic
  5000-evaluation cap rejects most of the parameter space (measured: 11 of 15 tested (b, z_init) combinations failed).
  BDF is even slower; Radau (L-stable, follows the attractor) needs ~2-5e3 evaluations (~50 ms), nearly independent of
  z_init, and agrees with the oscillation-resolving LSODA solution to <~4e-8 at rtol = 1e-6.  Radau is therefore
  registered via "ode_options".
  TOLERANCE.  The solve is ~96% of a likelihood evaluation (the Pantheon+ term costs ~2 ms), and its cost is set by rtol.
  Measured on Pantheon+SH0ES + CC + BAO + clustering (30 points over the posterior region and the corners; 250 random
  prior points for robustness; reference rtol = 1e-9):
        rtol   ms/evaluation   max|E^2 error|   max|d lnL|
        1e-6        38             8e-8            8e-6
        1e-5        25             8e-7            1e-4
        1e-4        17             2e-5            1e-3
  (hu_sawicki_de: 74 / 38 / 24 ms per solve.)  No solve changed between success and failure at any of the three
  tolerances.  rtol = 1e-4 is registered: lnL errors of 1e-3 are far below the O(0.1-1) differences an MCMC step is
  sensitive to.  Set "rtol" in "ode_options" to 1e-6 for ~100x smaller errors at ~2.4-3x the cost.  Note that the IC
  convergence number printed by main.py then shows the tolerance floor (~1e-5), not the IC dependence; it is
  an informational diagnostic, and check_fR.py tests the IC dependence at a tight tolerance.
  INITIAL CONDITIONS (a refinement).  The ODE is started on the model's own high-curvature attractor
  u = x + Lambda/3 + delta1, x = Omega_m e^{-3N}, with delta1 (first order in g = 2 Lambda c/(R + c)) derived in
  _ics_hu_sawicki, instead of the parameterisation's E^2.  The latter has Omega_Lambda = 1 - Omega_m, which equals
  Lambda/3 only at Lambda = 3(1 - Omega_m) (=2.1 for the anchor 0.3), and the mismatch is divided by the tiny f_RR.
  Measured effect (Radau, b = 0.1, z_init = 10): the parameterisation's ICs change E^2 by 2e-5..1e-4 for
  Lambda = 1.5..3.5 (decaying with z_init, and one failed solve at Lambda = 1.5, z_init = 40); the attractor ICs make
  E^2(z) independent of z_init to <~1e-7 everywhere tested (z_init = 10..80).  Both are far below the data errors,
  so this is about robustness and removing the dependence on the parameterisation's dark-energy parameters.
  Like R_lambda_R2, only Omega_m is taken from the parameterisation; its other parameters (w0, ...) are unused.
* "hu_sawicki_de" = "hu_sawicki" + the parameterisation's dark energy (wCDM, CPL, wowaCDM, JBP, HDE, wCDM_fixed_Omega_m)
  on top of the Lambda inside f(R).  kappa rho_DE = 3 [E^2_param(z) - Omega_m (1+z)^3] (anchor units) is added to the
  tt source; ICs = attractor ICs of the base model + the fluid (u += de, u' += d de/dN; the O(1e-9) cross term between
  delta1 and the fluid is neglected, checked by the z_init convergence test).  The total dark sector is Lambda/3 + fluid
  + the f(R) correction, so (Lambda, w0, ...) are partly degenerate; LCDM is rejected (a constant fluid duplicates Lambda).
  The fluid's amplitude is fixed by the parameterisation, (1 - Omega_m_anchor) = 0.7 of the matter-normalised density, so Lambda is
  an extra constant density: Omega_m,phys = 0.3/(1 + Lambda/3) needs Lambda <~ 0.3 for a realistic matter fraction, and the f(R)
  deviation (~ Lambda^2 b) is then small -> expect a near-wCDM fit.  Walker start: Lambda = 0.3, b = 0.5.
  Not available for R_lambda_R2: its closed form assumes a pure-matter source x ~ e^{-3N}.
* Initial-condition dependence.  The parameterisation's (u, u') at FR_Z_INIT are only
  a valid starting point if the f(R) model has a GR-like matter era at early times.
    - "power_law" (0<n<1): alpha R^n / R -> 0 as R -> infinity, so E^2(z) converges
      as FR_Z_INIT grows (changes by ~10x per doubling; FR_Z_INIT = 40 gives ~5e-4).
    - "R_log_R": F = 1 + alpha (ln R + 1) does NOT tend to 1, there is no GR matter-era
      attractor, and E^2(z) does NOT converge with FR_Z_INIT (10-50% changes found).
      Its results depend on the arbitrary starting redshift and must not be trusted.
  main.py prints this convergence test (ic_sensitivity) for every f(R) run.
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
from scipy.integrate import solve_ivp


# ════════════════════════════════════════════════════════════════════════════
# PRIOR BOUNDS  (uniform, hard-wall; edit here)
# ════════════════════════════════════════════════════════════════════════════
FR_PRIOR_BOUNDS = {
    "alpha_power": (-10.0, 10.0),
    "n_power"    : (  0.1,  2.0),
    "alpha_log"  : (  0.0,  3.0),
    "Lambda"     : (  0.0, 20.0),     # R_lambda_R2: cosmological constant in anchor-H0^2 units (LCDM-like value ~ 3(1-Omega_m) = 2.1)
    "alpha_R2"   : (  0.0,  1e-4),    # R_lambda_R2: R^2 coefficient (anchor-H0^-2 units); PERTURBATIVE_TOL cuts in at ~9e-5
    "b_HS"       : (  0.0,  2.0),     # hu_sawicki: b = c/Lambda (dimensionless); b > 0 for f_RR > 0, b < 2 for a de Sitter attractor
                                      #   (Lambda for hu_sawicki reuses the "Lambda" bound above)
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


def _derivs_R_lambda_R2(theta):
    Lam, alpha = theta
    return (lambda R: R - 2.0 * Lam + alpha * R**2,
            lambda R: 1.0 + 2.0 * alpha * R,
            lambda R: 2.0 * alpha,
            lambda R: 0.0)

def _prior_R_lambda_R2(theta):
    Lam, alpha = theta
    # f_RR = 2 alpha > 0 (stability)
    return 0.0 if (_in(Lam, "Lambda") and _in(alpha, "alpha_R2") and alpha > 0.0) else -np.inf


def _derivs_hu_sawicki(theta):
    Lam, b = theta
    c = Lam * b
    return (lambda R: R - 2.0 * Lam * R / (R + c),
            lambda R: 1.0 - 2.0 * Lam * c / (R + c) ** 2,
            lambda R: 4.0 * Lam * c / (R + c) ** 3,
            lambda R: -12.0 * Lam * c / (R + c) ** 4)

def _prior_hu_sawicki(theta):
    Lam, b = theta
    # f_RR = 4 Lambda^2 b/(R + Lambda b)^3 > 0 (stability) <=> Lambda > 0 and b > 0;  b < 2: a de Sitter attractor exists
    return 0.0 if (_in(Lam, "Lambda") and _in(b, "b_HS") and Lam > 0.0 and 0.0 < b < 2.0) else -np.inf

def _ics_hu_sawicki(Omega_m, theta, z_init):
    """
    (u, u') = (H^2/H0^2, du/dN) at z_init on the model's own high-curvature attractor, anchor units.

    Write f = R - 2 Lambda + g(R) with g = 2 Lambda c/(R + c)  (c = Lambda b), small for R >> c.  With
    x = Omega_m e^{-3N} (x' = -3x):
        zeroth order:  u0 = x + Lambda/3,  R0 = 3 u0' + 12 u0 = 3x + 4 Lambda,  R0' = -9x   (LCDM; tt equation exact for g = 0)
        first order :  put u = u0 + delta in  kappa rho = 3 F u + (f - R F)/2 + 3 u f_RR R'  = 3x.  Since g is first order,
                       delta enters only through the term 3 delta, so   delta1 = -(tt residual of u0)/3  with
                       tt residual = 3 (1 + g_R) u0 + (R0 - 2 Lambda + g - R0 (1 + g_R))/2 + 3 u0 g_RR R0' - 3x,  g, g_R, g_RR at R0.
    Simplified with sympy (check_fR.py re-derives it and confirms the residual of u0 + delta1 is O(g^2)):
        delta1  = Lambda^2 b (-Lambda^2 b^2 - 10 Lambda^2 b - 24 Lambda^2 - 3 Lambda b x + 18 Lambda x + 108 x^2) / (3 D^3)
        delta1' = 6 Lambda^2 b x (-Lambda^2 b^2 - 16 Lambda^2 b - 48 Lambda^2 - 39 Lambda b x - 126 Lambda x + 54 x^2) / D^4
        D = Lambda b + 4 Lambda + 3 x
    delta1 ~ 4 Lambda^2 b/(3x) at early times, i.e. ~4e-6 relative at z = 10 for Lambda = 2.1, b = 0.1.
    """
    Lam, b = theta
    x = Omega_m * (1.0 + z_init) ** 3
    D = Lam * b + 4.0 * Lam + 3.0 * x
    d1 = Lam**2 * b * (-Lam**2 * b**2 - 10.0 * Lam**2 * b - 24.0 * Lam**2 - 3.0 * Lam * b * x
                       + 18.0 * Lam * x + 108.0 * x**2) / (3.0 * D**3)
    d1_N = 6.0 * Lam**2 * b * x * (-Lam**2 * b**2 - 16.0 * Lam**2 * b - 48.0 * Lam**2 - 39.0 * Lam * b * x
                                   - 126.0 * Lam * x + 54.0 * x**2) / D**4
    return x + Lam / 3.0 + d1, -3.0 * x + d1_N


PERTURBATIVE_TOL = 1e-3    # max tolerated estimated truncation error of E^2 for perturbative backgrounds


def _background_R_lambda_R2(Omega_m, theta, z_max):
    """
    Second-order attractor solution of f = R - 2 Lambda + alpha R^2 (module docstring).
    Returns (Background or None, reason or None).  z_max = highest redshift E2 may be asked for.
    """
    Lam, alpha = theta

    def u(z, order):
        x = Omega_m * (1.0 + np.asarray(z, dtype=float)) ** 3
        out = x + Lam / 3.0
        if order >= 1:
            out = out + 13.5 * alpha * x**2
        if order >= 2:
            out = out - alpha**2 * (486.0 * Lam * x**2 + 729.0 * x**3)
        return out

    zg = np.linspace(0.0, z_max, 100)
    u2, u1 = u(zg, 2), u(zg, 1)
    if not (np.all(np.isfinite(u2)) and np.all(u2 > 0.0) and np.all(u1 > 0.0)):
        return None, "expansion in alpha gives u <= 0 (alpha too large for the perturbative solution)"
    err = float(np.max(np.abs((u1 / u1[0]) / (u2 / u2[0]) - 1.0)))
    if err > PERTURBATIVE_TOL:
        return None, (f"perturbative expansion in alpha not converged: estimated truncation error "
                      f"{err:.1e} > {PERTURBATIVE_TOL:.0e} up to z = {z_max:g} (reduce alpha or FR_Z_INIT)")
    u0 = float(u2[0])
    if not (U0_RANGE[0] < u0 < U0_RANGE[1]):
        return None, (f"u(0) = {u0:.3g} outside {U0_RANGE} (physical Omega_m = Omega_m/u(0) = {Omega_m / u0:.2g})")

    def E2(z):
        z = np.asarray(z, dtype=float)
        if np.any(z > z_max):
            raise ValueError(f"E2(z) requested at z={z.max():.3g} > FR_Z_INIT={z_max}; raise FR_Z_INIT in main.py.")
        out = u(z, 2) / u0
        return out if z.ndim else float(out)
    return Background(E2, u0, Omega_m, None, err), None


# ── "starobinsky": f = R + alpha R^2, no cosmological constant (= "R_lambda_R2" with Lambda = 0) ─────────────
def _derivs_starobinsky(theta):
    alpha, = theta
    return _derivs_R_lambda_R2((0.0, alpha))

def _prior_starobinsky(theta):
    alpha, = theta
    # f_RR = 2 alpha > 0 (stability); upper edge = alpha_R2 bound (the perturbative-error guard cuts in near ~9e-5)
    return 0.0 if (_in(alpha, "alpha_R2") and alpha > 0.0) else -np.inf

def _background_starobinsky(Omega_m, theta, z_max):
    alpha, = theta
    return _background_R_lambda_R2(Omega_m, (0.0, alpha), z_max)


# ════════════════════════════════════════════════════════════════════════════
# f(R) REGISTRY
# ════════════════════════════════════════════════════════════════════════════
#   param_names: names of the f(R) parameters, in order (usable in FR_FIXED_PARAMS)
#   labels     : LaTeX names of the extra MCMC parameters (appended after the
#                parameterisation's labels)
#   p0_center / p0_scatter : walker initialisation for those parameters
#   prior_func : theta_fR -> 0.0 or -inf
#   derivs     : theta_fR -> (f, f_R, f_RR, f_RRR);  None for GR
#   analytic_background (optional): (Omega_m, theta_fR, z_max) -> (Background|None, reason); replaces the ODE
#   z_init_default (optional): default FR_Z_INIT for this model (40 if absent)
#   ic_func (optional): (Omega_m, theta_fR, z_init) -> (u_i, u'_i); the ODE is started on the model's own attractor
#                       INSTEAD of the parameterisation's E^2 (removes an IC mismatch of ~1e-4; see "hu_sawicki" in the docstring)
#   ode_options (optional): extra solve_background keywords (method, rtol, max_rhs_evals) for models whose ODE is very stiff
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
    "R_lambda_R2": {
        "description": "f(R) = R - 2 Lambda + alpha R^2  (Starobinsky R^2 with a cosmological constant)",
        "param_names": ["Lambda", "alpha"],
        "labels"     : [r"$\Lambda_{f(R)}$", r"$\alpha_{f(R)}$"],
        "p0_center"  : np.array([2.1, 5e-5]),     # Lambda ~ 3(1-Omega_m) = 2.1 -> LCDM-like; alpha small
        "p0_scatter" : np.array([0.05, 1e-5]),
        "prior_func" : _prior_R_lambda_R2,
        "derivs"     : _derivs_R_lambda_R2,
        "analytic_background": _background_R_lambda_R2,   # used INSTEAD of the ODE
        "z_init_default": 2.5,                            # = highest redshift of validity
    },
    "starobinsky": {
        "description": "f(R) = R + alpha R^2  (Starobinsky R^2, NO cosmological constant)",
        "param_names": ["alpha"],
        "labels"     : [r"$\alpha_{f(R)}$"],
        "p0_center"  : np.array([5e-5]),          # inside (0, 1e-4], ~5 sigma from both walls
        "p0_scatter" : np.array([1e-5]),
        "prior_func" : _prior_starobinsky,
        "derivs"     : _derivs_starobinsky,
        "analytic_background": _background_starobinsky,   # used INSTEAD of the ODE
        "z_init_default": 2.5,                            # = highest redshift of validity
    },
    "hu_sawicki": {
        "description": "f(R) = R - 2 Lambda (1 - 1/(1 + (R/(Lambda b))^p)), p = 1  (Hu-Sawicki n = 1)",
        "param_names": ["Lambda", "b"],
        "labels"     : [r"$\Lambda_{f(R)}$", r"$b_{f(R)}$"],
        "p0_center"  : np.array([2.1, 0.1]),      # Lambda ~ 3(1-Omega_m) = 2.1 -> LCDM-like (u(0) ~ 0.98); b small = near-LCDM
        "p0_scatter" : np.array([0.05, 0.02]),
        "prior_func" : _prior_hu_sawicki,
        "derivs"     : _derivs_hu_sawicki,
        "ic_func"    : _ics_hu_sawicki,
        "ode_options": {"method": "Radau", "rtol": 1e-4, "max_rhs_evals": 20000},   # rtol: see "hu_sawicki" in the docstring
        "z_init_default": 10.0,                   # E^2(z) is independent of z_init to ~1e-7 for 10..80; larger only costs time
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


# ── "<model>_de": the parameterisation's dark energy ON TOP of the cosmological constant inside f(R) ──────────
# Same f(R) function, parameters, prior and ICs as the base model, but the parameterisation's dark-energy density
# rho_DE(N) = E^2_param(z) - Omega_m (1+z)^3 (a separately conserved component, p_DE from its own continuity
# equation) is added to the source of the f(R) field equations next to the constant -2 Lambda that lives in f.
# At the level of the tt equation (the only one integrated) it makes no difference whether this term is written as a
# T_mu_nu component or as Lambda(N) inside f; only the conserved version satisfies the Bianchi identity (kappa p check).
# Hence the parameterisation's parameters (w0, wa, c, ...) now enter E^2(z).  Omega_m stays the anchor.
def _de_variant(base):
    cfg = dict(FR_REGISTRY[base])
    cfg.update(description=FR_REGISTRY[base]["description"] + "  + parameterisation dark energy on top of Lambda",
               base_model=base, de_on_top=True,
               # the fluid already supplies (1 - Omega_m_anchor) = 0.7 of dark energy, so Lambda is an EXTRA constant density:
               # start small (Lambda = 2.1 would give Omega_m,phys ~ 0.18).  Small Lambda also suppresses the f(R) correction.
               p0_center=np.array([0.3, 0.5]), p0_scatter=np.array([0.05, 0.1]))
    return cfg
FR_REGISTRY["hu_sawicki_de"] = _de_variant("hu_sawicki")


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
    sol: scipy dense solution y(N) = (u, u') in anchor units (None for GR / perturbative models).
    truncation_error: perturbative models only: estimated relative error of E^2 (None for ODE models)."""
    def __init__(self, E2, u0=1.0, omega_m=None, sol=None, truncation_error=None):
        self.E2 = E2
        self.u0 = u0
        self.omega_m = omega_m
        self.sol = sol
        self.truncation_error = truncation_error


MAX_RHS_EVALS = 5000       # typical solve: ~300-600; stiff corners of parameter space blow past this
U0_RANGE = (1e-2, 1e2)     # u(0) outside this <=> physical Omega_m = Omega_m/u(0) outside ~(0.003, 30): runaway


def solve_background(E2_param, Omega_m, derivs, z_init, rtol=1e-8, return_reason=False,
                     ics=None, method="LSODA", max_rhs_evals=None, extra_kappa_rho=None):
    """
    Integrate the f(R) background from z_init to z=0 (see module docstring).
    Returns a Background, or None if the integration fails or leaves the
    physical domain (u <= 0, R <= 0, blow-up, non-finite output, too many function
    evaluations (stiff), or u(0) outside U0_RANGE (runaway solution)).
    With return_reason=True returns (background_or_None, reason_string_or_None).
    ics = (u_i, u'_i) overrides the initial conditions normally taken from E2_param (which may then be None);
    method / max_rhs_evals select the scipy integrator and the stiffness cap (default LSODA, MAX_RHS_EVALS).
    extra_kappa_rho(N) -> kappa*rho_extra (anchor units) is ADDED to the matter source 3 Omega_m e^{-3N} in the tt equation
    (a separately conserved dark-energy component; see "hu_sawicki_de").
    """
    def fail(msg):
        return (None, msg) if return_reason else None

    N_i = -np.log1p(z_init)
    h = 1e-4
    if ics is not None:
        u_i, v_i = float(ics[0]), float(ics[1])
    else:
        u_i = float(E2_param(z_init))
        v_i = -(1.0 + z_init) * (float(E2_param(z_init + h)) - float(E2_param(z_init - h))) / (2.0 * h)
    R_i = 3.0 * v_i + 12.0 * u_i
    if not (np.isfinite(u_i) and np.isfinite(v_i) and u_i > 0.0 and R_i > 0.0):
        return fail("invalid initial conditions from the parameterisation (u_i or R_i <= 0)")

    R_FLOOR = 1e-12          # keeps rhs finite inside a step; events catch the violation

    class _TooStiff(Exception):
        pass
    n_eval = [0]
    eval_cap = MAX_RHS_EVALS if max_rhs_evals is None else max_rhs_evals

    def rhs(N, y):
        n_eval[0] += 1
        if n_eval[0] > eval_cap:
            raise _TooStiff
        u, v = y
        R = max(3.0 * v + 12.0 * u, R_FLOOR)
        u = max(u, R_FLOOR)
        source = 3.0 * Omega_m * np.exp(-3.0 * N)
        if extra_kappa_rho is not None:
            source += float(extra_kappa_rho(N))
        R_N = R_N_from_matter(derivs, R, u, source)
        return [v, R_N / 3.0 - 4.0 * v]          # R' = 3u'' + 12u'

    def leaves_domain(N, y):                      # terminal: u<=0 or R<=0
        return min(y[0], 3.0 * y[1] + 12.0 * y[0]) - 1e-8
    leaves_domain.terminal, leaves_domain.direction = True, -1

    def blows_up(N, y):                           # terminal: |u'/u| huge
        return 1e3 - abs(y[1]) / max(abs(y[0]), R_FLOOR)
    blows_up.terminal, blows_up.direction = True, -1

    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(rhs, (N_i, 0.0), [u_i, v_i], method=method, rtol=rtol, atol=1e-10,
                            dense_output=True, events=(leaves_domain, blows_up))
    except _TooStiff:
        return fail(f"stiff: more than {eval_cap} right-hand-side evaluations")
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
