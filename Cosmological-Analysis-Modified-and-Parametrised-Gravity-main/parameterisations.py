# parameterisations.py
"""
Background-expansion parameterisations E^2(z) = H^2(z)/H0^2 (LCDM, wCDM, CPL, JBP, ...).

With FR_MODEL = "GR" in main.py a parameterisation IS the background, exactly as
before.  With a modified f(R) model chosen in fR.py, the parameterisation only
supplies the initial conditions (u = E^2 and du/dlna at FR_Z_INIT) and the
matter density Omega_m for the f(R) ODE -- see fR.py.

(This file was models.py; MODEL_REGISTRY is now PARAMETERISATION_REGISTRY.)
"""
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# PRIOR BOUNDS REGISTRY
# ════════════════════════════════════════════════════════════════════════════
# All prior limits live here. Edit these — do not touch the prior functions.
#
#   Key         : parameter name (must match usage in prior functions below)
#   Value       : (lower_bound, upper_bound)  — uniform, hard-wall priors
#
PRIOR_BOUNDS = {
    "H0"      : (50.0,  100.0),
    "Omega_m" : (0.05,  0.70),
    "Omega_0" : (0.05,  0.95),   # Used by parameterised H(z) models
    "w0"      : (-3.0,   0.0),
    "wa_cpl"  : (-5.0,   5.0),
    "wa_wowa" : (-5.0,   5.0),
    "alpha"   : (-2.0,   4.0),   # DE exponent, param_Hubble_fixed_delta
    "delta"   : (-2.0,   2.0),   # Matter exponent deviation, param_Hubble_fixed_alpha
    "c_hde"   : ( 0.1,   3.0),   # HDE IR cutoff parameter
    "xi"      : (-1.0,   1.0),   # IDE coupling strength
}

def _in_bounds(val, key):
    lo, hi = PRIOR_BOUNDS[key]
    return lo < val < hi

def E2_wcdm_fixed_Omega_m(z, theta_cosmo):
    w0, = theta_cosmo
    term_matter = 0.3 * (1.0 + z)**3
    term_de     = (1.0 - 0.3) * (1.0 + z)**(3.0 * (1.0 + w0))
    return term_matter + term_de

def prior_wcdm_fixed_Omega_m(H0, theta_cosmo):
    w0, = theta_cosmo
    if _in_bounds(H0, "H0") and _in_bounds(w0, "w0"):
        return 0.0
    return -np.inf



# ════════════════════════════════════════════════════════════════════════════
# MODEL 1: Flat ΛCDM
#   E²(z) = Ω_m (1+z)³ + (1 − Ω_m)
# ════════════════════════════════════════════════════════════════════════════
def E2_lcdm(z, theta_cosmo):
    Omega_m, = theta_cosmo
    return Omega_m * (1.0 + z)**3 + (1.0 - Omega_m)

def prior_lcdm(H0, theta_cosmo):
    Omega_m, = theta_cosmo
    if _in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m"):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# MODEL 2: Flat wCDM  (constant dark energy equation of state)
#   E²(z) = Ω_m (1+z)³ + (1 − Ω_m)(1+z)^{3(1+w₀)}
# ════════════════════════════════════════════════════════════════════════════
def E2_wcdm(z, theta_cosmo):
    Omega_m, w0 = theta_cosmo
    term_matter = Omega_m * (1.0 + z)**3
    term_de     = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w0))
    return term_matter + term_de

def prior_wcdm(H0, theta_cosmo):
    Omega_m, w0 = theta_cosmo
    if _in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m") and _in_bounds(w0, "w0"):
        return 0.0
    return -np.inf





# ════════════════════════════════════════════════════════════════════════════
# MODEL 3: Flat CPL  (Chevallier-Polarski-Linder)
#   w(z) = w₀ + wₐ · z/(1+z)
#   E²(z) = Ω_m (1+z)³ + (1−Ω_m)(1+z)^{3(1+w₀+wₐ)} exp(−3wₐ z/(1+z))
# ════════════════════════════════════════════════════════════════════════════
def E2_cpl(z, theta_cosmo):
    Omega_m, w0, wa = theta_cosmo
    term_matter = Omega_m * (1.0 + z)**3
    term_de     = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w0 + wa)) \
                  * np.exp(-3.0 * wa * z / (1.0 + z))
    return np.maximum(term_matter + term_de, 1e-9)

def prior_cpl(H0, theta_cosmo):
    Omega_m, w0, wa = theta_cosmo
    if (_in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m")
            and _in_bounds(w0, "w0") and _in_bounds(wa, "wa_cpl")):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# MODEL 4: Flat wowaCDM  (alternative two-parameter w(z) form)
#   Same functional form as CPL — exists as a separate entry so that
#   p0_center / prior widths can differ if needed for this parameterisation.
# ════════════════════════════════════════════════════════════════════════════
def E2_wowacdm(z, theta_cosmo):
    # Identical physics to CPL; kept separate for registry flexibility.
    return E2_cpl(z, theta_cosmo)

def prior_wowacdm(H0, theta_cosmo):
    Omega_m, w0, wa = theta_cosmo
    if (_in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m")
            and _in_bounds(w0, "w0") and _in_bounds(wa, "wa_wowa")):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# MODEL 5: Flat JBP  (Jassal-Bagla-Padmanabhan)
#   w(z) = w₀ + wₐ · z/(1+z)²
#   E²(z) = Ω_m (1+z)³ + (1−Ω_m)(1+z)^{3(1+w₀)} exp(3wₐ/2 · z²/(1+z)²)
# ════════════════════════════════════════════════════════════════════════════
def E2_jbp(z, theta_cosmo):
    Omega_m, w0, wa = theta_cosmo
    term_matter = Omega_m * (1.0 + z)**3
    term_de     = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w0)) \
                  * np.exp(1.5 * wa * z**2 / (1.0 + z)**2)
    return np.maximum(term_matter + term_de, 1e-9)

def prior_jbp(H0, theta_cosmo):
    Omega_m, w0, wa = theta_cosmo
    if (_in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m")
            and _in_bounds(w0, "w0") and _in_bounds(wa, "wa_cpl")):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# MODEL 6: HDE  (Holographic Dark Energy, constant-w_eff approximation)
#   w_eff = −1/3 − 2/(3c)   [from the HDE Friedmann constraint at leading order]
#   E²(z) = Ω_m (1+z)³ + (1−Ω_m)(1+z)^{3(1+w_eff)}
# ════════════════════════════════════════════════════════════════════════════
def E2_hde(z, theta_cosmo):
    Omega_m, c = theta_cosmo
    w_eff       = -1.0/3.0 - 2.0 / (3.0 * c)   # effective EoS from HDE IR cutoff
    term_matter = Omega_m * (1.0 + z)**3
    term_de     = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w_eff))
    return np.maximum(term_matter + term_de, 1e-9)

def prior_hde(H0, theta_cosmo):
    Omega_m, c = theta_cosmo
    if _in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m") and _in_bounds(c, "c_hde"):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# MODEL 7: IDE  (Interacting Dark Energy, phenomenological coupling ξ)
#   Continuity with coupling Q = ξ H ρ_de splits DE into two modes:
#     ρ_de(z) ∝  3/(3+ξ) (1+z)³  +  ξ/(3+ξ) (1+z)^{−ξ}
#   E²(z) = Ω_m (1+z)³ + (1−Ω_m)[ 3/(3+ξ)(1+z)³ + ξ/(3+ξ)(1+z)^{−ξ} ]
# ════════════════════════════════════════════════════════════════════════════
def E2_ide(z, theta_cosmo):
    Omega_m, xi = theta_cosmo
    term_matter  = Omega_m * (1.0 + z)**3
    # Two-mode DE density from energy transfer with coupling ξ
    f_drag       = 3.0 / (3.0 + xi)
    f_decay      = xi  / (3.0 + xi)
    term_de      = (1.0 - Omega_m) * (f_drag  * (1.0 + z)**3
                                     + f_decay * (1.0 + z)**(-xi))
    return np.maximum(term_matter + term_de, 1e-9)

def prior_ide(H0, theta_cosmo):
    Omega_m, xi = theta_cosmo
    if _in_bounds(H0, "H0") and _in_bounds(Omega_m, "Omega_m") and _in_bounds(xi, "xi"):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# PARAMETRISED H(z) MODELS  (custom two-exponent framework)
#   H²(z) = H₀² [ Ω₀(1+z)^{3+δ}  +  (1−Ω₀)(1+z)^α ]
#
#   Three cases depending on which exponents are free vs. fixed in main.py:
#     fixed_delta  : δ fixed externally, α is a free MCMC parameter
#     fixed_alpha  : α fixed externally, δ is a free MCMC parameter
#     both_fixed   : both δ and α fixed externally; only Ω₀ is sampled
# ════════════════════════════════════════════════════════════════════════════
def _E2_two_exponent(z, Omega_0, delta, alpha):
    """Core two-exponent expansion: returns E²(z) given both exponents."""
    Omega_matter_like = Omega_0       * (1.0 + z)**(3.0 + delta)
    Omega_de_like     = (1.0 - Omega_0) * (1.0 + z)**alpha
    return Omega_matter_like + Omega_de_like

def omega_components(z, Omega_0, delta, alpha):
    """Return (Ω_matter-like, Ω_DE-like) separately — used for component plots."""
    return (Omega_0       * (1.0 + z)**(3.0 + delta),
            (1.0 - Omega_0) * (1.0 + z)**alpha)

# Backward-compatible alias so main.py's plot_and_save_omega_components still works
calculate_omega_components = omega_components


# ── Case A: δ fixed in main.py, α free ──────────────────────────────────────
def E2_fixed_delta(z, theta, fixed_params):
    Omega_0, alpha = theta
    delta,         = fixed_params          # unpack single-element tuple
    return _E2_two_exponent(z, Omega_0, delta, alpha)

def prior_fixed_delta(H0, theta):
    Omega_0, alpha = theta
    if (_in_bounds(H0, "H0") and _in_bounds(Omega_0, "Omega_0")
            and _in_bounds(alpha, "alpha")):
        return 0.0
    return -np.inf


# ── Case B: α fixed in main.py, δ free ──────────────────────────────────────
def E2_fixed_alpha(z, theta, fixed_params):
    Omega_0, delta = theta
    alpha,         = fixed_params
    return _E2_two_exponent(z, Omega_0, delta, alpha)

def prior_fixed_alpha(H0, theta):
    Omega_0, delta = theta
    if (_in_bounds(H0, "H0") and _in_bounds(Omega_0, "Omega_0")
            and _in_bounds(delta, "delta")):
        return 0.0
    return -np.inf


# ── Case C: both δ and α fixed in main.py, only Ω₀ free ────────────────────
def E2_both_fixed(z, theta, fixed_params):
    Omega_0,       = theta
    delta, alpha   = fixed_params
    return _E2_two_exponent(z, Omega_0, delta, alpha)

def prior_both_fixed(H0, theta):
    Omega_0, = theta
    if _in_bounds(H0, "H0") and _in_bounds(Omega_0, "Omega_0"):
        return 0.0
    return -np.inf


# ════════════════════════════════════════════════════════════════════════════
# PARAMETERISATION REGISTRY
# ════════════════════════════════════════════════════════════════════════════
# Each entry specifies everything needed to run a parameterisation.
#
# Fields:
#   description  : one-line physics summary
#   labels       : LaTeX parameter names in MCMC order — [H0, *theta_cosmo]
#   p0_center    : initial walker positions (same order as labels)
#   p0_scatter   : per-parameter Gaussian scatter for walker initialisation
#   E2_func      : E²(z, theta_cosmo) or E²(z, theta_cosmo, fixed_params)
#   prior_func   : log-prior returning 0.0 or -inf
#   potential_func: optional V(z, theta_cosmo) reconstruction; None if unused
#   omega_m_func : theta_cosmo -> present-day matter density Omega_m (H0 units).
#                  Needed to combine the entry with a modified f(R) model.
#                  None = no standard conserved matter component -> GR only.
#   omega_m_index: position of Omega_m inside theta_cosmo (None if it is not sampled).
#                  In f(R) runs Omega_m is held fixed at FR_OMEGA_M_ANCHOR (main.py),
#                  because it is degenerate with alpha there (see fR.py).
#
PARAMETERISATION_REGISTRY = {

    "wCDM_fixed_Omega_m": {
        "description"   : "Flat wCDM fixed Omega_m=0.3",
        "labels"        : [r"$H_0$",  r"$w_0$"],
        "p0_center"     : np.array([67.0,  -1.00]),
        "p0_scatter"    : np.array([ 1.5,   0.05]),
        "E2_func"       : E2_wcdm_fixed_Omega_m,
        "prior_func"    : prior_wcdm_fixed_Omega_m,
        "potential_func": lambda z, theta: (
            0.5 * (1.0 - theta[0]) * (1.0 + z)**(3.0 * (1.0 + theta[0]))
        ),
        "omega_m_func"  : lambda theta: 0.3,   # Omega_m is hard-wired to 0.3 in E2_wcdm_fixed_Omega_m
        "omega_m_index" : None,
    },


    # ── Standard cosmological models ────────────────────────────────────────

    "LCDM": {
        "description"   : "Flat ΛCDM: matter + cosmological constant",
        "labels"        : [r"$H_0$",  r"$\Omega_m$"],
        "p0_center"     : np.array([70.0,  0.30]),
        "p0_scatter"    : np.array([ 2.0,  0.02]),
        "E2_func"       : E2_lcdm,
        "prior_func"    : prior_lcdm,
        "potential_func": None,
        "omega_m_func"  : lambda theta: theta[0],
        "omega_m_index" : 0,
    },

    "wCDM": {
        "description"   : "Flat wCDM: constant dark energy EoS w₀",
        "labels"        : [r"$H_0$",  r"$\Omega_m$",  r"$w_0$"],
        "p0_center"     : np.array([67.0,  0.31,  -1.00]),
        "p0_scatter"    : np.array([ 1.5,  0.01,   0.05]),
        "E2_func"       : E2_wcdm,
        "prior_func"    : prior_wcdm,
        "potential_func": lambda z, theta: (
            # V(φ)/ρ_de,0 reconstruction for constant-w scalar field
            0.5 * (1.0 - theta[1]) * (1.0 + z)**(3.0 * (1.0 + theta[1]))
            #            ^^^w0                             ^^^w0
        ),
        "omega_m_func"  : lambda theta: theta[0],
        "omega_m_index" : 0,
    },


    "CPL": {
        "description"   : "Flat CPL: w(z) = w₀ + wₐ z/(1+z)  [Chevallier-Polarski-Linder]",
        "labels"        : [r"$H_0$",  r"$\Omega_m$",  r"$w_0$",  r"$w_a$"],
        "p0_center"     : np.array([67.5,  0.31,  -1.00,  0.0]),
        "p0_scatter"    : np.array([ 1.5,  0.01,   0.05,  0.1]),
        "E2_func"       : E2_cpl,
        "prior_func"    : prior_cpl,
        "potential_func": lambda z, theta: (
            # V(φ)/ρ_de,0 reconstruction (CPL scalar field, leading-order)
            0.5 * (1.0 - (theta[1] + theta[2] * z / (1.0 + z)))
            #             ^^^w0      ^^^wa
            * (1.0 + z)**(3.0 * (1.0 + theta[1] + theta[2]))
            * np.exp(-3.0 * theta[2] * z / (1.0 + z))
        ),
        "omega_m_func"  : lambda theta: theta[0],
        "omega_m_index" : 0,
    },

    "wowaCDM": {
        "description"   : "Flat wowaCDM: same EoS as CPL, wider prior on wₐ",
        "labels"        : [r"$H_0$",  r"$\Omega_m$",  r"$w_0$",  r"$w_a$"],
        "p0_center"     : np.array([66.0,  0.31,  -0.50,  0.5]),
        "p0_scatter"    : np.array([ 1.0,  0.02,   0.10,  0.1]),
        "E2_func"       : E2_wowacdm,
        "prior_func"    : prior_wowacdm,
        "potential_func": None,
        "omega_m_func"  : lambda theta: theta[0],
        "omega_m_index" : 0,
    },

    "JBP": {
        "description"   : "Flat JBP: w(z) = w₀ + wₐ z/(1+z)²  [Jassal-Bagla-Padmanabhan]",
        "labels"        : [r"$H_0$",  r"$\Omega_m$",  r"$w_0$",  r"$w_a$"],
        "p0_center"     : np.array([67.5,  0.31,  -1.00,  0.0]),
        "p0_scatter"    : np.array([ 1.5,  0.01,   0.05,  0.1]),
        "E2_func"       : E2_jbp,
        "prior_func"    : prior_jbp,
        "potential_func": None,
        "omega_m_func"  : lambda theta: theta[0],
        "omega_m_index" : 0,
    },

    "HDE": {
        "description"   : "Holographic Dark Energy: IR cutoff parameter c",
        "labels"        : [r"$H_0$",  r"$\Omega_m$",  r"$c$"],
        "p0_center"     : np.array([68.0,  0.30,  1.0]),
        "p0_scatter"    : np.array([ 1.5,  0.02,  0.1]),
        "E2_func"       : E2_hde,
        "prior_func"    : prior_hde,
        "potential_func": None,
        "omega_m_func"  : lambda theta: theta[0],
        "omega_m_index" : 0,
    },

    "IDE": {
        "description"   : "Interacting Dark Energy: coupling parameter ξ",
        "labels"        : [r"$H_0$",  r"$\Omega_m$",  r"$\xi$"],
        "p0_center"     : np.array([67.3,  0.31,  0.00]),
        "p0_scatter"    : np.array([ 1.5,  0.02,  0.05]),
        "E2_func"       : E2_ide,
        "prior_func"    : prior_ide,
        "potential_func": None,
        "omega_m_func"  : None,   # interacting: matter is not conserved -> not combinable with f(R)
        "omega_m_index" : None,
    },

    # ── Parametrised H(z) models ─────────────────────────────────────────────
    # H²(z) = H₀²[ Ω₀(1+z)^{3+δ} + (1−Ω₀)(1+z)^α ]
    # Which of {δ, α} are free vs. fixed is set via FIXED_DELTA / FIXED_ALPHA
    # in main.py's SWITCHBOARD block.

    "param_Hubble_fixed_delta": {
        "description"   : "Two-exponent H(z): δ fixed externally, α free",
        "labels"        : [r"$H_0$",  r"$\Omega_0$",  r"$\alpha$"],
        "p0_center"     : np.array([65.6,  0.26,   0.66]),
        "p0_scatter"    : np.array([ 1.0,  0.02,   0.10]),
        "E2_func"       : E2_fixed_delta,
        "prior_func"    : prior_fixed_delta,
        "potential_func": lambda z, theta: (
            # V reconstruction assuming α drives DE-like component
            0.5 * (2.0 - theta[2] / 3.0) * (1.0 + z)**theta[2]
            #              ^^^alpha                    ^^^alpha
        ),
        "omega_m_func"  : None,   # matter-like term scales as (1+z)^(3+delta) -> not combinable with f(R)
        "omega_m_index" : None,
    },

    "param_Hubble_fixed_alpha": {
        "description"   : "Two-exponent H(z): α fixed externally, δ free",
        "labels"        : [r"$H_0$",  r"$\Omega_0$",  r"$\delta$"],
        "p0_center"     : np.array([66.1,  0.43,  -0.32]),
        "p0_scatter"    : np.array([ 1.0,  0.03,   0.05]),
        "E2_func"       : E2_fixed_alpha,
        "prior_func"    : prior_fixed_alpha,
        "potential_func": None,
        "omega_m_func"  : None,   # matter-like term scales as (1+z)^(3+delta) -> not combinable with f(R)
        "omega_m_index" : None,
    },

    "Custom_Universal": {
        "description"   : "Two-exponent H(z): both δ and α fixed externally, only Ω₀ free",
        "labels"        : [r"$H_0$",  r"$\Omega_0$"],
        "p0_center"     : np.array([66.0,  0.30]),
        "p0_scatter"    : np.array([ 1.5,  0.02]),
        "E2_func"       : E2_both_fixed,
        "prior_func"    : prior_both_fixed,
        "potential_func": None,
        "omega_m_func"  : None,   # matter-like term scales as (1+z)^(3+delta) -> not combinable with f(R)
        "omega_m_index" : None,
    },
}