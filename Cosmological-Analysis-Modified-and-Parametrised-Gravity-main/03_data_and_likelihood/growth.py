# growth.py
"""
Linear growth of matter perturbations and the f*sigma_8(z) likelihood.

WHAT THIS MODULE DOES
---------------------
Given a background (fR.Background: E^2(z), and for f(R) the curvature R(z)) it integrates the sub-horizon
(quasi-static) linear growth equation for the matter contrast delta, builds the observable

    f sigma_8 (z) = sigma_8,0 * delta'(z) / delta(0)          (' = d/dN,  N = ln a)

and compares it with RSD growth-rate data.  Nothing here changes the background likelihood.

EQUATIONS (derived in the thesis notes; N = ln a, ' = d/dN, units: anchor H0 for f(R) backgrounds)
---------------------------------------------------------------------------------------------------
    delta'' + (2 + H'/H) delta' - (3/2) Omega_m(N) (G_eff/G) delta = 0

    H'/H        = u'/(2u),                 u = H^2 / H0_anchor^2     (anchor units; GR backgrounds: u = E^2)
    Omega_m(N)  = kappa rho_m / (3 H^2) = Omega_m,phys e^{-3N} / E^2(N),   Omega_m,phys = Omega_m,anchor / u(0)

    GR + (smooth) dark energy :  G_eff / G = 1
    f(R) (quasi-static, Tsujikawa 2007; De Felice & Tsujikawa 2010):
        G_eff / G = (1/F) (1 + 4 q) / (1 + 3 q),      F = f_R(R),     q = (k / a)^2 f_RR(R) / F
        R = 3 u' + 12 u   (anchor H0^2 units, same convention as fR.py)
    q is evaluated at ONE comoving scale k (default 0.1 h/Mpc).  With k in h/Mpc and H0_anchor = H0 / sqrt(u(0)):
        q = ( c k sqrt(u(0)) e^{-N} / 100 )^2 f_RR / F,        c = 299792.458 km/s     (H0 cancels)
    For f_RR = 0 (Lambda CDM) q = 0 and G_eff = G/F = G, so the GR growth equation is recovered exactly.

INITIAL CONDITIONS
    f = delta'/delta obeys the Riccati equation  f' = -f^2 - A f + B,  A = 2 + H'/H,  B = (3/2) Omega_m G_eff/G.
    At z_start: f = f0 - f0'/(2 f0 + A), with f0 the positive root of f0^2 + A f0 - B = 0 (the local power law delta = a^p;
    p = 1 in a matter era with GR, p = 1.186 if G_eff = 4/3 G) and f0' its N-derivative (slow-roll, first order).
    The residual error is a decaying mode ~ a^(-3/2) relative to the growing one; it is measured in check_growth.py.

AMPLITUDE sigma_8,0 IS MARGINALISED ANALYTICALLY
    The model is linear in sigma_8,0:  fs8_model,i = sigma_8,0 * g_i,   g_i = ratio(z_i) * delta'(z_i)/delta(0).
    With a flat prior on sigma_8,0 and data covariance C:
        A  = g^T C^-1 g,     Bd = g^T C^-1 d,      chi2_min = d^T C^-1 d - Bd^2 / A
        ln L = -chi2_min / 2 - (1/2) ln A + const         (the ln A term depends on the model, so it is kept)
        sigma_8,0 | theta ~ Normal(Bd/A, 1/A)
    This is the same device the pipeline already uses for the SNIa offset M_B, and it keeps the sampled vector unchanged.
    CONSEQUENCE: sigma_8,0 is fixed by the fs8 data alone.  There is NO CMB/primordial normalisation here, so the
    derived sigma_8 is a growth-data estimate, not a prediction of the model.

ALCOCK-PACZYNSKI RESCALING (Nesseris et al. 2017, their eq. 2.5-2.6)
    The surveys assumed a flat LCDM fiducial (Omega_m,fid).  The prediction is rescaled by
        ratio(z) = E(z) chi(z) / ( E_fid(z) chi_fid(z) ),   chi(z) = int_0^z dz'/E(z')   (H0 cancels)
    This is first order and typically a ~1-3 % effect.

ASSUMPTIONS / LIMITS (also in readme section 12)
    * Quasi-static, sub-horizon, linear.  Needs k >> aH.  For f(R) the scalaron mass sets the scale of the modification,
      so the result depends on the chosen k (FS8_K).  This is a modelling choice, not a measured quantity.
    * Dark energy parameterisations (w0, wa, ...) are treated as smooth (no clustering).
    * Matter must be conserved: parameterisations with omega_m_func = None (IDE, param_Hubble_*, Custom_Universal) are rejected.
    * Background must be valid down to the redshift where growth starts: z_start = min(Z_GROWTH_INIT, FR_Z_INIT).
      Closed-form models (R_lambda_R2, starobinsky) only exist up to FR_Z_INIT = 2.5, so growth starts there and carries
      a small initial-condition error (see check_growth.py).
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

from collections import namedtuple

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline

C_LIGHT_KMS = 299792.458

Z_GROWTH_INIT = 10.0        # redshift where the growing-mode initial condition is imposed (capped by the background's range)
K_GROWTH_DEFAULT = 0.1      # h/Mpc, single comoving scale used for the scale-dependent f(R) G_eff
N_STEP = 0.01               # RK4 step in ln a (error ~ N_STEP^4; tested against DOP853 in check_growth.py)

# WiggleZ 3x3 covariance, eq. (2.8) of Nesseris, Pantazis & Perivolaropoulos (2017)
WIGGLEZ_COV = 1e-3 * np.array([[6.400, 2.570, 0.000],
                               [2.570, 3.969, 2.540],
                               [0.000, 2.540, 5.184]])


# ════════════════════════════════════════════════════════════════════════════
# DATA CONTAINER
# ════════════════════════════════════════════════════════════════════════════
class FS8Data:
    """z, fs8, covariance and its inverse, plus the fiducial-cosmology quantities needed for the AP rescaling."""
    def __init__(self, z, fs8, cov, om_fid, names=None):
        self.z = np.asarray(z, dtype=float)
        self.fs8 = np.asarray(fs8, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.om_fid = np.asarray(om_fid, dtype=float)
        self.names = list(names) if names is not None else [""] * len(self.z)
        self.cinv = np.linalg.inv(self.cov)
        self.chi2_0 = float(self.fs8 @ self.cinv @ self.fs8)          # d^T C^-1 d
        # fiducial flat LCDM: E_fid(z) and chi_fid(z) = int_0^z dz'/E_fid  (units of c/H0)
        self.e_fid = np.sqrt(self.om_fid * (1.0 + self.z) ** 3 + 1.0 - self.om_fid)
        self.chi_fid = np.array([quad(lambda x, om=om: 1.0 / np.sqrt(om * (1.0 + x) ** 3 + 1.0 - om), 0.0, zi,
                                      epsabs=1e-13, epsrel=1e-13)[0] for zi, om in zip(self.z, self.om_fid)])


def build_fs8_data(z, fs8, sigma, om_fid, block, names=None):
    """Diagonal covariance, except rows with block == 1 (WiggleZ) which share WIGGLEZ_COV (their order = file order)."""
    z, fs8, sigma = (np.asarray(a, dtype=float) for a in (z, fs8, sigma))
    block = np.asarray(block, dtype=int)
    cov = np.diag(sigma ** 2)
    idx = np.where(block == 1)[0]
    if len(idx):
        if len(idx) != 3:
            raise ValueError(f"block == 1 (WiggleZ) must have exactly 3 rows, found {len(idx)}")
        cov[np.ix_(idx, idx)] = WIGGLEZ_COV
    return FS8Data(z, fs8, cov, om_fid, names)


# ════════════════════════════════════════════════════════════════════════════
# GROWTH ODE
# ════════════════════════════════════════════════════════════════════════════
def _background_u(bg, N):
    """(u, u') = (H^2/H0_anchor^2, du/dN) at the e-folds N (array), anchor units.  u(0) = bg.u0."""
    z = np.expm1(-N)
    if bg.sol is not None:                                   # ODE f(R) backgrounds: dense solution y = (u, u')
        y = bg.sol.sol(N)
        return np.asarray(y[0]), np.asarray(y[1])
    if getattr(bg, "dudN", None) is not None:                # closed-form f(R) backgrounds
        return bg.u0 * np.asarray(bg.E2(z), dtype=float), np.asarray(bg.dudN(z), dtype=float)
    h = 1e-3                                                 # GR + parameterisation: u = E^2, 4th-order central difference in N
    E2 = lambda n: np.asarray(bg.E2(np.expm1(-n)), dtype=float)
    u = E2(N)
    uN = (-E2(N + 2.0 * h) + 8.0 * E2(N + h) - 8.0 * E2(N - h) + E2(N - 2.0 * h)) / (12.0 * h)   # error ~ h^4 (~1e-12)
    return u, uN


def growth_coefficients(bg, omega_m_phys, derivs, k, N):
    """
    A(N) = 2 + H'/H  and  B(N) = (3/2) Omega_m(N) G_eff/G on the grid N.  None if the background is unphysical.
    derivs = (f, f_R, f_RR, f_RRR) of the f(R) model (fR.py), or None for GR + parameterisation.
    """
    u, uN = _background_u(bg, N)
    if not (np.all(np.isfinite(u)) and np.all(u > 0.0) and np.all(np.isfinite(uN))):
        return None
    A = 2.0 + uN / (2.0 * u)
    omega_m_N = omega_m_phys * np.exp(-3.0 * N) * bg.u0 / u          # kappa rho_m / (3 H^2)
    if derivs is None:
        geff = 1.0
    else:
        R = 3.0 * uN + 12.0 * u
        if not (np.all(np.isfinite(R)) and np.all(R > 0.0)):
            return None
        F = derivs[1](R)
        fRR = derivs[2](R)
        if not (np.all(np.isfinite(F)) and np.all(F > 0.0) and np.all(np.isfinite(fRR))):
            return None
        q = (C_LIGHT_KMS * k * np.sqrt(bg.u0) * np.exp(-N) / 100.0) ** 2 * fRR / F
        geff = (1.0 / F) * (1.0 + 4.0 * q) / (1.0 + 3.0 * q)
    B = 1.5 * omega_m_N * geff
    return A, B


def solve_growth(N, A, B):
    """
    RK4 for  y1' = y2,  y2' = -A y2 + B y1  on the fine grid N (length 2n+1: nodes at even indices, midpoints at odd).
    Initial condition: slow-roll growing mode (local power law + first-order correction, see below), delta(N_i) = 1.
    Returns (N_nodes, delta, delta', delta'') on the n+1 nodes.
    """
    n = (len(N) - 1) // 2
    h = (N[-1] - N[0]) / n
    Al, Bl = A.tolist(), B.tolist()
    # f = delta'/delta obeys the Riccati equation  f' = -f^2 - A f + B.  Slow-roll: f0 = positive root of f0^2 + A f0 - B = 0
    # (the local power law delta = a^p), corrected to first order, f = f0 - f0'/(2 f0 + A), with f0' from the first nodes.
    # The correction removes most of the decaying mode excited by a start at finite z (check_growth.py, tests 1 and 7).
    root = np.sqrt(Al[0] ** 2 + 4.0 * Bl[0])
    f0 = 0.5 * (-Al[0] + root)
    h2 = 2.0 * h                                              # spacing of the nodes
    dA = (-3.0 * Al[0] + 4.0 * Al[2] - Al[4]) / (2.0 * h2)    # one-sided 3-point derivatives wrt N
    dB = (-3.0 * Bl[0] + 4.0 * Bl[2] - Bl[4]) / (2.0 * h2)
    f0_N = 0.5 * (-dA + (Al[0] * dA + 2.0 * dB) / root)
    p = f0 - f0_N / (2.0 * f0 + Al[0])
    y1, y2 = 1.0, float(p)
    d, dp = [y1], [y2]
    for j in range(n):
        a0, am, a1 = Al[2 * j], Al[2 * j + 1], Al[2 * j + 2]
        b0, bm, b1 = Bl[2 * j], Bl[2 * j + 1], Bl[2 * j + 2]
        k1a, k1b = y2, -a0 * y2 + b0 * y1
        t1, t2 = y1 + 0.5 * h * k1a, y2 + 0.5 * h * k1b
        k2a, k2b = t2, -am * t2 + bm * t1
        t1, t2 = y1 + 0.5 * h * k2a, y2 + 0.5 * h * k2b
        k3a, k3b = t2, -am * t2 + bm * t1
        t1, t2 = y1 + h * k3a, y2 + h * k3b
        k4a, k4b = t2, -a1 * t2 + b1 * t1
        y1 += h / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        y2 += h / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
        d.append(y1)
        dp.append(y2)
    d, dp = np.array(d), np.array(dp)
    N_nodes = N[0::2]
    ddp = -A[0::2] * dp + B[0::2] * d
    return N_nodes, d, dp, ddp


# ════════════════════════════════════════════════════════════════════════════
# AP RESCALING + FS8 FIT
# ════════════════════════════════════════════════════════════════════════════
_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def ap_ratio(bg, data):
    """
    ratio(z_i) = E chi / (E_fid chi_fid) for the model background (None if E^2 is not positive).
    chi(z_i) = int_0^{z_i} dz/E by 24-point Gauss-Legendre on each [0, z_i] (relative error << 1e-10 for smooth E).
    """
    zq = 0.5 * data.z[:, None] * (_GL_X[None, :] + 1.0)                 # (n_data, 24) nodes
    E2q = np.asarray(bg.E2(zq.ravel()), dtype=float).reshape(zq.shape)
    if not (np.all(np.isfinite(E2q)) and np.all(E2q > 0.0)):
        return None
    chi = 0.5 * data.z * (_GL_W[None, :] / np.sqrt(E2q)).sum(axis=1)
    E = np.sqrt(np.asarray(bg.E2(data.z), dtype=float))
    return (E * chi) / (data.e_fid * data.chi_fid)


class GrowthFit(namedtuple("GrowthFit", "lnL chi2_min sigma8 sigma8_err omega_m_phys A Bd chi2_0 g")):
    """Result of fs8_fit.  chi2_at(s) is chi^2 for a FIXED amplitude sigma_8,0 = s."""
    __slots__ = ()
    def chi2_at(self, s):
        return self.chi2_0 - 2.0 * s * self.Bd + s * s * self.A


def fs8_fit(bg, omega_m_phys, derivs, data, k=K_GROWTH_DEFAULT, z_start=Z_GROWTH_INIT, z_bg_max=None):
    """
    Growth prediction, AP rescaling and the analytically marginalised fs8 log-likelihood for one background.
    Returns a GrowthFit, or None if the background is unphysical (the sampler then rejects the point).
    z_bg_max = highest redshift at which the background exists (FR_Z_INIT for f(R) runs, None for GR).
    """
    z_i = z_start if z_bg_max is None else min(z_start, float(z_bg_max))
    if data.z.max() >= z_i:
        raise ValueError(f"growth starts at z = {z_i:g} but the fs8 data reach z = {data.z.max():g}; "
                         f"raise FR_Z_INIT (f(R) runs) or Z_GROWTH_INIT.")
    N_i = -np.log1p(z_i)
    n = int(np.ceil(-N_i / N_STEP))
    N = np.linspace(N_i, 0.0, 2 * n + 1)
    coef = growth_coefficients(bg, omega_m_phys, derivs, k, N)
    if coef is None:
        return None
    N_nodes, d, dp, ddp = solve_growth(N, *coef)
    if not (np.all(np.isfinite(d)) and np.all(d > 0.0)):
        return None
    shape = CubicHermiteSpline(N_nodes, dp, ddp)(-np.log1p(data.z)) / d[-1]       # delta'(z_i)/delta(0)
    ratio = ap_ratio(bg, data)
    if ratio is None:
        return None
    g = ratio * shape
    Cg = data.cinv @ g
    A = float(g @ Cg)
    Bd = float(Cg @ data.fs8)
    if not (np.isfinite(A) and A > 0.0):
        return None
    chi2_min = data.chi2_0 - Bd * Bd / A
    return GrowthFit(lnL=-0.5 * chi2_min - 0.5 * np.log(A), chi2_min=chi2_min, sigma8=Bd / A, sigma8_err=A ** -0.5,
                     omega_m_phys=omega_m_phys, A=A, Bd=Bd, chi2_0=data.chi2_0, g=g)


# ════════════════════════════════════════════════════════════════════════════
# DERIVED sigma_8 / S_8 FROM A CHAIN
# ════════════════════════════════════════════════════════════════════════════
DERIVED_NAMES = ("sigma_8,0", "S_8", "chi2_min_fs8")


def derived_sigma8_chain(flat_chain, build_background, evaluate, fs8_data, n_param_theta, max_samples=20000, seed=0):
    """
    sigma_8,0, S_8 = sigma_8,0 sqrt(Omega_m,phys/0.3) and the profiled fs8 chi^2 for the chain rows.
    For each row theta the conditional posterior of sigma_8,0 is Normal(Bd/A, 1/A) (flat prior), so ONE draw per row gives
    the marginal posterior.  If the chain has more than max_samples rows, every ceil(n/max_samples)-th row is used.
    Returns (index, derived): index = rows of flat_chain used, derived = array (len(index), 3) with columns DERIVED_NAMES,
    NaN where the background or growth solve failed.
    """
    rng = np.random.default_rng(seed)
    step = max(1, int(np.ceil(len(flat_chain) / max_samples)))
    index = np.arange(0, len(flat_chain), step)
    derived = np.full((len(index), 3), np.nan)
    for j, i in enumerate(index):
        p = flat_chain[i]
        theta_p, theta_fr = tuple(p[1:1 + n_param_theta]), tuple(p[1 + n_param_theta:])
        bg = build_background(theta_p, theta_fr)
        fit = None if bg is None else evaluate(bg, theta_p, theta_fr, fs8_data)
        if fit is None:
            continue
        draw = rng.normal(fit.sigma8, fit.sigma8_err)
        derived[j] = (draw, draw * np.sqrt(fit.omega_m_phys / 0.3), fit.chi2_min)
    return index, derived


def summarise_derived(derived, fs8_data):
    """Text lines (median, 16th and 84th percentile) for the columns of derived_sigma8_chain."""
    labels = ("sigma_8,0 (from fs8 data)", "S_8 = sigma_8,0 sqrt(Omega_m,phys/0.3)",
              "chi2_min(fs8 | theta), sigma_8,0 profiled")
    lines = []
    for col, name in enumerate(labels):
        arr = derived[:, col]
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            lines.append(f"{name}: no valid samples")
            continue
        med, lo, hi = np.median(arr), np.percentile(arr, 16), np.percentile(arr, 84)
        lines.append(f"{name}: {med:.4f} +{hi - med:.4f} / -{med - lo:.4f}   ({len(arr)} samples)")
    lines.append(f"fs8 data: N = {len(fs8_data.z)} points, z = {fs8_data.z.min():g}-{fs8_data.z.max():g}; "
                 f"sigma_8,0 is set by these data alone (no CMB normalisation).")
    return lines


def derived_sigma8_summary(flat_chain, build_background, evaluate, fs8_data, n_param_theta, n_samples=300, seed=0):
    """Summary lines from a subset of the chain (kept for scripts written against patch 1).  Returns (lines, derived)."""
    index, derived = derived_sigma8_chain(flat_chain, build_background, evaluate, fs8_data, n_param_theta,
                                          max_samples=n_samples, seed=seed)
    return summarise_derived(derived, fs8_data), {n: derived[:, c] for c, n in enumerate(DERIVED_NAMES)}


def derived_file_name(tag, run_mode):
    """File written next to the chain: derived_<tag>_<run_mode>.npz  (does not start with 'chain_': not mistaken for a chain)."""
    return f"derived_{tag}_{run_mode}.npz"
