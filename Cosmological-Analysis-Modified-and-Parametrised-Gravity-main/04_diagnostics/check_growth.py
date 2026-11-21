# check_growth.py
"""
Self-tests of growth.py (growth ODE, G_eff, AP rescaling, sigma_8 marginalisation, Gold-2017 data).

    python 04_diagnostics/check_growth.py

Every check prints PASS / FAIL with the measured number.  The last block reproduces two chi^2 values quoted in
Nesseris, Pantazis & Perivolaropoulos (2017) as an end-to-end test of the data file + AP rescaling + growth solver.
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
from scipy.special import hyp2f1
from scipy.integrate import trapezoid

import data_loader as dl
import fR
import growth

FAILS = []


def report(name, value, tol, fmt=".2e"):
    ok = bool(abs(value) <= tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:{fmt}}   (tolerance {tol:g})")
    if not ok:
        FAILS.append(name)


def lcdm_bg(om):
    return fR.Background(lambda z: om * (1.0 + np.asarray(z, dtype=float)) ** 3 + 1.0 - om, 1.0)


def run_growth(bg, om_phys, derivs=None, k=0.1, z_start=10.0):
    """(N_nodes, delta, delta', delta'') using the production code path."""
    N_i = -np.log1p(z_start)
    n = int(np.ceil(-N_i / growth.N_STEP))
    N = np.linspace(N_i, 0.0, 2 * n + 1)
    A, B = growth.growth_coefficients(bg, om_phys, derivs, k, N)
    return growth.solve_growth(N, A, B)


def delta_lcdm_exact(a, om):
    """Growing mode of the Lambda CDM growth equation (Silveira & Waga; eq. 2.2 of Nesseris et al. 2017, w = -1)."""
    return a * hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, a ** 3 * (1.0 - 1.0 / om))


# ───────────────────────────────────────────────────────────────────────────
print("1. Lambda CDM growth vs the exact hypergeometric solution")
om = 0.3
Nn, d, dp, _ = run_growth(lcdm_bg(om), om)
a = np.exp(Nn)
ex = delta_lcdm_exact(a, om)
h = 1e-5
f_ex = (np.log(delta_lcdm_exact(a * np.exp(h), om)) - np.log(delta_lcdm_exact(a * np.exp(-h), om))) / (2 * h)
sel = (1.0 / a - 1.0) < 1.5      # the fs8 data live at z <= 1.4; the start at z = 10 leaves a decaying-mode transient above that
report("max |delta(a)/delta(1) - exact|, z < 1.5", np.max(np.abs((d / d[-1] - ex / ex[-1])[sel])), 1e-5)
report("max |f(z) - exact|, z < 1.5", np.max(np.abs((dp / d - f_ex)[sel])), 1e-5)

print("2. Einstein-de Sitter: delta = a, f = 1")
bg = fR.Background(lambda z: (1.0 + np.asarray(z, dtype=float)) ** 3, 1.0)
Nn, d, dp, _ = run_growth(bg, 1.0)
# the remaining ~1e-8 is the RK4 truncation error, which scales as N_STEP^4 (see test 6)
report("max |delta/a - 1|  (normalised at start)", np.max(np.abs(d / np.exp(Nn) / (d[0] / np.exp(Nn[0])) - 1.0)), 1e-7)
report("max |f - 1|", np.max(np.abs(dp / d - 1.0)), 1e-7)

print("3. Constant G_eff = 4/3 G (q -> infinity) in a matter era: f = p = (-1/2 + sqrt(1/4 + 8))/2")
big = (None, lambda R: 1.0 + 0.0 * R, lambda R: 1e30 + 0.0 * R, None)
Nn, d, dp, _ = run_growth(bg, 1.0, derivs=big)
p_exact = 0.5 * (-0.5 + np.sqrt(0.25 + 8.0))
report("max |f - p|", np.max(np.abs(dp / d - p_exact)), 1e-7)

print("4. f(R) code path with f_RR = 0 reproduces the GR path exactly")
gr_like = (None, lambda R: 1.0 + 0.0 * R, lambda R: 0.0 * R, None)
bgl = lcdm_bg(0.3)
bgl.u0, bgl.omega_m = 1.0, 0.3
_, d_gr, _, _ = run_growth(bgl, 0.3, derivs=None)
bg_f = fR.Background(bgl.E2, 1.0, 0.3, None, None, lambda z: -3.0 * 0.3 * (1.0 + np.asarray(z, dtype=float)) ** 3)
_, d_fr, _, _ = run_growth(bg_f, 0.3, derivs=gr_like)
report("max |delta_fR/delta_GR - 1|", np.max(np.abs(d_fr / d_gr - 1.0)), 1e-10)

print("5. Closed-form d u/dN (R_lambda_R2 / starobinsky) vs finite difference of E^2")
for name, Lam, al in (("R_lambda_R2", 2.1, 5e-5), ("starobinsky", 0.0, 5e-5)):
    bgs, why = (fR._background_R_lambda_R2(0.3, (Lam, al), 2.5) if name == "R_lambda_R2"
                else fR._background_starobinsky(0.3, (al,), 2.5))
    zz = np.linspace(0.01, 2.4, 50)
    hN = 1e-5
    u = lambda z: bgs.u0 * bgs.E2(z)
    fd = (u(np.expm1(-(-np.log1p(zz) + hN))) - u(np.expm1(-(-np.log1p(zz) - hN)))) / (2 * hN)
    report(f"{name}: max |dudN/fd - 1|", np.max(np.abs(bgs.dudN(zz) / fd - 1.0)), 1e-7)

print("6. Numerical convergence: RK4 step halving (Hu-Sawicki ODE background, scale-dependent G_eff)")
Lam, b = 2.1, 0.1
derivs = fR._derivs_hu_sawicki((Lam, b))
ics = fR._ics_hu_sawicki(0.3, (Lam, b), 10.0)
bg_hs = fR.solve_background(None, 0.3, derivs, 10.0, ics=ics, rtol=1e-9, method="Radau", max_rhs_evals=20000)
data = dl.load_fs8_data()
old = growth.N_STEP
growth.N_STEP = 0.01
f1 = growth.fs8_fit(bg_hs, 0.3 / bg_hs.u0, derivs, data, z_bg_max=10.0)
growth.N_STEP = 0.005
f2 = growth.fs8_fit(bg_hs, 0.3 / bg_hs.u0, derivs, data, z_bg_max=10.0)
growth.N_STEP = old
report("max |g(N_STEP=0.01)/g(0.005) - 1|", np.max(np.abs(f1.g / f2.g - 1.0)), 1e-6)
print(f"      (Hu-Sawicki b = {b}, k = 0.1 h/Mpc: sigma_8,0 = {f1.sigma8:.4f}, chi2_min = {f1.chi2_min:.2f})")

print("7. Initial-condition error: Lambda CDM-like closed-form f(R) started at z = 2.5 vs the exact solution")
bgs, _ = fR._background_R_lambda_R2(0.3, (2.1, 1e-9), 2.5)
om_phys = 0.3 / bgs.u0
Nn, d, dp, _ = run_growth(bgs, om_phys, derivs=fR._derivs_R_lambda_R2((2.1, 1e-9)), z_start=2.5)
a = np.exp(Nn)
ex = delta_lcdm_exact(a, om_phys)
zsel = (1.0 / a - 1.0) < 1.5
f_ex = (np.log(delta_lcdm_exact(a * np.exp(1e-5), om_phys)) - np.log(delta_lcdm_exact(a * np.exp(-1e-5), om_phys))) / 2e-5
report("max |delta(a)/delta(1) - exact|, z < 1.5 (data range)", np.max(np.abs((d / d[-1] - ex / ex[-1])[zsel])), 1e-3)
report("max |f - exact|, z < 1.5 (data range)", np.max(np.abs((dp / d - f_ex)[zsel])), 5e-3)
print("      (this is the cost of starting at z = 2.5, the limit of the closed-form backgrounds; models whose background reaches")
print("       z = 10 start there and have ~1e-6 error.  Data errors on fs8 are >= 3.8 %, so ~0.2 % is not the limiting error.)")

print("8. Analytic marginalisation over sigma_8,0 vs brute-force integration")
fit = growth.fs8_fit(lcdm_bg(0.3), 0.3, None, data)
s = np.linspace(fit.sigma8 - 12 * fit.sigma8_err, fit.sigma8 + 12 * fit.sigma8_err, 200001)
chi2 = np.array([fit.chi2_at(x) for x in s[::1]])
integral = trapezoid(np.exp(-0.5 * (chi2 - fit.chi2_min)), s)
report("ln[ integral / sqrt(2 pi / A) ]", np.log(integral / np.sqrt(2 * np.pi / fit.A)), 1e-6)
report("chi2_at(sigma8_hat) - chi2_min", fit.chi2_at(fit.sigma8) - fit.chi2_min, 1e-9)

print("9. AP rescaling: ratio = 1 when the model equals every point's fiducial cosmology")
dat1 = growth.build_fs8_data(data.z, data.fs8, np.sqrt(np.diag(data.cov)), np.full(len(data.z), 0.3), np.zeros(len(data.z), int))
r = growth.ap_ratio(lcdm_bg(0.3), dat1)
report("max |ratio - 1|", np.max(np.abs(r - 1.0)), 1e-10)

print("10. End-to-end: reproduce chi^2 values quoted for the Gold-2017 compilation (Nesseris et al. 2017, sec. III)")
fit = growth.fs8_fit(lcdm_bg(0.3156), 0.3156, None, data)
print(f"      Planck15 LCDM (Omega_m = 0.3156, sigma_8 = 0.831): chi2 = {fit.chi2_at(0.831):.2f}   (paper: 22.8)")
report("chi2 - 22.8", fit.chi2_at(0.831) - 22.8, 0.25, fmt=".2f")
fit = growth.fs8_fit(lcdm_bg(0.21), 0.21, None, data)
print(f"      best-fit LCDM (Omega_m = 0.21): chi2 = {fit.chi2_at(0.88):.2f} at sigma_8 = 0.88, profile minimum "
      f"{fit.chi2_min:.2f} at sigma_8 = {fit.sigma8:.3f}   (paper: 11.8, 0.88)")
report("chi2 - 11.8", fit.chi2_at(0.88) - 11.8, 0.25, fmt=".2f")

print()
print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
_sys.exit(1 if FAILS else 0)
