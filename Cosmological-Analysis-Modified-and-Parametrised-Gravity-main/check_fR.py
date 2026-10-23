# check_fR.py
"""
Self-tests for the f(R) background solver (fR.py).  Run:  python check_fR.py

1. Bianchi / rr-equation: kappa p from fR_friedmann.py's closed form must vanish
   along the solution (matter has p = 0).  Independent test of the tt-based ODE.
2. Gauge degeneracy: E^2(z) is unchanged when (Omega_m, alpha) are moved along
   alpha * Omega_m^(n-1) = const  (power law) -> Omega_m must not be sampled.
3. GR passthrough: FR_MODEL = "GR" returns the parameterisation's own E^2(z).
4. Initial-condition convergence versus FR_Z_INIT (power law converges; R_log_R does not).
5. Pinned f(R) parameters (FR_FIXED_PARAMS) give the same background as sampling them.
6. R_lambda_R2 (perturbative attractor): alpha -> 0 is exactly LCDM; the closed form solves the tt
   equation symbolically through O(alpha); it matches the full ODE started on the attractor;
   and the truncation guard accepts small alpha and rejects large alpha.
"""
import numpy as np

import fR
import parameterisations as P
from likelihood import make_background_builder

OM = 0.3
E2_lcdm = lambda Om: (lambda z: P.E2_lcdm(z, (Om,)))
ok_all = True
def report(name, ok, detail, info=False):
    global ok_all
    if not info:
        ok_all &= bool(ok)
    print(f"[{'INFO' if info else 'PASS' if ok else 'FAIL'}] {name}: {detail}")

# ── 1. kappa p = 0 along the solution ──────────────────────────────────────────
for model, theta in [("power_law", (-2.7, 0.29)), ("power_law", (0.01, 1.5)), ("R_log_R", (0.05,))]:
    d = fR.FR_REGISTRY[model]["derivs"](theta)
    bg = fR.solve_background(E2_lcdm(OM), OM, d, 10.0, rtol=1e-10)
    if bg is None:
        report(f"kappa p [{model} {theta}]", False, "background could not be solved"); continue
    def state(N):
        u, v = bg.sol.sol(N); R = 3 * v + 12 * u
        return u, v, R, fR.R_N_from_matter(d, R, u, 3 * OM * np.exp(-3 * N))
    worst = 0.0
    for z in (4.0, 2.0, 1.0, 0.3, 0.1):
        N, h = -np.log1p(z), 1e-4
        u, v, R, R_N = state(N)
        R_NN = (state(N + h)[3] - state(N - h)[3]) / (2 * h)
        kp, kr = fR.kappa_p(d, R, R_N, R_NN, u, v), fR.kappa_rho(d, R, R_N, u)
        worst = max(worst, abs(kp / kr))
    report(f"kappa p/kappa rho [{model} {theta}]", worst < 1e-5, f"max |ratio| = {worst:.1e}  (threshold 1e-5, finite-difference limited)")

# ── 2. gauge degeneracy ───────────────────────────────────────────────────────
n, s, zs = 0.286, -6.5, np.linspace(0, 2.4, 25)
curves = []
for Om in (0.15, 0.3, 0.7):
    bg = fR.solve_background(E2_lcdm(Om), Om, fR.FR_REGISTRY["power_law"]["derivs"]((s / Om**(n - 1), n)), 10.0)
    curves.append((bg.E2(zs), Om / bg.u0))
dev = max(np.max(np.abs(c[0] / curves[0][0] - 1)) for c in curves)
report("gauge degeneracy [power_law]", dev < 2e-3,
       f"max dE2/E2 = {dev:.1e} across anchor Omega_m = 0.15..0.7; physical Omega_m/u0 = "
       f"{', '.join(f'{c[1]:.4f}' for c in curves)}")

# ── 3. GR passthrough ─────────────────────────────────────────────────────────
bg = make_background_builder("LCDM", "GR", 0.05, 0.17, 40.0)((0.31,), ())
report("GR passthrough", np.allclose(bg.E2(zs), P.E2_lcdm(zs, (0.31,)), rtol=0, atol=0), "E2(z) identical to parameterisation")

# ── 4. initial-condition convergence ──────────────────────────────────────────
for model, theta in [("power_law", (-2.7, 0.29)), ("R_log_R", (0.45,))]:
    mk = lambda zi: make_background_builder("LCDM", model, 0.05, 0.17, zi, OM)
    sens = [fR.ic_sensitivity(mk(zi), mk(2 * zi), (), theta) for zi in (10.0, 20.0, 40.0)]
    conv = sens[-1] is not None and sens[-1] < 1e-3
    report(f"IC convergence [{model} {theta}]", conv,
           "max|E2(zi)/E2(2zi)-1| for zi=10,20,40: " + ", ".join("fail" if s is None else f"{s:.1e}" for s in sens)
           + ("" if conv else "   <- NOT converged: results depend on FR_Z_INIT (expected for R_log_R)"),
           info=(model != "power_law"))

# ── 5. pinned parameters ──────────────────────────────────────────────────────
free = make_background_builder("LCDM", "power_law", 0.05, 0.17, 40.0, OM)((), (-2.7, 0.29))
pinned = make_background_builder("LCDM", "power_law", 0.05, 0.17, 40.0, OM, {"n": 0.29})((), (-2.7,))
report("pinned parameters [power_law n=0.29]", np.array_equal(free.E2(zs), pinned.E2(zs)),
       "E2(z) identical to the free-parameter build")

# ── 6. R_lambda_R2 ─────────────────────────────────────────────────────────────
Lam, m_ = 2.1, OM
bR2 = make_background_builder("LCDM", "R_lambda_R2", 0.05, 0.17, 2.5, OM)
bg0 = bR2((), (Lam, 0.0))
ref = P.E2_lcdm(zs, (m_ / bg0.u0,))
report("R_lambda_R2: alpha->0 is LCDM", np.max(np.abs(bg0.E2(zs) / ref - 1)) < 1e-12,
       f"max |E2/E2_LCDM - 1| = {np.max(np.abs(bg0.E2(zs) / ref - 1)):.1e}, Omega_m_phys = {m_/bg0.u0:.4f}")

try:
    import sympy as sp
    N_, mm, LL, aa = sp.symbols("N m Lambda alpha", positive=True)
    x_ = mm * sp.exp(-3 * N_)
    u_ = x_ + LL / 3 + sp.Rational(27, 2) * aa * x_**2 - aa**2 * (486 * LL * x_**2 + 729 * x_**3)
    R_ = 3 * sp.diff(u_, N_) + 12 * u_
    F_ = 1 + 2 * aa * R_
    tt = sp.expand(sp.simplify(3 * F_ * u_ + (R_ - 2 * LL + aa * R_**2 - R_ * F_) / 2 + 3 * u_ * sp.diff(F_, N_) - 3 * x_))
    ok = all(sp.simplify(tt.coeff(aa, k)) == 0 for k in (0, 1, 2))
    report("R_lambda_R2: tt residual vanishes through O(alpha^2) (sympy)", ok, "coefficients of alpha^0, alpha^1, alpha^2 are zero")
except ImportError:
    report("R_lambda_R2: symbolic check", True, "sympy not installed, skipped", info=True)

from scipy.integrate import solve_ivp
a_ = 1e-5
x_of = lambda N: m_ * np.exp(-3 * N)
u_of = lambda N: x_of(N) + Lam / 3 + 13.5 * a_ * x_of(N)**2 - a_**2 * (486 * Lam * x_of(N)**2 + 729 * x_of(N)**3)
dR2 = fR.FR_REGISTRY["R_lambda_R2"]["derivs"]((Lam, a_)); N_i = -np.log1p(4.0); hh = 1e-5
def rhs(N, y):
    R = max(3 * y[1] + 12 * y[0], 1e-9)
    return [y[1], fR.R_N_from_matter(dR2, R, max(y[0], 1e-9), 3 * m_ * np.exp(-3 * N)) / 3 - 4 * y[1]]
sol = solve_ivp(rhs, (N_i, 0), [u_of(N_i), (u_of(N_i + hh) - u_of(N_i - hh)) / (2 * hh)], method="Radau",
                rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.01)
Ng = -np.log1p(zs)
dev = np.max(np.abs(bR2((), (Lam, a_)).E2(zs) / (sol.sol(Ng)[0] / sol.sol(0.0)[0]) - 1))
report("R_lambda_R2: closed form vs full ODE (alpha=1e-5)", dev < 1e-4, f"max |dE2/E2| = {dev:.1e}")

ok_small, ok_big = bR2((), (Lam, 5e-5)), bR2((), (Lam, 2e-4))
report("R_lambda_R2: truncation guard", ok_small is not None and ok_big is None,
       f"alpha=5e-5 accepted (err est {ok_small.truncation_error:.1e}); alpha=2e-4 rejected (err est ~5e-3 > 1e-3)")

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
