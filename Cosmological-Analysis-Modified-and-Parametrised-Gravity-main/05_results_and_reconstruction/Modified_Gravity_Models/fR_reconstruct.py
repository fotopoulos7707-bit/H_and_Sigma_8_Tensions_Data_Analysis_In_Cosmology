"""
fR_reconstruct.py -- reconstruct f(R) from a dark-energy equation-of-state w(z),
and compute the effective w_DE of any f(R) from F(N), f(N).

N = ln a, ' = d/dN, u = H^2, units: H0 = 1, kappa = 8 pi G.
Background:  u = Om e^{-3N} + (1-Om) fDE(N),  fDE = exp(-3 Int (1+w) dN)
Curvature:   R = 3u' + 12u            (since Hdot = u'/2)
tt equation (flat):  3uF' = k rho_m - 3uF - (f - F R)/2 ,  k rho_m = 3 Om e^{-3N}
with f' = F R'.   First-order system in (f, F).
"""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp

# ----------------------------------------------------------- user parametrisation
N = sp.Symbol('N')
Om, w0, wa = sp.symbols('Om w0 wa')
def w_of_N(N):                      # <<< put YOUR w(a) here, as a function of N = ln a
    a = sp.exp(N)
    return w0 + wa*(1 - a)          # CPL
# ---------------------------------------------------------------------------------
x = sp.Symbol('x')
intg = sp.integrate(sp.expand(1 + w_of_N(x)), (x, 0, N))     # Int_0^N (1+w) dN'
fDE = sp.exp(-3*intg)
u = Om*sp.exp(-3*N) + (1 - Om)*fDE
R = 3*sp.diff(u, N) + 12*u
args = (N, Om, w0, wa)
u_f, up_f, R_f, Rp_f, fDE_f = [sp.lambdify(args, e, 'numpy') for e in
                        (u, sp.diff(u, N), R, sp.diff(R, N), fDE)]

def reconstruct(Om_v, w0_v, wa_v, N_i=-7.0, N_f=0.5, n=2000):
    p = (Om_v, w0_v, wa_v)
    krho_m = lambda N: 3*Om_v*np.exp(-3*N)
    def rhs(N, y):
        f, F = y
        uu, RR, Rp = u_f(N, *p), R_f(N, *p), Rp_f(N, *p)
        Fp = (krho_m(N) - 3*uu*F - (f - F*RR)/2)/(3*uu)
        return [F*Rp, Fp]
    # early-time ICs: GR plus a cosmological constant equal to the DE density then
    Ri = R_f(N_i, *p)
    Lam_i = 3*(1 - Om_v)*fDE_f(N_i, *p)          # = kappa rho_DE at N_i  (H0 = 1)
    y0 = [Ri - 2*Lam_i, 1.0]
    Ns = np.linspace(N_i, N_f, n)
    sol = solve_ivp(rhs, (N_i, N_f), y0, t_eval=Ns, method='Radau', rtol=1e-10, atol=1e-13)
    return sol, p

def w_eff(Ns, f, F, p):
    """Effective DE equation of state from F(N), f(N) (flat, uses F', F'' analytically)."""
    uu, up, RR, Rp = [fn(Ns, *p) for fn in (u_f, up_f, R_f, Rp_f)]
    km = 3*p[0]*np.exp(-3*Ns)
    Fp = (km - 3*uu*F - (f - F*RR)/2)/(3*uu)
    Fpp = (-3*km - 3*up*F - 3*uu*Fp + Fp*RR/2)/(3*uu) - (up/uu)*Fp
    rhoDE = 3*uu*(1 - F) + (F*RR - f)/2 - 3*uu*Fp          # kappa rho_DE
    rpDE = uu*Fpp + 0.5*up*Fp - uu*Fp + up*(F - 1)          # kappa (rho_DE + p_DE)
    return -1 + rpDE/rhoDE, rhoDE

if __name__ == "__main__":
    Om_v = 0.3
    # (w0, wa): the first two have w -> -1 in the past (viable-looking), the last does not
    for (w0_v, wa_v) in [(-0.95, -0.05), (-0.9, -0.1), (-0.9, 0.2)]:
        sol, p = reconstruct(Om_v, w0_v, wa_v)
        Ns, (f, F) = sol.t, sol.y
        w, rhoDE = w_eff(Ns, f, F, p)
        w_in = w0_v + wa_v*(1 - np.exp(Ns))
        print(f"\nw0={w0_v}, wa={wa_v}   (solver status {sol.status})")
        print(f"{'z':>5} {'F=f_R':>14} {'w_input':>9} {'w_eff[f(R)]':>12}")
        for z in (0, 0.5, 1, 2, 5, 10, 50):
            i = np.argmin(abs(Ns + np.log(1 + z)))
            print(f"{z:5.1f} {F[i]:14.8f} {w_in[i]:9.4f} {w[i]:12.4f}")