"""
fR_friedmann.py -- Friedmann equations of metric f(R) gravity on k-curved FLRW.

Field eq:  F R_ab - f g_ab/2 - Nabla_a Nabla_b F + g_ab Box F = kappa T_ab,  F = f_R
           kappa rho = E_tt,   kappa p = E^r_r      (signature -+++)

Pipeline
  1. Full tensor computation (metric -> Christoffel -> Ricci -> Hessian of F)
     for generic F(t), f(t); self-checks (R, isotropy, closed forms, Bianchi).
  2. Models: (a) full tensor route with F = f'(R[a(t)]), converted to H(t);
             (b) closed-form route with the verified generic formulas.
     They must agree (checked numerically at random points).
  3. Print results (plain + LaTeX); curvature kept only if INCLUDE_CURVATURE.
"""
import time
import sympy as sp

INCLUDE_CURVATURE = False
EXPAND_IN_H = False   # also print the fully expanded H, H', H'', H''' form (long)
T0 = time.time()
def stamp(msg):
    print(f"[{time.time()-T0:6.2f}s] {msg}")

# ---------------------------------------------------------------- 1. geometry
t, r, th, ph = sp.symbols('t r theta phi', real=True)
k, kappa, alpha, n = sp.symbols('k kappa alpha n', real=True)
X = [t, r, th, ph]
a = sp.Function('a')(t)
Ft, ft = sp.Function('F')(t), sp.Function('f')(t)

g = sp.diag(-1, a**2/(1 - k*r**2), a**2*r**2, a**2*r**2*sp.sin(th)**2)
gi = sp.simplify(g.inv())

Gam = [[[sp.simplify(sum(gi[l, s]*(sp.diff(g[s, m], X[v]) + sp.diff(g[s, v], X[m])
        - sp.diff(g[m, v], X[s])) for s in range(4))/2)
        for v in range(4)] for m in range(4)] for l in range(4)]

def ricci(m, v):
    e = sum(sp.diff(Gam[l][m][v], X[l]) for l in range(4))
    e -= sum(sp.diff(Gam[l][m][l], X[v]) for l in range(4))
    e += sum(Gam[l][l][s]*Gam[s][m][v] for l in range(4) for s in range(4))
    e -= sum(Gam[l][v][s]*Gam[s][m][l] for l in range(4) for s in range(4))
    return sp.simplify(e)

Ric = sp.Matrix(4, 4, lambda m, v: ricci(m, v))
Rs = sp.simplify(sum(gi[m, v]*Ric[m, v] for m in range(4) for v in range(4)))
ok_R = sp.simplify(Rs - 6*(a.diff(t, 2)/a + a.diff(t)**2/a**2 + k/a**2)) == 0
stamp(f"R(t) = {Rs};  matches 6(a''/a + a'^2/a^2 + k/a^2): {ok_R}")

def field_eqs(F, f):
    dF = [sp.diff(F, x) for x in X]
    Hess = sp.Matrix(4, 4, lambda m, v: sp.diff(F, X[m], X[v])
                     - sum(Gam[l][m][v]*dF[l] for l in range(4)))
    box = sum(gi[m, v]*Hess[m, v] for m in range(4) for v in range(4))
    E = F*Ric - f*g/2 - Hess + box*g
    return E, gi*E          # E_ab, E^a_b

# ---------------------------------------------- generic F(t), f(t) + self-checks
E, Em = field_eqs(Ft, ft)
krho, kp = E[0, 0], Em[1, 1]

ok_iso = all(sp.simplify(Em[i, i] - Em[1, 1]) == 0 for i in (2, 3))
ok_off = all(sp.simplify(Em[i, j]) == 0 for i in range(4) for j in range(4) if i != j)
stamp(f"isotropy: {ok_iso};  off-diagonal zero: {ok_off}")

# Hubble form: a'/a = H, using chain rule on a symbol aS (aS' = H aS)
H = sp.Function('H')(t)
aS = sp.Symbol('aS', positive=True)
Fh, fh = sp.Function('F')(t), sp.Function('f')(t)

def ddt(e):
    return sp.diff(e, t) + sp.diff(e, aS)*H*aS

def to_hubble(e):
    """a(t) -> aS with a' = H a, a'' = (H'+H^2) a, ... via repeated chain rule."""
    reps, d = {}, aS
    for m in range(1, 7):
        d = ddt(d)
        reps[sp.Derivative(a, (t, m))] = d
    for m in range(6, 0, -1):
        e = e.subs(sp.Derivative(a, (t, m)), reps[sp.Derivative(a, (t, m))])
    return e.subs(a, aS)

Rh = 6*(H.diff(t) + 2*H**2 + k/aS**2)
rho_gen = sp.simplify(to_hubble(krho))
p_gen = sp.simplify(to_hubble(kp))
rho_exp = 3*Ft*(H**2 + k/aS**2) + (ft - Ft*Rh)/2 + 3*H*Ft.diff(t)
p_exp = Ft*(H.diff(t) + 3*H**2 + 2*k/aS**2) - ft/2 - Ft.diff(t, 2) - 2*H*Ft.diff(t)
ok_rho = sp.simplify(rho_gen - rho_exp) == 0
ok_p = sp.simplify(p_gen - p_exp) == 0
stamp(f"kappa rho == 3F(H^2+k/a^2)+(f-FR)/2+3HF' : {ok_rho}")
stamp(f"kappa p   == F(H'+3H^2+2k/a^2)-f/2-F''-2HF' : {ok_p}")

# Bianchi: rho' + 3H(rho+p) = 0 once f' = F R'
Hs = a.diff(t)/a
cont = (sp.diff(krho, t) + 3*Hs*(krho + kp)).subs(ft.diff(t), Ft*sp.diff(Rs, t))
ok_cont = sp.simplify(cont) == 0
stamp(f"continuity (generic, f'=F R'): {ok_cont}")

# --------------------------------------------------------------------- 2. models
Rsym = sp.Symbol('R', positive=True)
models = {
    "f = R":               Rsym,
    "f = R + alpha R^n":   Rsym + alpha*Rsym**n,
    "f = R (1 + alpha ln R)": Rsym*(1 + alpha*sp.log(Rsym)),
}

def closed_form(fR):
    """Route (b): generic verified formulas with F, f as explicit functions of (H, aS)."""
    F = sp.diff(fR, Rsym).subs(Rsym, Rh)
    f = fR.subs(Rsym, Rh)
    F1 = ddt(F); F2 = ddt(F1)
    rho = 3*F*(H**2 + k/aS**2) + (f - F*Rh)/2 + 3*H*F1
    p = F*(H.diff(t) + 3*H**2 + 2*k/aS**2) - f/2 - F2 - 2*H*F1
    return rho, p

R0, Rd, Rdd = sp.symbols('R Rdot Rddot')   # R(t), dR/dt, d^2R/dt^2

def compact_form(fR):
    """Readable form:  F = f_R, F' = f_RR R', F'' = f_RRR R'^2 + f_RR R''."""
    F = sp.diff(fR, Rsym).subs(Rsym, R0)
    f = fR.subs(Rsym, R0)
    F1 = sp.diff(fR, Rsym, 2).subs(Rsym, R0)*Rd
    F2 = sp.diff(fR, Rsym, 3).subs(Rsym, R0)*Rd**2 + sp.diff(fR, Rsym, 2).subs(Rsym, R0)*Rdd
    rho = 3*F*(H**2 + k/aS**2) + (f - R0*F)/2 + 3*H*F1
    p = F*(R0/6 + H**2 + k/aS**2) - f/2 - F2 - 2*H*F1
    return rho, p

def compact_to_bg(e):
    Rbg = Rh.subs(aS, a_bg).subs(H, H_bg).doit()
    reps = {R0: Rbg, Rd: sp.diff(Rbg, t), Rdd: sp.diff(Rbg, t, 2)}
    e = e.subs(aS, a_bg).subs(H, H_bg).doit().subs(reps)
    return e.subs(num).subs(t, sp.Rational(17, 10))

def full_tensor(fR):
    """Route (a): full tensor computation with F = f_R(R[a(t)])."""
    F = sp.diff(fR, Rsym).subs(Rsym, Rs)
    f = fR.subs(Rsym, Rs)
    E, Em = field_eqs(F, f)
    return E[0, 0], Em[1, 1]

# numeric test background: a = t^(2/3), k=3/10, alpha=3/10, n=9/5, evaluated at t=17/10
num = {k: sp.Rational(3, 10), alpha: sp.Rational(3, 10), n: sp.Rational(9, 5)}
a_bg = t**sp.Rational(2, 3)
H_bg = sp.Rational(2, 3)/t

def hubble_to_bg(e):
    e = e.subs(aS, a_bg).subs(H, H_bg).doit()   # H(t) -> 2/(3t), derivatives evaluated
    return e.subs(num).subs(t, sp.Rational(17, 10))

pt = {r: sp.Rational(1, 3), th: 1}      # E^r_r and E_tt must not depend on these

def tensor_to_bg(e):
    e = e.subs(a, a_bg).doit()
    return e.subs(num).subs(pt).subs(t, sp.Rational(17, 10))

results = {}
for label, fR in models.items():
    t1 = time.time()
    rho_c, p_c = closed_form(fR)
    rho_t, p_t = full_tensor(fR)

    rho_k, p_k = compact_form(fR)

    # agreement of the three routes at a numeric point
    d_rho = sp.N(hubble_to_bg(rho_c) - tensor_to_bg(rho_t), 30)
    d_p = sp.N(hubble_to_bg(p_c) - tensor_to_bg(p_t), 30)
    d_rho2 = sp.N(compact_to_bg(rho_k) - tensor_to_bg(rho_t), 30)
    d_p2 = sp.N(compact_to_bg(p_k) - tensor_to_bg(p_t), 30)

    # Bianchi check on route (b): rho' + 3H(rho+p)
    bian = ddt(rho_c) + 3*H*(rho_c + p_c)
    d_b = sp.N(hubble_to_bg(bian), 30)

    k_rule = {} if INCLUDE_CURVATURE else {k: 0}
    rho_s = sp.simplify(rho_k.subs(k_rule))
    p_s = sp.simplify(p_k.subs(k_rule))
    results[label] = (rho_s, p_s)

    print(f"\n=== {label} ===   ({time.time()-t1:.2f}s)")
    print("  with R = 6(H' + 2H^2" + (" + k/a^2" if INCLUDE_CURVATURE else "") + "), Rdot = dR/dt, Rddot = d^2R/dt^2")
    print("  kappa rho =", rho_s)
    print("  kappa p   =", p_s)
    print("  LaTeX rho :", sp.latex(rho_s))
    print("  LaTeX p   :", sp.latex(p_s))
    if EXPAND_IN_H:
        print("  kappa rho (expanded in H) =", sp.simplify(rho_c.subs(k_rule)))
        print("  kappa p   (expanded in H) =", sp.simplify(p_c.subs(k_rule)))
    print(f"  full tensor vs H-closed-form : d_rho={d_rho}, d_p={d_p}")
    print(f"  full tensor vs compact form  : d_rho={d_rho2}, d_p={d_p2}")
    print(f"  Bianchi check (numeric, ~0): {d_b}")

stamp("done")