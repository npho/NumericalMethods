"""Pure numpy math for the Runge-Kutta video. No plotting, no audio.

Everything shown on screen or spoken in the narration is computed here, so the
video cannot drift out of sync with the mathematics.

Test problems:  y' = y, y(0)=1            (exact e^t)
                y' = y (1 - y), y(0)=0.1  (logistic; nonlinear)
                y' = sqrt(t), y(0)=0      (not smooth at t=0: the honest limit of high order)
                x'' = -x                  (the spring, as in the leapfrog video)
"""
import math
import numpy as np

E = math.e


# ================================================================== integrators (scalar or vector y)
def euler(f, t0, y0, h, n):
    t = t0 + h * np.arange(n + 1)
    y = np.empty((n + 1,) + np.shape(y0)); y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * np.asarray(f(t[i], y[i]))
    return t, y


def midpoint(f, t0, y0, h, n):
    t = t0 + h * np.arange(n + 1)
    y = np.empty((n + 1,) + np.shape(y0)); y[0] = y0
    for i in range(n):
        k = np.asarray(f(t[i], y[i]))
        y[i + 1] = y[i] + h * np.asarray(f(t[i] + h / 2, y[i] + h / 2 * k))
    return t, y


def rk4_stages(f, t, y, h):
    k1 = np.asarray(f(t, y)); k2 = np.asarray(f(t + h / 2, y + h / 2 * k1))
    k3 = np.asarray(f(t + h / 2, y + h / 2 * k2)); k4 = np.asarray(f(t + h, y + h * k3))
    return k1, k2, k3, k4


def rk4(f, t0, y0, h, n):
    t = t0 + h * np.arange(n + 1)
    y = np.empty((n + 1,) + np.shape(y0)); y[0] = y0
    for i in range(n):
        k1, k2, k3, k4 = rk4_stages(f, t[i], y[i], h)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


METHODS = {'euler': euler, 'midpoint': midpoint, 'rk4': rk4}
EVALS = {'euler': 1, 'midpoint': 2, 'rk4': 4}
ORDER = {'euler': 1, 'midpoint': 2, 'rk4': 4}

f_lin = lambda t, y: y
f_log = lambda t, y: y * (1 - y)
f_sqrt = lambda t, y: math.sqrt(max(t, 0.0))
f_spr = lambda t, u: np.array([u[1], -u[0]])


def exact_log(t, y0=0.1): return 1.0 / (1.0 + (1 / y0 - 1) * np.exp(-t))
EXACT_SQRT_1 = 2.0 / 3.0                                       # y(1) = (2/3) t^{3/2}


def end_error(method, f, y0, exact, h_inv, T=1.0):
    n = int(h_inv)
    y = METHODS[method](f, 0.0, y0, T / n, n)[1][-1]
    return abs(float(y) - exact)


# ---- closed forms for y' = y: one step multiplies by a polynomial in h
def factor_poly(method, z):
    if method == 'euler': return 1 + z
    if method == 'midpoint': return 1 + z + z * z / 2
    return 1 + z + z * z / 2 + z ** 3 / 6 + z ** 4 / 24


def _x(method, h):
    """factor - 1 written without cancellation (matters when h is tiny)."""
    if method == 'euler': return h
    if method == 'midpoint': return h + h * h / 2
    return h + h * h / 2 + h ** 3 / 6 + h ** 4 / 24


def err_lin(method, n, T=1.0):
    h = T / n
    return abs(E ** T * -math.expm1(n * math.log1p(_x(method, h)) - T))


# ================================================================== scene 2-3: one step, h = 1
ONE = {m: float(METHODS[m](f_lin, 0, 1.0, 1.0, 1)[1][-1]) for m in METHODS}
ONE_ERR = {m: E - ONE[m] for m in ONE}
K = rk4_stages(f_lin, 0.0, 1.0, 1.0)                           # k1..k4 for the first step
WEIGHTS = (1, 2, 2, 1)
K_AVG = sum(w * k for w, k in zip(WEIGHTS, K)) / 6.0           # weighted average slope
ONE_RK4 = 1.0 + 1.0 * K_AVG

# ================================================================== scene 5: h = 1/2 to t = 2
TAB_H, TAB_N = 0.5, 4
TAB_T, TAB_Y = rk4(f_lin, 0.0, 1.0, TAB_H, TAB_N)
TAB_EULER = euler(f_lin, 0.0, 1.0, TAB_H, TAB_N)[1]
TAB_EXACT = math.exp(2.0)


# ================================================================== Simpson's rule connection
def rk4_step_pure_t(f, t, h):
    """One RK4 step of y' = f(t) (no y dependence) from y=0."""
    k1, k2, k3, k4 = rk4_stages(lambda tt, yy: f(tt), t, 0.0, h)
    return h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simpson(f, t, h): return h / 6 * (f(t) + 4 * f(t + h / 2) + f(t + h))


# ================================================================== scene 6: order
ORD_N = 2 ** np.arange(0, 9)                                    # n = 1 .. 256
ORD_H = 1.0 / ORD_N
ORD_LIN = {m: np.array([end_error(m, f_lin, 1.0, E, n) for n in ORD_N]) for m in METHODS}
ORD_LOG = {m: np.array([end_error(m, f_log, 0.1, float(exact_log(1.0)), n) for n in ORD_N]) for m in METHODS}
ORD_SQRT = {m: np.array([end_error(m, f_sqrt, 0.0, EXACT_SQRT_1, n) for n in 2 ** np.arange(0, 11)]) for m in METHODS}
ORD_N_SQRT = 2 ** np.arange(0, 11)


def local_slope(err, n_arr):
    """Log-log slope of error against h over the finest pair (errors above round-off)."""
    h = 1.0 / np.asarray(n_arr)
    return float(np.diff(np.log(err))[-1] / np.diff(np.log(h))[-1])


# ================================================================== scene 7: error versus cost
BUDGET = 100                                                    # slope evaluations
BUD = {}
for _m in METHODS:
    _n = BUDGET // EVALS[_m]
    BUD[_m] = dict(n=_n, lin=end_error(_m, f_lin, 1.0, E, _n), log=end_error(_m, f_log, 0.1, float(exact_log(1.0)), _n))
TOL = 1e-10


def evals_to_reach(method, tol=TOL):
    hi = 1
    while err_lin(method, hi) >= tol: hi *= 2
    lo = hi // 2 if hi > 1 else 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err_lin(method, mid) < tol: hi = mid
        else: lo = mid
    return hi * EVALS[method], hi


WORK_N = {'euler': np.unique(np.round(10 ** np.linspace(0, 10.5, 60)).astype(np.int64)),
          'midpoint': np.unique(np.round(10 ** np.linspace(0, 6.0, 40)).astype(np.int64)),
          'rk4': np.unique(np.round(10 ** np.linspace(0, 3.4, 30)).astype(np.int64))}
WORK_ERR = {m: np.array([err_lin(m, int(n)) for n in WORK_N[m]]) for m in WORK_N}

# ================================================================== scene 8: stability
STAB_Z = np.linspace(0, 4, 801)


def amp(method, z):
    """|R(-z)| for y' = -lambda y with z = lambda h: size of the per-step factor."""
    return abs(factor_poly(method, -np.asarray(z, dtype=float)))


def stable_limit(method):
    zs = np.linspace(1e-6, 10, 20001)
    bad = np.where(amp(method, zs) >= 1.0)[0]
    a, b = zs[bad[0] - 1], zs[bad[0]]
    for _ in range(60):
        m = 0.5 * (a + b)
        if amp(method, m) < 1.0: a = m
        else: b = m
    return 0.5 * (a + b)


LIMIT = {m: float(stable_limit(m)) for m in METHODS}
LIMIT_PER_EVAL = {m: LIMIT[m] / EVALS[m] for m in METHODS}

# ================================================================== scene 9: the spring, long run
SPR_H = 0.5
SPR_PERIODS = 200
SPR_N = int(round(SPR_PERIODS * 2 * math.pi / SPR_H))


def spr_energy(u): return 0.5 * (u[..., 0] ** 2 + u[..., 1] ** 2)


def rk4_spring(h=SPR_H, n=SPR_N):
    t, u = rk4(f_spr, 0.0, np.array([1.0, 0.0]), h, n)
    return t, u


def leapfrog_spring(h=SPR_H, n=SPR_N):
    """Kick-drift-kick, synchronised (x_n, v_n)."""
    x = np.empty(n + 1); v = np.empty(n + 1); x[0], v[0] = 1.0, 0.0
    for i in range(n):
        vh = v[i] - 0.5 * h * x[i]
        x[i + 1] = x[i] + h * vh
        v[i + 1] = vh - 0.5 * h * x[i + 1]
    return x, v


def rk4_energy_factor(h):
    """Exact per-step energy factor |R(ih)|^2 of RK4 on the spring: 1 - h^6/72 + h^8/576."""
    return 1 - h ** 6 / 72 + h ** 8 / 576


# ================================================================== spoken numbers
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n):
    n = int(n)
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_words(n % 100))
    if n < 1_000_000:
        return int_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + int_words(n % 1000))
    raise ValueError(n)


def words(x, nd=0):
    """2.34 -> 'two point three four' (nd=2). Negative numbers are spoken without sign."""
    s = f"{abs(x):.{nd}f}"
    ip, _, fp = s.partition('.')
    out = int_words(ip)
    if fp: out += " point " + " ".join(_ONES[int(d)] for d in fp)
    return out


def big_words(x):
    """Rough spoken sizes: 1.4e10 -> 'about fourteen billion', 130000 -> 'a hundred thirty thousand'."""
    if x >= 1e9: return words(x / 1e9, 1) + " billion"
    if x >= 1e6: return words(x / 1e6, 1) + " million"
    if x >= 1000:
        e = int(math.floor(math.log10(x)))
        return int_words(int(round(x, -(e - 2))))              # three significant figures
    return int_words(int(x))


def sci_words(x):
    """1.3e-4 -> 'one point three times ten to the minus four'."""
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(m, 1) >= 10: m /= 10; e += 1
    return f"{words(m, 1)} times ten to the minus {int_words(-e)}"


# ================================================================== facts for narration / scenes
def facts():
    F = {}
    F['e'] = E
    F['one'] = ONE; F['one_err'] = ONE_ERR
    F['k'] = [float(k) for k in K]; F['k_avg'] = float(K_AVG)
    F['tab_y'] = [float(v) for v in TAB_Y]; F['tab_exact'] = TAB_EXACT
    F['tab_err'] = TAB_EXACT - float(TAB_Y[-1]); F['tab_euler_end'] = float(TAB_EULER[-1])
    F['tab_euler_pct'] = 100 * (TAB_EXACT - float(TAB_EULER[-1])) / TAB_EXACT
    for m in METHODS:
        F['sl_lin_' + m] = local_slope(ORD_LIN[m], ORD_N)
        F['sl_log_' + m] = local_slope(ORD_LOG[m], ORD_N)
        F['sl_sqrt_' + m] = local_slope(ORD_SQRT[m], ORD_N_SQRT)
    F['rk4_err_over_h4'] = float(ORD_LIN['rk4'][-1] / ORD_H[-1] ** 4)
    F['rk4_const'] = E / 120.0
    F['bud'] = BUD
    for m in METHODS:
        F['ev_' + m], F['n_' + m] = evals_to_reach(m)
    F['limit'] = LIMIT; F['limit_per_eval'] = LIMIT_PER_EVAL
    t, u = rk4_spring()
    Er = spr_energy(u) / spr_energy(u[0])
    xl, vl = leapfrog_spring()
    El = 0.5 * (xl ** 2 + vl ** 2) / 0.5
    F['spr_rk4_end'] = float(Er[-1]); F['spr_rk4_factor'] = rk4_energy_factor(SPR_H)
    F['spr_rk4_pred_end'] = rk4_energy_factor(SPR_H) ** SPR_N
    F['spr_lf_min'] = float(El.min()); F['spr_lf_max'] = float(El.max())
    F['simp_val'] = simpson(math.exp, 0.0, 1.0); F['simp_rk4'] = rk4_step_pure_t(math.exp, 0.0, 1.0)
    F['simp_exact'] = math.e - 1.0
    F['spr_rk4_h025'] = rk4_energy_factor(0.25) ** int(round(SPR_PERIODS * 2 * math.pi / 0.25))
    return F


def spoken_facts(F):
    b = F['bud']
    return {
        'w_e': words(E, 3),
        'w_rk4_one': words(F['one']['rk4'], 3), 'w_mid_one': words(F['one']['midpoint'], 1),
        'w_rk4_one_err': words(F['one_err']['rk4'], 3), 'w_mid_one_err': words(F['one_err']['midpoint'], 3),
        'w_k_avg': words(F['k_avg'], 3),
        'w_tab_y4': words(F['tab_y'][4], 3), 'w_tab_exact': words(F['tab_exact'], 3),
        'w_tab_err': words(F['tab_err'], 3), 'w_tab_euler_pct': words(F['tab_euler_pct']),
        'w_sl_sqrt': words(F['sl_sqrt_rk4'], 1),
        'w_bud_n_rk4': int_words(b['rk4']['n']), 'w_bud_err_rk4': sci_words(b['rk4']['lin']),
        'w_bud_err_mid': sci_words(b['midpoint']['lin']), 'w_bud_err_euler': words(b['euler']['lin'], 3),
        'w_ev_euler': big_words(F['ev_euler']), 'w_ev_midpoint': big_words(F['ev_midpoint']),
        'w_ev_rk4': int_words(F['ev_rk4']),
        'w_lim_rk4': words(F['limit']['rk4'], 2),
        'w_lim_rk4_ev': words(F['limit_per_eval']['rk4'], 1),
        'w_spr_pct': words(100 * (1 - F['spr_rk4_end'])),
    }


FACTS = facts()
FACTS.update(spoken_facts(FACTS))

if __name__ == "__main__":
    for k, v in FACTS.items(): print(f"{k:18s} {v!r}")
