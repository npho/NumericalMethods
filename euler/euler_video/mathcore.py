"""Pure numpy math for the Euler-method video. No plotting, no audio.

Everything shown on screen or spoken in the narration is computed here, so the
video cannot drift out of sync with the mathematics.

Main test problem:   y' = y,  y(0) = 1        (exact: e^t)
Other problems:      y' = -5 y                (stability)
                     y' = y (1 - y), y(0)=0.1 (logistic: order is not special to the linear case)
                     x'' = -x as a system     (the spring, as in the leapfrog video)
"""
import math
import numpy as np

E = math.e


# ================================================================== integrators
def euler(f, t0, y0, h, n):
    """Explicit (forward) Euler. Returns t[0..n], y[0..n]."""
    t = t0 + h * np.arange(n + 1)
    y = np.empty(n + 1); y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y


def midpoint(f, t0, y0, h, n):
    t = t0 + h * np.arange(n + 1)
    y = np.empty(n + 1); y[0] = y0
    for i in range(n):
        k = f(t[i], y[i])
        y[i + 1] = y[i] + h * f(t[i] + h / 2, y[i] + h / 2 * k)
    return t, y


def rk4(f, t0, y0, h, n):
    t = t0 + h * np.arange(n + 1)
    y = np.empty(n + 1); y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i]); k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2); k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


f_lin = lambda t, y: y
f_dec = lambda t, y: -5.0 * y
f_log = lambda t, y: y * (1 - y)
LAM = 5.0


def exact_log(t, y0=0.1): return 1.0 / (1.0 + (1 / y0 - 1) * np.exp(-t))


# ---- closed forms for y' = y (one step multiplies y by a polynomial in h); used for huge n
def _L(method, h):
    if method == 'euler': return math.log1p(h)
    if method == 'midpoint': return math.log1p(h + h * h / 2)
    if method == 'rk4': return math.log1p(h + h * h / 2 + h ** 3 / 6 + h ** 4 / 24)
    raise ValueError(method)


def err_lin(method, n, T=1.0):
    """|e^T - y_n| for y' = y with n steps over [0, T], by closed form (precise even when tiny)."""
    h = T / n
    return abs(E ** T * -math.expm1(n * _L(method, h) - T))


EVALS = {'euler': 1, 'midpoint': 2, 'rk4': 4}                 # slope evaluations per step


# ================================================================== scene 2: one step
ONE_H = 1.0
ONE_Y1 = 1.0 + ONE_H * 1.0
ONE_ERR = E - ONE_Y1

# ================================================================== scene 3: h = 1/2 to t = 2
TAB_H, TAB_N = 0.5, 4
TAB_T, TAB_Y = euler(f_lin, 0.0, 1.0, TAB_H, TAB_N)
TAB_EXACT = math.exp(2.0)
TAB_ERR = TAB_EXACT - TAB_Y[-1]

# ================================================================== scene 4: smaller steps, aim at t = 1
H_LIST = [1.0, 0.5, 0.25, 0.1]
H_VAL = [euler(f_lin, 0.0, 1.0, h, int(round(1 / h)))[1] for h in H_LIST]
LIM_N = 1000
LIM_VAL = (1 + 1 / LIM_N) ** LIM_N

# ================================================================== scene 5: order
ORD_N = 2 ** np.arange(0, 11)                                  # n = 1 .. 1024, h = 1/n
ORD_H = 1.0 / ORD_N
ORD_LIN = np.array([abs(E - euler(f_lin, 0, 1, 1.0 / n, int(n))[1][-1]) for n in ORD_N])
ORD_LOG = np.array([abs(exact_log(1.0) - euler(f_log, 0, 0.1, 1.0 / n, int(n))[1][-1]) for n in ORD_N])


def local_slope(err):
    """Log-log slope against h over the finest step pair."""
    return float(np.diff(np.log(err))[-1] / np.diff(np.log(ORD_H))[-1])


LOC_H = 0.1
LOC_ERR = math.exp(LOC_H) - (1 + LOC_H)                        # one step from (0, 1)
LOC_PRED = 0.5 * LOC_H ** 2                                    # (1/2) y'' h^2 with y'' = 1

# ================================================================== scenes 6-7: stability
STAB_H = [0.1, 0.3, 0.5]
STAB_T = 3.0
STAB = {}
for _h in STAB_H:
    _n = int(round(STAB_T / _h))
    STAB[_h] = euler(f_dec, 0.0, 1.0, _h, _n)
STAB_LIMIT = 2 / LAM


def euler_factor(z): return abs(1 - z)
def backward_factor(z): return 1.0 / (1.0 + z)
def midpoint_factor(z): return abs(1 - z + z * z / 2)
def rk4_factor(z): return abs(1 - z + z * z / 2 - z ** 3 / 6 + z ** 4 / 24)


def stable_limit(factor, lo=1e-6, hi=10.0):
    """Smallest z>0 where |factor(z)| reaches 1 (bisection, scanning for the first crossing)."""
    zs = np.linspace(lo, hi, 20001)
    bad = np.where(np.array([factor(z) for z in zs]) >= 1.0)[0]
    a, b = zs[bad[0] - 1], zs[bad[0]]
    for _ in range(60):
        m = 0.5 * (a + b)
        if factor(m) < 1.0: a = m
        else: b = m
    return 0.5 * (a + b)


# ================================================================== scene 8: Euler on the spring
SPR_H = 0.2
SPR_N = 60


def spring_euler(h=SPR_H, n=SPR_N):
    x = np.empty(n + 1); v = np.empty(n + 1); x[0], v[0] = 1.0, 0.0
    for i in range(n):
        x[i + 1] = x[i] + h * v[i]
        v[i + 1] = v[i] - h * x[i]
    return x, v


# ================================================================== scene 9: error versus work
TOL = 1e-6


def evals_to_reach(method, tol=TOL):
    """Fewest slope evaluations (over [0,1], y' = y) for error below tol (n found by doubling, then bisection)."""
    hi = 1
    while err_lin(method, hi) >= tol: hi *= 2
    lo = hi // 2 if hi > 1 else 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err_lin(method, mid) < tol: hi = mid
        else: lo = mid
    return hi * EVALS[method], hi


WORK_N = {'euler': np.unique(np.round(10 ** np.linspace(0, 6.3, 44)).astype(int)),
          'midpoint': np.unique(np.round(10 ** np.linspace(0, 4, 32)).astype(int)),
          'rk4': np.unique(np.round(10 ** np.linspace(0, 2.6, 24)).astype(int))}
WORK_ERR = {m: np.array([err_lin(m, int(n)) for n in WORK_N[m]]) for m in WORK_N}


def work_slope(m):
    """log-log slope of error against slope evaluations, over the last several points above the round-off floor."""
    ev = WORK_N[m] * EVALS[m]; er = WORK_ERR[m]
    ok = er > 1e-12
    ev, er = ev[ok][-6:], er[ok][-6:]
    return float(np.polyfit(np.log(ev), np.log(er), 1)[0])


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
    """2.34 -> 'two point three' (nd=1). Negative numbers are spoken without sign; say 'minus' in the script."""
    s = f"{abs(x):.{nd}f}"
    ip, _, fp = s.partition('.')
    out = int_words(ip)
    if fp: out += " point " + " ".join(_ONES[int(d)] for d in fp)
    return out


def round_sig(x, k=2):
    e = math.floor(math.log10(abs(x)))
    return round(x, -(e - k + 1))


def big_words(x):
    """About-numbers for the voiceover: 1.36e6 -> 'one point four million', 1344 -> 'thirteen hundred'."""
    if x >= 1e6: return words(x / 1e6, 1) + " million"
    r = round_sig(x, 2)
    if r >= 1000 and (r // 100) % 10 == 0 or r < 1000: return int_words(r)
    return int_words(r // 100) + " hundred" if r % 100 == 0 else int_words(r)


# ================================================================== facts for narration / scenes
def facts():
    F = {}
    F['e'] = E
    F['one_y1'] = ONE_Y1; F['one_err'] = ONE_ERR
    F['tab_y'] = list(TAB_Y); F['tab_exact'] = TAB_EXACT; F['tab_err'] = TAB_ERR
    F['tab_pct'] = 100 * TAB_ERR / TAB_EXACT
    F['h_val'] = [float(v[-1]) for v in H_VAL]
    F['lim_val'] = LIM_VAL; F['lim_n'] = LIM_N
    F['sl_lin'] = local_slope(ORD_LIN); F['sl_log'] = local_slope(ORD_LOG)
    F['err_over_h'] = float(ORD_LIN[-1] / ORD_H[-1])
    F['loc_err'] = LOC_ERR; F['loc_pred'] = LOC_PRED
    n3 = 1
    while err_lin('euler', n3) >= 1e-3: n3 += 1
    F['n_1e3'] = n3                                              # Euler steps to get below 1e-3
    F['stab_limit'] = STAB_LIMIT
    F['stab_fac'] = [1 - LAM * h for h in STAB_H]
    F['stab_end'] = [float(STAB[h][1][-1]) for h in STAB_H]
    F['be_fac_05'] = backward_factor(LAM * 0.5)
    F['lim_mid'] = stable_limit(midpoint_factor); F['lim_rk4'] = stable_limit(rk4_factor)
    x, v = spring_euler()
    Ee = 0.5 * (x ** 2 + v ** 2)
    n1 = int(round(2 * math.pi / SPR_H))
    F['spr_fac'] = 1 + SPR_H ** 2; F['spr_E1p'] = float(Ee[n1] / Ee[0])
    for m in ('euler', 'midpoint', 'rk4'):
        F['ev_' + m], F['n_' + m] = evals_to_reach(m)
        F['wsl_' + m] = work_slope(m)
    return F


def spoken_facts(F):
    return {
        'w_e': words(E, 3), 'w_one_err': words(F['one_err'], 3),
        'w_tab_y4': words(F['tab_y'][4], 2), 'w_tab_exact': words(F['tab_exact'], 3), 'w_tab_pct': words(F['tab_pct']),
        'w_hv_025': words(F['h_val'][2], 2), 'w_hv_01': words(F['h_val'][3], 2), 'w_lim': words(F['lim_val'], 3),
        'w_n_1e3': int_words(int(round(F['n_1e3'], -1))),
        'w_err_over_h': words(F['err_over_h'], 2),
        'w_spr_E1p': words(F['spr_E1p'], 1),
        'w_ev_euler': big_words(F['ev_euler']), 'w_ev_midpoint': big_words(F['ev_midpoint']),
        'w_ev_rk4': big_words(F['ev_rk4']),
    }


FACTS = facts()
FACTS.update(spoken_facts(FACTS))

if __name__ == "__main__":
    for k, v in FACTS.items(): print(f"{k:16s} {v!r}")
