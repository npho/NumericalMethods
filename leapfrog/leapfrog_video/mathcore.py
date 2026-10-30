"""Pure-Python/numpy math for the leapfrog video. No plotting, no audio.

Everything shown on screen or spoken in the narration is computed here, so the
video cannot drift out of sync with the mathematics.

Two test problems:
  * harmonic oscillator   x'' = -x,            x(0)=1, v(0)=0   (exact: cos t)
  * Kepler orbit          x'' = -x/|x|^3       GM=1, a=1, started at apoapsis
"""
import math, os, pickle
import numpy as np

# =========================================================================== harmonic oscillator
TWO_PI = 2 * math.pi


def ho_euler(dt, n, x0=1.0, v0=0.0):
    """Explicit Euler (both updates use the OLD values). Returns arrays x[0..n], v[0..n]."""
    x = np.empty(n + 1); v = np.empty(n + 1); x[0], v[0] = x0, v0
    for i in range(n):
        x[i + 1] = x[i] + dt * v[i]
        v[i + 1] = v[i] - dt * x[i]
    return x, v


def ho_kdk(dt, n, x0=1.0, v0=0.0):
    """Leapfrog in kick-drift-kick form: positions and velocities at the SAME times."""
    x = np.empty(n + 1); v = np.empty(n + 1); x[0], v[0] = x0, v0
    for i in range(n):
        vh = v[i] - 0.5 * dt * x[i]
        x[i + 1] = x[i] + dt * vh
        v[i + 1] = vh - 0.5 * dt * x[i + 1]
    return x, v


def ho_staggered(dt, n, x0=1.0, v0=0.0):
    """Leapfrog in staggered form. Returns x[0..n], vh[0..n-1] with vh[i] = v(t_{i+1/2}).
    The first velocity is a half kick; after that the two variables simply hop over each other."""
    x = np.empty(n + 1); vh = np.empty(n); x[0] = x0
    vh[0] = v0 - 0.5 * dt * x0
    x[1] = x[0] + dt * vh[0]
    for i in range(1, n):
        vh[i] = vh[i - 1] - dt * x[i]
        x[i + 1] = x[i] + dt * vh[i]
    return x, vh


def ho_rk4(dt, n, x0=1.0, v0=0.0):
    x = np.empty(n + 1); v = np.empty(n + 1); x[0], v[0] = x0, v0
    for i in range(n):
        xi, vi = x[i], v[i]
        k1x, k1v = vi, -xi
        k2x, k2v = vi + 0.5 * dt * k1v, -(xi + 0.5 * dt * k1x)
        k3x, k3v = vi + 0.5 * dt * k2v, -(xi + 0.5 * dt * k2x)
        k4x, k4v = vi + dt * k3v, -(xi + dt * k3x)
        x[i + 1] = xi + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v[i + 1] = vi + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, v


def ho_yoshida(dt, n, x0=1.0, v0=0.0):
    """4th-order Yoshida composition of three leapfrog (KDK) steps: weights w1, w0, w1."""
    c = 2 ** (1 / 3); w1 = 1 / (2 - c); w0 = -c / (2 - c)
    x = np.empty(n + 1); v = np.empty(n + 1); x[0], v[0] = x0, v0
    xi, vi = x0, v0
    for i in range(n):
        for w in (w1, w0, w1):
            h = w * dt
            vh = vi - 0.5 * h * xi
            xi = xi + h * vh
            vi = vh - 0.5 * h * xi
        x[i + 1], v[i + 1] = xi, vi
    return x, v


YOSHIDA_W1 = 1 / (2 - 2 ** (1 / 3))
YOSHIDA_W0 = -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3))


def ho_energy(x, v): return 0.5 * v ** 2 + 0.5 * x ** 2


# ---- the worked table (scene 4): dt = 0.5
TAB_DT = 0.5
TAB_N = 4
TAB_X, TAB_VH = ho_staggered(TAB_DT, TAB_N)           # x_0..x_4, v_{1/2}..v_{7/2}

# ---- midpoint link (scene 3): v(t) = -sin t over one step of size 1, from t = 0 to 1
MID_LO, MID_HI = 0.0, 1.0
MID = dict(
    mid_exact=math.cos(MID_HI) - math.cos(MID_LO),                 # true change in x
    mid_mid=(MID_HI - MID_LO) * -math.sin(0.5 * (MID_LO + MID_HI)),
    mid_left=(MID_HI - MID_LO) * -math.sin(MID_LO),
    mid_right=(MID_HI - MID_LO) * -math.sin(MID_HI),
)
MID['mid_pct'] = 100 * abs(MID['mid_mid'] - MID['mid_exact']) / abs(MID['mid_exact'])
MID['left_pct'] = 100 * abs(MID['mid_left'] - MID['mid_exact']) / abs(MID['mid_exact'])
MID['right_pct'] = 100 * abs(MID['mid_right'] - MID['mid_exact']) / abs(MID['mid_exact'])

# velocity update: a(t) = -cos t, one step of size 1 centred on t_n = 1  (from 1/2 to 3/2)
MID2 = dict(
    acc_exact=-(math.sin(1.5) - math.sin(0.5)),                    # = integral of -cos t from 1/2 to 3/2
    acc_mid=1.0 * -math.cos(1.0),
)
MID2['acc_pct'] = 100 * abs(MID2['acc_mid'] - MID2['acc_exact']) / abs(MID2['acc_exact'])

# ---- order study (scene 5): error of x at T = 10
ORD_T = 10.0
ORD_N = np.array([50, 100, 200, 400, 800, 1600])
ORD_DT = ORD_T / ORD_N
ORD_ERR = {}
for _name, _fn in [('euler', ho_euler), ('leapfrog', ho_kdk), ('rk4', ho_rk4)]:
    ORD_ERR[_name] = np.array([abs(_fn(ORD_T / n, int(n))[0][-1] - math.cos(ORD_T)) for n in ORD_N])


def ord_slope(name):
    """Local log-log slope over the finest step pair (the asymptotic order shows up there)."""
    return float(np.diff(np.log(ORD_ERR[name]))[-1] / np.diff(np.log(ORD_DT))[-1])


# ---- reversibility (scene 6)
REV_DT, REV_N = 0.2, 30


def reversal(method, dt=REV_DT, n=REV_N, x0=1.0, v0=0.3):
    """Run n steps, flip the velocity, run n steps again. Returns forward path, back path, return distance."""
    fn = ho_euler if method == 'euler' else ho_kdk
    xf, vf = fn(dt, n, x0, v0)
    xb, vb = fn(dt, n, xf[-1], -vf[-1])
    dist = math.hypot(xb[-1] - x0, -vb[-1] - v0)
    return (xf, vf), (xb, -vb), dist


# ---- phase-space area (scene 7)
AREA_DT = 0.3
AREA_STEPS = [0, 4, 8, 12]
AREA_C, AREA_H = (1.0, 0.0), 0.3          # a square of side 2*AREA_H centred at AREA_C


def square_boundary(k=60):
    """Boundary of the starting square, k points per edge (corners included once)."""
    s = np.linspace(-1, 1, k, endpoint=False) * AREA_H
    x = np.r_[s, np.full(k, AREA_H), -s, np.full(k, -AREA_H)]
    y = np.r_[np.full(k, -AREA_H), s, np.full(k, AREA_H), -s]
    return AREA_C[0] + x, AREA_C[1] + y


def shoelace(x, y): return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def map_cloud(method, nsteps, dt=AREA_DT):
    x, v = square_boundary()
    for _ in range(nsteps):
        if method == 'euler':
            x, v = x + dt * v, v - dt * x
        else:
            vh = v - 0.5 * dt * x
            x = x + dt * vh
            v = vh - 0.5 * dt * x
    return x, v


def step_matrix(method, dt):
    if method == 'euler': return np.array([[1.0, dt], [-dt, 1.0]])
    return np.array([[1 - dt ** 2 / 2, dt], [-dt * (1 - dt ** 2 / 4), 1 - dt ** 2 / 2]])


# ---- shadow energy (scene 8)
SH_DT = 0.8


def shadow(dt, x, v): return 0.5 * v ** 2 + 0.5 * (1 - dt ** 2 / 4) * x ** 2


# =========================================================================== Kepler
ECC = 0.5
P_ORBIT = TWO_PI
H_LF = 0.0125         # leapfrog and Euler step
H_RK = 0.05           # RK4 step: 4 force evaluations each -> same cost per orbit as H_LF


def kep_init(e=ECC): return 1 + e, 0.0, 0.0, math.sqrt((1 - e) / (1 + e))


def _acc(x, y):
    r3 = (x * x + y * y) ** 1.5
    return -x / r3, -y / r3


def kep_energy(x, y, vx, vy): return 0.5 * (vx * vx + vy * vy) - 1.0 / math.hypot(x, y)


def kep_run(method, h, T, e=ECC, every=1, var=False):
    """Integrate to time T. Returns rows (t, x, y, vx, vy) every `every` steps (plus the last)."""
    x, y, vx, vy = kep_init(e)
    t, out, i = 0.0, [(0.0, x, y, vx, vy)], 0
    ax, ay = _acc(x, y)
    while t < T - 1e-12:
        hh = h * math.hypot(x, y) ** 1.5 if var else h
        if t + hh > T: hh = T - t
        if method == 'euler':
            x, y, vx, vy = x + hh * vx, y + hh * vy, vx + hh * ax, vy + hh * ay
            ax, ay = _acc(x, y)
        elif method == 'leapfrog':
            vx += 0.5 * hh * ax; vy += 0.5 * hh * ay
            x += hh * vx; y += hh * vy
            ax, ay = _acc(x, y)
            vx += 0.5 * hh * ax; vy += 0.5 * hh * ay
        elif method == 'rk4':
            def F(s):
                a1, a2 = _acc(s[0], s[1]); return (s[2], s[3], a1, a2)
            s0 = (x, y, vx, vy)
            k1 = F(s0)
            k2 = F(tuple(s0[j] + 0.5 * hh * k1[j] for j in range(4)))
            k3 = F(tuple(s0[j] + 0.5 * hh * k2[j] for j in range(4)))
            k4 = F(tuple(s0[j] + hh * k3[j] for j in range(4)))
            x, y, vx, vy = (s0[j] + hh / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(4))
        else:
            raise ValueError(method)
        t += hh; i += 1
        if i % every == 0 or t >= T - 1e-12: out.append((t, x, y, vx, vy))
    return np.array(out)


def energy_envelope(method, h, n_orbits, e=ECC, var=False, marks=None):
    """Running max of |E - E0| sampled at each orbit boundary 1..n_orbits (and E - E0 itself)."""
    x, y, vx, vy = kep_init(e)
    E0 = kep_energy(x, y, vx, vy)
    ax, ay = _acc(x, y)
    t, runmax, k = 0.0, 0.0, 1
    env, signed = [], []
    while k <= n_orbits:
        hh = h * math.hypot(x, y) ** 1.5 if var else h
        if method == 'euler':
            x, y, vx, vy = x + hh * vx, y + hh * vy, vx + hh * ax, vy + hh * ay
            ax, ay = _acc(x, y)
        elif method == 'leapfrog':
            vx += 0.5 * hh * ax; vy += 0.5 * hh * ay
            x += hh * vx; y += hh * vy
            ax, ay = _acc(x, y)
            vx += 0.5 * hh * ax; vy += 0.5 * hh * ay
        else:
            def F(s):
                a1, a2 = _acc(s[0], s[1]); return (s[2], s[3], a1, a2)
            s0 = (x, y, vx, vy)
            k1 = F(s0); k2 = F(tuple(s0[j] + 0.5 * hh * k1[j] for j in range(4)))
            k3 = F(tuple(s0[j] + 0.5 * hh * k2[j] for j in range(4)))
            k4 = F(tuple(s0[j] + hh * k3[j] for j in range(4)))
            x, y, vx, vy = (s0[j] + hh / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) for j in range(4))
        t += hh
        dE = kep_energy(x, y, vx, vy) - E0
        if abs(dE) > runmax: runmax = abs(dE)
        if t >= k * P_ORBIT:
            env.append(runmax); signed.append(dE); k += 1
        if not math.isfinite(runmax) or runmax > 1e6: break
    return np.array(env), np.array(signed)


def lrl_angle_rate(rows):
    """Apsidal precession (degrees per orbit) from a linear fit of the Laplace-Runge-Lenz angle."""
    t, x, y, vx, vy = rows.T
    L = x * vy - y * vx
    Ax = vy * L - x / np.hypot(x, y)
    Ay = -vx * L - y / np.hypot(x, y)
    ang = np.unwrap(np.arctan2(Ay, Ax))
    slope_rad_per_t = np.polyfit(t, ang, 1)[0]
    return float(np.degrees(slope_rad_per_t * P_ORBIT))


def kep_sma(rows):
    """Semi-major axis from energy: a = -1/(2E)."""
    t, x, y, vx, vy = rows.T
    E = 0.5 * (vx ** 2 + vy ** 2) - 1 / np.hypot(x, y)
    return -1 / (2 * E)


def kepler_jacobian_defect(method, h=0.05, nsteps=40, e=ECC, eps=1e-6):
    """|| J^T Omega J - Omega ||_F of the n-step map in (x,y,vx,vy), by central finite differences."""
    def flow(s):
        x, y, vx, vy = s
        ax, ay = _acc(x, y)
        for _ in range(nsteps):
            if method == 'euler':
                x, y, vx, vy = x + h * vx, y + h * vy, vx + h * ax, vy + h * ay
                ax, ay = _acc(x, y)
            else:
                vx += 0.5 * h * ax; vy += 0.5 * h * ay
                x += h * vx; y += h * vy
                ax, ay = _acc(x, y)
                vx += 0.5 * h * ax; vy += 0.5 * h * ay
        return np.array([x, y, vx, vy])
    s0 = np.array(kep_init(e))
    J = np.empty((4, 4))
    for j in range(4):
        d = np.zeros(4); d[j] = eps
        J[:, j] = (flow(s0 + d) - flow(s0 - d)) / (2 * eps)
    Om = np.zeros((4, 4)); Om[0, 2] = Om[1, 3] = 1; Om[2, 0] = Om[3, 1] = -1
    return float(np.linalg.norm(J.T @ Om @ J - Om)), float(np.linalg.det(J))


# ---------------------------------------------------------------------------- cached heavy runs
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "cache_kepler.pkl")
_VER = 3


def _compute_kepler():
    d = {'ver': _VER}
    # orbit panels (scene 9): 2 orbits, equal cost
    d['orb_euler'] = kep_run('euler', H_LF, 2 * P_ORBIT, every=4)
    d['orb_rk4'] = kep_run('rk4', H_RK, 2 * P_ORBIT, every=1)
    d['orb_lf'] = kep_run('leapfrog', H_LF, 2 * P_ORBIT, every=4)
    # envelopes (scene 10)
    d['env_euler'], d['sig_euler'] = energy_envelope('euler', H_LF, 30)
    d['env_rk4'], d['sig_rk4'] = energy_envelope('rk4', H_RK, 1000)
    d['env_lf'], d['sig_lf'] = energy_envelope('leapfrog', H_LF, 1000)
    d['env_rk4_3000'], _ = energy_envelope('rk4', H_RK, 3000)
    # limits (scene 11): coarse leapfrog h = 0.1, 30 orbits
    d['coarse'] = kep_run('leapfrog', 0.1, 30 * P_ORBIT, every=1)
    d['coarse_fine'] = kep_run('leapfrog', 0.05, 30 * P_ORBIT, every=1)
    d['coarse_env'], _ = energy_envelope('leapfrog', 0.1, 1000)
    # naive variable step
    d['var_env'], d['var_sig'] = energy_envelope('leapfrog', H_LF, 1000, var=True)
    return d


def kepler_data():
    try:
        with open(_CACHE, 'rb') as fh:
            d = pickle.load(fh)
        if d.get('ver') == _VER: return d
    except Exception:
        pass
    d = _compute_kepler()
    try:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        tmp = _CACHE + f".{os.getpid()}.tmp"
        with open(tmp, 'wb') as fh: pickle.dump(d, fh)
        os.replace(tmp, _CACHE)
    except Exception:
        pass
    return d


KD = kepler_data()


# ---------------------------------------------------------------------------- facts for the narration
def facts():
    F = {}
    # --- Euler on the oscillator (scene 1)
    dt = 0.2
    F['eul_dt'] = dt
    F['eul_fac'] = 1 + dt ** 2                                   # exact energy factor per step
    n1 = int(round(TWO_PI / dt))                                 # steps in one period (31)
    F['eul_nper'] = n1
    xe, ve = ho_euler(dt, 60)
    Ee = ho_energy(xe, ve)
    F['eul_E1p'] = float(Ee[n1] / Ee[0])                         # energy RATIO after one period
    F['eul_E12'] = float(Ee[60] / Ee[0])                         # ... and at t = 12
    F['eul_E1p_r'] = round(F['eul_E1p'], 1)
    F['eul_E12_r'] = round(F['eul_E12'], 1)
    # --- worked table
    F['tab_x4'] = float(TAB_X[4]); F['tab_cos2'] = math.cos(2.0)
    F['tab_x1'] = float(TAB_X[1]); F['tab_vh0'] = float(TAB_VH[0])
    # --- order
    F['sl_euler'] = ord_slope('euler'); F['sl_lf'] = ord_slope('leapfrog'); F['sl_rk4'] = ord_slope('rk4')
    F['lf_err_a'] = float(ORD_ERR['leapfrog'][0]); F['lf_err_b'] = float(ORD_ERR['leapfrog'][-1])
    F['ord_ratio'] = float(ORD_ERR['leapfrog'][2] / ORD_ERR['leapfrog'][3])
    # --- reversibility
    _, _, d_e = reversal('euler'); _, _, d_l = reversal('leapfrog')
    F['rev_euler'] = d_e; F['rev_lf'] = d_l
    F['rev_dt'] = REV_DT; F['rev_n'] = REV_N
    # --- area
    F['area_dt'] = AREA_DT
    a0 = shoelace(*map_cloud('euler', 0))
    F['area_e12'] = shoelace(*map_cloud('euler', 12)) / a0
    F['area_l12'] = shoelace(*map_cloud('leapfrog', 12)) / a0
    F['det_e'] = float(np.linalg.det(step_matrix('euler', AREA_DT)))
    F['det_l'] = float(np.linalg.det(step_matrix('leapfrog', AREA_DT)))
    # --- shadow energy
    xs, vs = ho_kdk(SH_DT, 1000)
    Hs = shadow(SH_DT, xs, vs)
    F['sh_dt'] = SH_DT
    F['sh_coef'] = 1 - SH_DT ** 2 / 4
    F['sh_drift'] = float(np.abs(Hs - Hs[0]).max())
    F['sh_bound'] = SH_DT ** 2 / 8
    Etrue = ho_energy(xs, vs)
    F['sh_Emin'] = float((Etrue - 0.5).min()); F['sh_Emax'] = float((Etrue - 0.5).max())
    # --- Kepler panels (scene 9)
    e = ECC
    F['ecc'] = e
    F['n_lf'] = int(round(P_ORBIT / H_LF)); F['n_rk'] = int(round(P_ORBIT / H_RK))
    F['evals'] = F['n_lf']                                       # force evaluations per orbit, all three
    rows = KD['orb_euler']; F['kep_euler_apo2'] = float(np.max(np.hypot(rows[:, 1], rows[:, 2])))
    rows = KD['orb_lf']; F['kep_lf_apo2'] = float(np.max(np.hypot(rows[:, 1], rows[:, 2])))
    F['kep_lf_a'] = float(kep_sma(rows).mean())
    # --- envelope (scene 10)
    F['env_rk4_1'] = float(KD['env_rk4'][0]); F['env_rk4_1000'] = float(KD['env_rk4'][999])
    F['env_rk4_10'] = float(KD['env_rk4'][9]); F['env_rk4_100'] = float(KD['env_rk4'][99])
    F['env_lf_max'] = float(KD['env_lf'][-1]); F['env_lf_1'] = float(KD['env_lf'][0])
    F['env_eul_30'] = float(KD['env_euler'][-1]); F['env_eul_1'] = float(KD['env_euler'][0])
    # crossover: first orbit where RK4's running max exceeds the leapfrog bound
    cross = int(np.argmax(KD['env_rk4'] > KD['env_lf'][-1])) + 1
    F['cross'] = cross
    F['rk4_growth_per_orbit'] = float(np.polyfit(np.arange(1, 1001), KD['env_rk4'], 1)[0])
    F['rk4_ratio_1000_1'] = F['env_rk4_1000'] / F['env_rk4_1']
    # --- limits (scene 11)
    F['prec_01'] = lrl_angle_rate(KD['coarse'])
    F['prec_005'] = lrl_angle_rate(KD['coarse_fine'])
    F['prec_01_abs'] = abs(F['prec_01'])
    F['prec_005_abs'] = abs(F['prec_005'])
    F['prec_ratio'] = F['prec_01'] / F['prec_005']
    F['coarse_env_max'] = float(KD['coarse_env'][-1])
    F['coarse_sma'] = float(kep_sma(KD['coarse']).mean())
    F['var_per_orbit'] = float(np.polyfit(np.arange(1, 1001), KD['var_sig'], 1)[0])
    F['var_per_orbit_abs'] = abs(F['var_per_orbit'])
    F['var_1000'] = float(KD['var_sig'][-1]); F['var_1000_abs'] = abs(F['var_1000'])
    # --- summary / Yoshida
    errs = np.array([abs(ho_yoshida(ORD_T / n, int(n))[0][-1] - math.cos(ORD_T)) for n in ORD_N[:5]])
    F['yosh_slope'] = float(np.polyfit(np.log(ORD_DT[:5]), np.log(errs), 1)[0])
    F['yosh_w1'] = YOSHIDA_W1; F['yosh_w0'] = YOSHIDA_W0; F['yosh_w0_abs'] = abs(YOSHIDA_W0)
    # --- derived quantities quoted in the voiceover
    F['tab_err'] = abs(F['tab_x4'] - F['tab_cos2'])
    F['eul_pct'] = 100 * F['env_eul_30'] / 0.5                   # |E0| = 1/2 for the a = 1 Kepler orbit
    F['lf_e4'] = F['env_lf_max'] * 1e4
    F['rk_over_lf'] = F['env_rk4_1000'] / F['env_lf_max']
    F['var_e7'] = F['var_per_orbit_abs'] * 1e7
    F.update(MID); F.update(MID2)
    return F


# ---------------------------------------------------------------------------- spoken numbers
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n):
    n = int(n)
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_words(n % 100))
    raise ValueError(n)


def words(x, nd=0):
    """Spell a number the way the voiceover should say it: 2.34 -> 'two point three' (nd=1)."""
    s = f"{abs(x):.{nd}f}"
    ip, _, fp = s.partition('.')
    out = int_words(ip)
    if fp: out += " point " + " ".join(_ONES[int(d)] for d in fp)
    return out


def spoken_facts(F):
    return {
        'w_eul_E1p': words(F['eul_E1p'], 1), 'w_eul_E12': words(F['eul_E12'], 1),
        'w_tab_err': words(F['tab_err'], 2), 'w_rev_euler': words(F['rev_euler'], 1),
        'w_eul_pct': words(F['eul_pct']), 'w_lf_e4': words(F['lf_e4'], 1), 'w_cross': words(F['cross']),
        'w_rk_over_lf': words(F['rk_over_lf']), 'w_prec_01': words(F['prec_01_abs'], 1),
        'w_prec_005': words(F['prec_005_abs'], 2), 'w_var_e7': words(F['var_e7']),
        'w_yosh_w1': words(F['yosh_w1'], 2), 'w_yosh_w0': words(F['yosh_w0_abs'], 2),
    }


FACTS = facts()
FACTS.update(spoken_facts(FACTS))

if __name__ == "__main__":
    for k, v in FACTS.items(): print(f"{k:22s} {v!r}")
