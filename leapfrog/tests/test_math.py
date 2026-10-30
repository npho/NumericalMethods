"""Numerical checks of every mathematical claim made in the video.
Run with:  python build.py test   (or: pytest -q tests)
"""
import json, math, os
import numpy as np
import pytest

from leapfrog_video import mathcore as M
from leapfrog_video.mathcore import FACTS as F, KD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ the method itself
def test_staggered_form_equals_kick_drift_kick():
    """Scene 2: the hopping (staggered) form and the synchronised KDK form give identical positions."""
    for dt in (0.5, 0.2, 0.05):
        n = 40
        xs, vh = M.ho_staggered(dt, n, 1.0, 0.3)
        xk, vk = M.ho_kdk(dt, n, 1.0, 0.3)
        assert np.allclose(xs, xk, atol=1e-13)
        # v at the half step is the average of the synchronised velocities on either side (to O(dt^2))
        assert np.allclose(vh[:-1], 0.5 * (vk[1:n] + vk[:n - 1]), atol=dt ** 2)


def test_worked_example_table():
    """Scene 4: dt = 0.5 hand computation."""
    assert M.TAB_VH[0] == pytest.approx(-0.25)
    assert M.TAB_X[1] == pytest.approx(0.875)
    assert M.TAB_VH[1] == pytest.approx(-0.6875)
    assert M.TAB_X[4] == pytest.approx(-0.435546875)
    assert F['tab_err'] == pytest.approx(abs(-0.435546875 - math.cos(2)))
    assert 0.019 < F['tab_err'] < 0.020


def test_midpoint_rule_link():
    """Scene 3: the middle rectangle really is the best of the three, 4.3% off."""
    exact = math.cos(1) - math.cos(0)
    xs = np.linspace(0, 1, 2_000_001)
    assert np.trapezoid(-np.sin(xs), xs) == pytest.approx(exact, abs=1e-12)
    assert F['mid_pct'] == pytest.approx(4.29, abs=0.01)
    assert F['left_pct'] == pytest.approx(100.0) and 82 < F['right_pct'] < 84
    xs = np.linspace(0.5, 1.5, 2_000_001)
    assert np.trapezoid(-np.cos(xs), xs) == pytest.approx(F['acc_exact'], abs=1e-12)
    assert F['acc_pct'] == pytest.approx(4.29, abs=0.01)


# ------------------------------------------------------------------ Euler is the wrong kind of wrong
def test_euler_energy_factor_is_exactly_one_plus_dt_squared():
    for dt in (0.2, 0.05):
        x, v = M.ho_euler(dt, 50)
        E = M.ho_energy(x, v)
        assert np.allclose(E[1:] / E[:-1], 1 + dt ** 2, rtol=1e-12)
    assert F['eul_fac'] == pytest.approx(1.04)
    assert F['eul_E1p'] == pytest.approx(1.04 ** 31) and F['eul_E1p_r'] == 3.4
    assert F['eul_E12'] == pytest.approx(1.04 ** 60) and F['eul_E12_r'] == 10.5


# ------------------------------------------------------------------ accuracy
def test_orders_of_accuracy():
    assert F['sl_euler'] == pytest.approx(1.0, abs=0.05)
    assert F['sl_lf'] == pytest.approx(2.0, abs=0.01)
    assert F['sl_rk4'] == pytest.approx(4.0, abs=0.05)
    assert F['ord_ratio'] == pytest.approx(4.0, abs=0.05)          # halve dt -> error / 4


def test_rk4_beats_leapfrog_on_a_short_run():
    """Scene 5 'honest' line: on the spring at equal dt, RK4's error is far smaller."""
    assert np.all(M.ORD_ERR['rk4'] < 0.1 * M.ORD_ERR['leapfrog'])


# ------------------------------------------------------------------ reversibility
def test_reversibility():
    assert F['rev_lf'] < 1e-12
    assert F['rev_euler'] > 1.0 and 2.2 < F['rev_euler'] < 2.4        # "2.3 away" in the voiceover
    # Yoshida is a symmetric composition, so it is reversible too; RK4 is not (exactly)
    for fn, tol in ((M.ho_yoshida, 1e-12),):
        xf, vf = fn(0.2, 30, 1.0, 0.3)
        xb, vb = fn(0.2, 30, xf[-1], -vf[-1])
        assert math.hypot(xb[-1] - 1.0, -vb[-1] - 0.3) < tol
    xf, vf = M.ho_rk4(0.5, 30, 1.0, 0.3)
    xb, vb = M.ho_rk4(0.5, 30, xf[-1], -vf[-1])
    assert math.hypot(xb[-1] - 1.0, -vb[-1] - 0.3) > 1e-8


# ------------------------------------------------------------------ area preservation
def test_step_matrix_determinants():
    assert np.linalg.det(M.step_matrix('leapfrog', 0.3)) == pytest.approx(1.0, abs=1e-14)
    assert np.linalg.det(M.step_matrix('euler', 0.3)) == pytest.approx(1.09)
    # the matrix form matches the integrators
    for method, fn in (('euler', M.ho_euler), ('leapfrog', M.ho_kdk)):
        x, v = fn(0.3, 5, 1.0, 0.0)
        z = np.array([1.0, 0.0])
        for _ in range(5): z = M.step_matrix(method, 0.3) @ z
        assert np.allclose(z, [x[-1], v[-1]], atol=1e-13)


def test_cloud_area():
    assert F['area_l12'] == pytest.approx(1.0, abs=1e-12)
    assert F['area_e12'] == pytest.approx(1.09 ** 12, rel=1e-10)
    assert round(F['area_e12'], 2) == 2.81


# ------------------------------------------------------------------ shadow energy
def test_shadow_energy_is_conserved_exactly():
    for dt in (0.8, 0.3):
        x, v = M.ho_kdk(dt, 1000)
        H = M.shadow(dt, x, v)
        assert np.abs(H - H[0]).max() < 1e-12
    assert F['sh_drift'] < 1e-12
    # the points of scene 8 lie on the ellipse x = cos, v = sqrt(1 - dt^2/4) sin  (to rounding)
    x, v = M.ho_kdk(0.8, 40)
    assert np.allclose(v ** 2 / (1 - 0.8 ** 2 / 4) + x ** 2, 1.0, atol=1e-12)


def test_true_energy_error_bound():
    """|E - E0| <= dt^2/8 for this orbit, and the bound is attained."""
    dt = 0.8
    x, v = M.ho_kdk(dt, 5000)
    err = M.ho_energy(x, v) - 0.5
    assert err.max() <= 1e-12 and err.min() >= -dt ** 2 / 8 - 1e-12
    assert err.min() < -0.99 * dt ** 2 / 8
    # exact identity E - E0 = dt^2/8 (x^2 - 1)
    assert np.allclose(err, dt ** 2 / 8 * (x ** 2 - 1), atol=1e-12)


# ------------------------------------------------------------------ Kepler
def test_kepler_setup():
    x, y, vx, vy = M.kep_init()
    assert M.kep_energy(x, y, vx, vy) == pytest.approx(-0.5)         # a = 1  =>  E = -1/(2a)
    assert F['n_lf'] == 503 and F['n_rk'] == 126
    assert abs(F['n_rk'] * 4 - F['n_lf']) <= 2                      # equal cost: 4 evaluations per RK4 step


def test_kepler_symplectic_jacobian():
    d_lf, det_lf = M.kepler_jacobian_defect('leapfrog')
    d_eu, det_eu = M.kepler_jacobian_defect('euler')
    assert d_lf < 1e-7 and det_lf == pytest.approx(1.0, abs=1e-7)
    assert d_eu > 1e-3 and abs(det_eu - 1.0) > 1e-3


def test_kepler_euler_spirals_out():
    assert F['kep_euler_apo2'] > 2.1                                 # was 1.50
    assert F['kep_lf_apo2'] == pytest.approx(1.5, abs=1e-3)
    assert 80 < F['eul_pct'] < 95                                   # "87 percent" at orbit 30


def test_long_run_energy_envelope():
    env_lf, env_rk, env_eu = KD['env_lf'], KD['env_rk4'], KD['env_euler']
    # leapfrog: bounded (envelope stops growing after the first orbit); about 2e-4
    assert env_lf[-1] / env_lf[0] < 1.01
    assert 1.5e-4 < env_lf[-1] < 2.5e-4 and F['lf_e4'] == pytest.approx(2.1, abs=0.1)
    # RK4: starts much lower, then grows linearly with the number of orbits
    assert env_rk[0] < 0.1 * env_lf[0]
    assert F['rk4_ratio_1000_1'] == pytest.approx(1000, rel=0.05)
    assert F['cross'] == 45 and 22 < F['rk_over_lf'] < 24
    # Euler: error is a large fraction of |E| within 30 orbits
    assert env_eu[-1] > 0.4


def test_equal_cost_comparison_is_not_a_strawman():
    """RK4 at 4x the leapfrog step is stable and accurate over the 2 orbits shown."""
    rows = KD['orb_rk4']
    assert np.max(np.hypot(rows[:, 1], rows[:, 2])) == pytest.approx(1.5, abs=1e-3)


def test_coarse_step_precession_scales_like_h_squared():
    assert F['prec_01'] < 0 and F['prec_005'] < 0
    assert 3.5 < F['prec_ratio'] < 4.5                              # halving h -> about 1/4
    assert F['prec_01_abs'] == pytest.approx(2.2, abs=0.05)
    assert F['coarse_env_max'] < 0.02                                # energy still bounded
    assert F['coarse_sma'] == pytest.approx(1.0, abs=5e-3)


def test_naive_variable_step_drifts_but_fixed_step_does_not():
    env_v = KD['var_env']
    assert env_v[-1] > 5 * env_v[0]                                  # keeps growing
    assert F['var_per_orbit'] < 0 and 3.5 < F['var_e7'] < 4.5
    assert KD['env_lf'][-1] / KD['env_lf'][0] < 1.01                 # fixed step: bounded


# ------------------------------------------------------------------ Yoshida
def test_yoshida_is_fourth_order_symmetric_and_symplectic():
    assert F['yosh_slope'] == pytest.approx(4.0, abs=0.05)
    assert 2 * M.YOSHIDA_W1 + M.YOSHIDA_W0 == pytest.approx(1.0)      # weights sum to one step
    assert round(M.YOSHIDA_W1, 2) == 1.35 and round(abs(M.YOSHIDA_W0), 2) == 1.70 and M.YOSHIDA_W0 < 0
    # symplectic: area-preserving map for the spring (composition of unit-determinant maps)
    z = np.array([[1, 0], [0, 1]], float)
    for w in (M.YOSHIDA_W1, M.YOSHIDA_W0, M.YOSHIDA_W1): z = M.step_matrix('leapfrog', w * 0.4) @ z
    assert np.linalg.det(z) == pytest.approx(1.0, abs=1e-14)


def test_yoshida_energy_bounded_on_kepler():
    """Table entry 'bounded': the running max error does not grow between orbit 20 and orbit 200."""
    c = 2 ** (1 / 3); w = (1 / (2 - c), -c / (2 - c), 1 / (2 - c))
    x, y, vx, vy = M.kep_init(); E0 = M.kep_energy(x, y, vx, vy)
    h = 0.05; t = 0.0; runmax = 0.0; marks = {}
    k = 1
    while k <= 200:
        for wi in w:
            hh = wi * h
            ax, ay = M._acc(x, y); vx += 0.5 * hh * ax; vy += 0.5 * hh * ay
            x += hh * vx; y += hh * vy
            ax, ay = M._acc(x, y); vx += 0.5 * hh * ax; vy += 0.5 * hh * ay
        t += h
        runmax = max(runmax, abs(M.kep_energy(x, y, vx, vy) - E0))
        if t >= k * M.P_ORBIT: marks[k] = runmax; k += 1
    assert marks[200] / marks[20] < 1.5
    assert marks[200] < 1e-3


# ------------------------------------------------------------------ the narration only quotes computed numbers
def test_every_narration_placeholder_resolves():
    from leapfrog_video.narration import load_script
    sc = load_script(os.path.join(ROOT, "narration.json"))
    for s in sc["scenes"]:
        for ln in s["lines"]:
            assert "{" not in ln["spoken"] and "}" not in ln["spoken"]
        names = [ln["key"] for ln in s["lines"]]
        assert len(names) == len(set(names)), f"duplicate cue keys in scene {s['name']}"
    assert len(sc["scenes"]) == 13


def test_numbers_in_voiceover_match_facts():
    sc = json.load(open(os.path.join(ROOT, "narration.json"), encoding="utf-8"))
    text = " ".join(ln["text"] for s in sc["scenes"] for ln in s["lines"])
    assert "one point oh four" in text and F['eul_fac'] == pytest.approx(1.04)
    assert "point eight seven five" in text and M.TAB_X[1] == pytest.approx(0.875)
    assert "minus point four three five five" in text and round(M.TAB_X[4], 4) == -0.4355
    assert "minus point four one six one" in text and round(math.cos(2), 4) == -0.4161
    assert "four percent" in text and 4 < F['mid_pct'] < 5
    assert "thirty Oiler steps" in text and M.REV_N == 30
    assert "one point oh nine" in text and F['det_e'] == pytest.approx(1.09)
    assert "delta t equal to point eight" in text and M.SH_DT == 0.8
    assert "one minus delta t squared over four" in text
    assert "delta t squared over eight" in text
    assert "eccentricity one half" in text and M.ECC == 0.5
    assert "about five hundred force evaluations" in text and abs(F['evals'] - 500) < 10
    assert "a thousand orbits" in text
    assert "step of point one" in text


def test_spoken_numbers():
    assert M.words(2.34, 1) == "two point three" and M.words(0.02, 2) == "zero point zero two"
    assert M.words(87.1) == "eighty seven" and M.words(45) == "forty five" and M.words(10.52, 1) == "ten point five"
    assert M.words(1.7024, 2) == "one point seven zero" and M.words(23) == "twenty three"
    # the spoken variants agree with the numbers shown on screen
    assert F['w_eul_E1p'] == "three point four" and F['w_eul_E12'] == "ten point five"
    assert F['w_prec_01'] == "two point two" and F['w_prec_005'] == "zero point five six"
    assert F['w_yosh_w1'] == "one point three five" and F['w_yosh_w0'] == "one point seven zero"
    assert F['w_cross'] == "forty five" and F['w_rk_over_lf'] == "twenty three" and F['w_var_e7'] == "four"
