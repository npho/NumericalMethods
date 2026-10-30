"""Numerical checks of every mathematical claim made in the video.
Run with:  python build.py test   (or: pytest -q tests)
"""
import json, math, os
import numpy as np
import pytest

from euler_video import mathcore as M
from euler_video.mathcore import FACTS as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ the method and the closed forms
def test_closed_forms_match_the_loop_integrators():
    for name, integ in (('euler', M.euler), ('midpoint', M.midpoint), ('rk4', M.rk4)):
        for n in (1, 2, 5, 17, 100):
            y = integ(M.f_lin, 0, 1, 1.0 / n, n)[1][-1]
            assert M.err_lin(name, n) == pytest.approx(abs(math.e - y), rel=1e-6, abs=1e-13)


def test_euler_on_y_equals_y_is_one_plus_one_over_n_to_the_n():
    for n in (1, 2, 4, 10, 1000):
        assert M.euler(M.f_lin, 0, 1, 1.0 / n, n)[1][-1] == pytest.approx((1 + 1 / n) ** n, rel=1e-12)
    assert abs((1 + 1 / 10 ** 7) ** 10 ** 7 - math.e) < 1e-6


def test_one_step_and_table():
    """Scenes 2-3."""
    assert F['one_y1'] == 2.0 and F['one_err'] == pytest.approx(math.e - 2)
    assert list(F['tab_y']) == pytest.approx([1, 1.5, 2.25, 3.375, 5.0625])
    assert F['tab_exact'] == pytest.approx(math.exp(2))
    assert 31.0 < F['tab_pct'] < 32.0                      # "short by about thirty one percent"


def test_euler_path_undershoots_a_convex_solution():
    """Scene 3: the tangent-line path always lies below e^t, so errors accumulate one way."""
    for h in (1.0, 0.5, 0.1, 0.01):
        n = int(round(2 / h))
        t, y = M.euler(M.f_lin, 0, 1, h, n)
        assert np.all(y[1:] < np.exp(t[1:]))


def test_values_for_the_small_multiples():
    assert F['h_val'] == pytest.approx([2.0, 2.25, 2.44140625, 2.5937424601])
    assert F['lim_val'] == pytest.approx(2.71692393, abs=1e-7)


# ------------------------------------------------------------------ order of accuracy
def test_euler_is_first_order_on_linear_and_logistic():
    assert F['sl_lin'] == pytest.approx(1.0, abs=0.01)
    assert F['sl_log'] == pytest.approx(1.0, abs=0.01)


def test_error_constant_is_e_over_two():
    """err(h) ~ (e/2) h, as derived from (1+h)^(1/h) = e (1 - h/2 + ...)."""
    assert F['err_over_h'] == pytest.approx(math.e / 2, abs=0.005)


def test_local_error_matches_half_y_double_prime_h_squared():
    assert F['loc_err'] == pytest.approx(F['loc_pred'], rel=0.05)


def test_steps_needed_for_one_thousandth():
    n = F['n_1e3']
    assert M.err_lin('euler', n) < 1e-3 <= M.err_lin('euler', n - 1)
    h = M.euler(M.f_lin, 0, 1, 1.0 / n, n)[1][-1]            # confirmed with the real loop
    assert abs(math.e - h) < 1e-3


def test_midpoint_and_rk4_orders():
    for name, order in (('midpoint', 2), ('rk4', 4)):
        n = np.array([8, 16, 32, 64])
        e = np.array([M.err_lin(name, int(k)) for k in n])
        s = np.diff(np.log(e)) / np.diff(np.log(1.0 / n))
        assert s[-1] == pytest.approx(order, abs=0.05)
    assert F['wsl_euler'] == pytest.approx(-1, abs=0.01)
    assert F['wsl_midpoint'] == pytest.approx(-2, abs=0.01)
    assert F['wsl_rk4'] == pytest.approx(-4, abs=0.06)


# ------------------------------------------------------------------ stability
def test_decay_factors_and_endpoints():
    assert F['stab_fac'] == pytest.approx([0.5, -0.5, -1.5])
    t, y = M.STAB[0.1]
    assert np.allclose(y[1:] / y[:-1], 0.5)
    t, y = M.STAB[0.3]
    assert np.allclose(y[1:] / y[:-1], -0.5)
    t, y = M.STAB[0.5]
    assert np.allclose(y[1:] / y[:-1], -1.5)
    assert abs(y[-1]) == pytest.approx(1.5 ** 6)               # 11.39 at t = 3
    assert abs(y[-1]) > 11 and abs(y[-1]) > abs(M.STAB[0.5][1][0])


def test_stability_threshold_is_two_over_lambda():
    assert F['stab_limit'] == pytest.approx(0.4)
    for h, grows in ((0.39, False), (0.401, True), (0.41, True), (0.2, False)):
        n = int(round(40 / h))
        y = M.euler(M.f_dec, 0, 1, h, n)[1]
        assert (abs(y[-1]) > 1) == grows, h


def test_sign_flip_between_quarter_and_half_of_limit():
    """Oscillating-but-decaying regime: 1/lambda < h < 2/lambda."""
    y = M.euler(M.f_dec, 0, 1, 0.3, 10)[1]
    assert np.all(np.sign(y[1:]) != np.sign(y[:-1])) and abs(y[-1]) < abs(y[0])


def test_backward_euler_is_below_one_for_every_step():
    z = np.logspace(-3, 4, 500)
    assert np.all(M.backward_factor(z) < 1)
    # and matches an actual implicit step y1 = y0 + h*(-lam*y1)
    h, lam, y0 = 0.5, 5.0, 1.0
    y1 = y0 / (1 + lam * h)
    assert y1 == pytest.approx(y0 + h * (-lam * y1))
    assert F['be_fac_05'] == pytest.approx(1 / 3.5)


def test_backward_euler_is_first_order():
    """Scene 7 'still only first order': solve y' = -y implicitly, T = 1."""
    errs = []
    for n in (100, 200):
        h = 1.0 / n; y = (1 / (1 + h)) ** n
        errs.append(abs(y - math.exp(-1)))
    assert math.log2(errs[0] / errs[1]) == pytest.approx(1.0, abs=0.02)


def test_stability_limits_of_other_methods():
    assert F['lim_mid'] == pytest.approx(2.0, abs=1e-6)
    assert F['lim_rk4'] == pytest.approx(2.7853, abs=1e-3)


# ------------------------------------------------------------------ spring
def test_euler_energy_factor_is_one_plus_h_squared_every_step():
    x, v = M.spring_euler()
    E = 0.5 * (x ** 2 + v ** 2)
    assert np.allclose(E[1:] / E[:-1], 1 + M.SPR_H ** 2)
    assert F['spr_fac'] == pytest.approx(1.04)
    assert F['spr_E1p'] == pytest.approx(1.04 ** 31) and 3.3 < F['spr_E1p'] < 3.5
    assert E[60] / E[0] == pytest.approx(1.04 ** 60) and 10 < E[60] / E[0] < 11


# ------------------------------------------------------------------ cost comparison (not a strawman)
def test_evaluation_counts_for_one_millionth():
    for m, ev in (('euler', 1359140), ('midpoint', 1346), ('rk4', 52)):
        assert F['ev_' + m] == ev
    # each is the true minimum, checked with the real integrators
    for m, integ in (('midpoint', M.midpoint), ('rk4', M.rk4)):
        n = F['n_' + m]
        assert abs(math.e - integ(M.f_lin, 0, 1, 1.0 / n, n)[1][-1]) < 1e-6
        assert abs(math.e - integ(M.f_lin, 0, 1, 1.0 / (n - 1), n - 1)[1][-1]) >= 1e-6
    n = F['n_euler']
    assert abs(math.e - M.euler(M.f_lin, 0, 1, 1.0 / n, n)[1][-1]) < 1e-6
    assert abs(math.e - M.euler(M.f_lin, 0, 1, 1.0 / (n - 1), n - 1)[1][-1]) >= 1e-6


def test_evaluation_counts_per_step():
    assert M.EVALS == {'euler': 1, 'midpoint': 2, 'rk4': 4}


# ------------------------------------------------------------------ the narration only quotes computed numbers
def test_every_narration_placeholder_resolves():
    from euler_video.narration import load_script
    sc = load_script(os.path.join(ROOT, "narration.json"))
    for s in sc['scenes']:
        for ln in s['lines']:
            assert '{' not in ln['spoken'] and '}' not in ln['spoken'], ln
        names = [ln['key'] for ln in s['lines']]
        assert len(names) == len(set(names)), f"duplicate cue keys in scene {s['name']}"
    assert len(sc['scenes']) == 11


def test_spoken_numbers():
    assert M.words(2.718281828, 3) == 'two point seven one eight'
    assert M.int_words(1346) == 'one thousand three hundred forty six'
    assert M.big_words(1359140) == 'one point four million'
    assert M.big_words(1346) == 'thirteen hundred'
    assert M.big_words(52) == 'fifty two'
    assert F['w_n_1e3'] == 'one thousand three hundred sixty'
    assert F['w_tab_pct'] == 'thirty one' and F['w_hv_01'] == 'two point five nine'


def test_numbers_in_voiceover_match_facts():
    """Spot-check the plain-word numbers that are not placeholders."""
    sc = json.load(open(os.path.join(ROOT, "narration.json"), encoding="utf-8"))
    txt = {(i, l['key']): l['text'] for i, s in enumerate(sc['scenes']) for l in s['lines']}
    flat = ' '.join(txt.values())
    for phrase in ("minus one and a half", "point four", "one point oh four", "two point two five",
                   "one and a half", "factor is minus one half"):
        assert phrase in flat, phrase
    # the arithmetic behind them:
    assert 1 - 5 * 0.5 == -1.5 and 1 - 5 * 0.3 == pytest.approx(-0.5) and 1.5 ** 1 == 1.5
    assert 1 + 0.2 ** 2 == pytest.approx(1.04)
