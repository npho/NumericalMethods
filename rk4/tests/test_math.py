"""Numerical checks of every mathematical claim made in the video.
Run with:  python build.py test   (or: pytest -q tests)
"""
import json, math, os
import numpy as np
import pytest

from rk4_video import mathcore as M
from rk4_video.mathcore import FACTS as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ the method
def test_one_step_on_y_equals_y():
    """Scenes 2-4: Euler 2, midpoint 5/2, RK4 65/24, exact e."""
    assert F['one']['euler'] == 2.0 and F['one']['midpoint'] == 2.5
    assert F['one']['rk4'] == pytest.approx(65 / 24)
    assert F['one_err']['rk4'] == pytest.approx(math.e - 65 / 24) and 0.0099 < F['one_err']['rk4'] < 0.0100


def test_four_slopes_and_weights():
    assert F['k'] == pytest.approx([1.0, 1.5, 1.75, 2.75])
    assert (1 + 2 * 1.5 + 2 * 1.75 + 2.75) / 6 == pytest.approx(F['k_avg']) == pytest.approx(41 / 24)
    assert 1 + F['k_avg'] == pytest.approx(F['one']['rk4'])


def test_stage_definitions_match_the_loop():
    """k2 = f(y + h/2 k1), k3 = f(y + h/2 k2), k4 = f(y + h k3), checked by hand for y' = y."""
    k1 = 1.0; k2 = 1 + 0.5 * k1; k3 = 1 + 0.5 * k2; k4 = 1 + 1.0 * k3
    assert [k1, k2, k3, k4] == pytest.approx(F['k'])


def test_midpoint_step_is_the_two_slope_version():
    for h in (1.0, 0.5, 0.1):
        y = M.midpoint(M.f_lin, 0, 1.0, h, 1)[1][-1]
        assert y == pytest.approx(1 + h * (1 + h / 2))


def test_rk4_factor_is_the_taylor_polynomial_of_exp():
    for z in (0.1, 0.5, 1.0):
        y = M.rk4(M.f_lin, 0, 1.0, z, 1)[1][-1]
        assert y == pytest.approx(sum(z ** k / math.factorial(k) for k in range(5)))


def test_rk4_local_error_is_h_to_the_fifth_over_120():
    for z in (0.05, 0.1):
        y = M.rk4(M.f_lin, 0, 1.0, z, 1)[1][-1]
        assert math.exp(z) - y == pytest.approx(z ** 5 / 120, rel=0.05 * z * 10)


def test_rk4_reduces_to_simpsons_rule_when_f_depends_only_on_t():
    """Scene 5: weights 1, 2, 2, 1 over 6 are Simpson's 1, 4, 1 over 6."""
    for f in (np.cos, lambda t: t ** 3 + 2 * t, np.exp, lambda t: 1 / (1 + t * t)):
        for t0, h in ((0.0, 1.0), (0.3, 0.5)):
            assert M.rk4_step_pure_t(f, t0, h) == pytest.approx(M.simpson(f, t0, h), rel=1e-12, abs=1e-14)


def test_simpson_is_exact_for_cubics():
    f = lambda t: 4 * t ** 3 - 3 * t ** 2 + t
    assert M.simpson(f, 0.0, 2.0) == pytest.approx(2 ** 4 - 2 ** 3 + 2)  # integral of f over [0, 2] = 16 - 8 + 2


def test_worked_table():
    assert F['tab_y'] == pytest.approx([1, 1.6484375, 2.71734619140625, 4.47937536, 7.38397032])
    assert 1.6484375 == pytest.approx(1 + 0.5 + 0.125 + 0.5 ** 3 / 6 + 0.5 ** 4 / 24)
    assert F['tab_err'] == pytest.approx(math.exp(2) - 7.38397032, abs=1e-8) and F['tab_err'] < 0.0051
    assert 31 < F['tab_euler_pct'] < 32


# ------------------------------------------------------------------ order of accuracy
def test_orders_on_linear_and_logistic():
    for m, p in (('euler', 1), ('midpoint', 2), ('rk4', 4)):
        assert F['sl_lin_' + m] == pytest.approx(p, abs=0.02 * p + 0.01)
        assert F['sl_log_' + m] == pytest.approx(p, abs=0.02 * p + 0.01)


def test_rk4_error_constant_is_e_over_120():
    assert F['rk4_err_over_h4'] == pytest.approx(math.e / 120, rel=0.01)


def test_closed_forms_match_loops():
    for m in M.METHODS:
        for n in (1, 3, 10, 40):
            assert M.err_lin(m, n) == pytest.approx(M.end_error(m, M.f_lin, 1.0, math.e, n), rel=1e-6, abs=1e-13)


def test_nonsmooth_problem_limits_the_order():
    """Scene 6: y' = sqrt(t) is not smooth at 0, RK4 falls to order 1.5 (it is Simpson's rule on sqrt)."""
    assert F['sl_sqrt_rk4'] == pytest.approx(1.5, abs=0.02)
    assert F['sl_sqrt_midpoint'] == pytest.approx(1.5, abs=0.05)
    assert F['sl_sqrt_euler'] == pytest.approx(1.0, abs=0.05)
    # and on a smooth integrand of the same form, Simpson really is order 4
    e1 = abs(sum(M.simpson(np.cos, i / 8, 1 / 8) for i in range(8)) - math.sin(1))
    e2 = abs(sum(M.simpson(np.cos, i / 16, 1 / 16) for i in range(16)) - math.sin(1))
    assert math.log2(e1 / e2) == pytest.approx(4.0, abs=0.1)


# ------------------------------------------------------------------ cost
def test_equal_budget_of_one_hundred_evaluations():
    b = F['bud']
    assert (b['euler']['n'], b['midpoint']['n'], b['rk4']['n']) == (100, 50, 25)
    assert b['rk4']['lin'] < b['midpoint']['lin'] < b['euler']['lin']
    assert b['rk4']['log'] < b['midpoint']['log'] < b['euler']['log']
    assert b['rk4']['lin'] == pytest.approx(5.6e-8, rel=0.02) and b['midpoint']['lin'] == pytest.approx(1.79e-4, rel=0.02)


def test_evaluations_needed_for_ten_to_minus_ten():
    assert F['ev_rk4'] == 492 and F['ev_midpoint'] == 134618
    assert 13.5e9 < F['ev_euler'] < 13.7e9
    # RK4 and midpoint minima confirmed with the real integrators
    for m, n in (('rk4', F['n_rk4']), ('midpoint', F['n_midpoint'])):
        assert M.end_error(m, M.f_lin, 1.0, math.e, n) < 1e-10 <= M.end_error(m, M.f_lin, 1.0, math.e, n - 1) + 5e-13
    # Euler: closed form cross-checked where the loop is cheap, and against the first-order law 1.36/n
    assert M.err_lin('euler', 10 ** 6) == pytest.approx(math.e / 2e6, rel=1e-3)
    assert M.err_lin('euler', F['n_euler']) < 1e-10 < M.err_lin('euler', F['n_euler'] - 10 ** 6)


def test_rk4_costs_more_per_step_but_wins_at_equal_cost_even_for_a_loose_tolerance():
    """Not a strawman: at a loose tolerance Euler is not hopeless but RK4 is still cheaper."""
    for tol in (1e-2, 1e-4, 1e-6):
        ev = {m: M.evals_to_reach(m, tol)[0] for m in M.METHODS}
        assert ev['rk4'] <= ev['midpoint'] <= ev['euler'] or ev['rk4'] < ev['euler']


# ------------------------------------------------------------------ stability
def test_stability_limits():
    assert F['limit']['euler'] == pytest.approx(2.0, abs=1e-6)
    assert F['limit']['midpoint'] == pytest.approx(2.0, abs=1e-6)
    assert F['limit']['rk4'] == pytest.approx(2.7853, abs=1e-3)
    per = F['limit_per_eval']
    assert per['euler'] == pytest.approx(2.0, abs=1e-6) and per['midpoint'] == pytest.approx(1.0, abs=1e-6)
    assert per['rk4'] == pytest.approx(0.6963, abs=1e-3)


def test_stability_limit_by_actual_integration():
    """Run y' = -y with h just inside and just outside the RK4 limit (lambda = 1)."""
    for h, grows in ((2.7, False), (2.9, True)):
        y = M.rk4(lambda t, y: -y, 0, 1.0, h, 60)[1]
        assert (abs(y[-1]) > 1) == grows


# ------------------------------------------------------------------ the spring, long run
def test_rk4_energy_factor_formula_and_slow_decay():
    t, u = M.rk4_spring(0.5, 50)
    E = M.spr_energy(u)
    assert np.allclose(E[1:] / E[:-1], M.rk4_energy_factor(0.5), rtol=1e-12)
    assert F['spr_rk4_end'] == pytest.approx(F['spr_rk4_pred_end'], rel=1e-9)
    assert 0.55 < F['spr_rk4_end'] < 0.62 and F['spr_rk4_factor'] < 1


def test_leapfrog_energy_stays_bounded_on_the_same_run():
    assert F['spr_lf_max'] <= 1.0 + 1e-9 and F['spr_lf_min'] == pytest.approx(1 - 0.5 ** 2 / 4, abs=1e-6)


def test_simpson_example_and_smaller_step_energy():
    assert F['simp_val'] == pytest.approx(F['simp_rk4'], rel=1e-12)
    assert abs(F['simp_val'] - F['simp_exact']) < 6e-4
    t, u = M.rk4_spring(0.25, int(round(10 * 2 * math.pi / 0.25)))          # 10 periods by real integration
    E = M.spr_energy(u)
    assert E[-1] / E[0] == pytest.approx(M.rk4_energy_factor(0.25) ** (len(u) - 1), rel=1e-9)
    assert 0.97 < F['spr_rk4_h025'] < 0.99


# ------------------------------------------------------------------ the narration only quotes computed numbers
def test_every_narration_placeholder_resolves():
    from rk4_video.narration import load_script
    sc = load_script(os.path.join(ROOT, "narration.json"))
    for s in sc['scenes']:
        for ln in s['lines']:
            assert '{' not in ln['spoken'] and '}' not in ln['spoken'], ln
        names = [ln['key'] for ln in s['lines']]
        assert len(names) == len(set(names)), f"duplicate cue keys in scene {s['name']}"
    assert len(sc['scenes']) == 12


def test_spoken_numbers():
    assert M.words(2.708333, 3) == 'two point seven zero eight'
    assert M.big_words(13591368958) == 'thirteen point six billion'
    assert M.big_words(134618) == 'one hundred thirty five thousand'
    assert M.sci_words(5.6e-8) == 'five point six times ten to the minus eight'
    assert F['w_ev_rk4'] == 'four hundred ninety two' and F['w_spr_pct'] == 'forty one'
