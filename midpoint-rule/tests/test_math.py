"""Numerical checks of every mathematical claim made in the video.
Run with:  python build.py test   (or: pytest -q tests)
"""
import numpy as np
import pytest

from midpoint_video.mathcore import (f, f1, f2, EXACT, Lsum, Rsum, Msum, Tsum, Ssum, NS, ERR, slope,
                                      STRIP_ERR, STRIPS, FACTS, M_endpoint_singularity)


def test_exact_value():
    xs = np.linspace(0, 2, 2_000_001)
    assert EXACT == pytest.approx(np.trapezoid(f(xs), xs), abs=1e-12)


def test_midpoint_exact_for_lines_not_quadratics():
    x = 0.5 * (np.arange(4) + .5)
    assert (0.5 * (3 * x + 1).sum()) == pytest.approx(8.0, abs=1e-12)            # int_0^2 3x+1 = 8
    # quadratic: exact - M = f'' (b-a) h^2 / 24 = 2*2*0.25/24
    assert 8 / 3 - 0.5 * (x ** 2).sum() == pytest.approx(2 / 24 * 2 * 0.25, abs=1e-12)


def test_rectangle_equals_tangent_trapezoid():
    m, h = 1.25, 0.5
    tl = lambda x: f(m) + f1(m) * (x - m)
    assert h * f(m) == pytest.approx(0.5 * (tl(m - h / 2) + tl(m + h / 2)) * h, abs=1e-14)


def test_single_strip_error_formula():
    xs = np.linspace(1.0, 1.5, 200001)
    sliver = np.trapezoid(f(xs) - (f(1.25) + f1(1.25) * (xs - 1.25)), xs)
    assert sliver == pytest.approx(f2(1.25) * 0.5 ** 3 / 24, rel=0.03)            # close, not equal


def test_strip_errors_sum_to_total():
    assert sum(STRIP_ERR) == pytest.approx(EXACT - Msum(4), abs=1e-9)
    assert STRIP_ERR[0] < 0 and all(e > 0 for e in STRIP_ERR[1:])                # 1st strip overshoots


def test_asymptotic_error_constants():
    for n in (256, 1024):
        h = 2 / n; d = f1(2.0) - f1(0.0)
        assert (EXACT - Msum(n)) == pytest.approx(h * h / 24 * d, rel=1e-3)
        assert (EXACT - Tsum(n)) == pytest.approx(-h * h / 12 * d, rel=1e-3)
    assert FACTS["ET_EM"] == pytest.approx(-2.0, abs=1e-3)


def test_convergence_orders():
    assert slope(ERR["L"]) == pytest.approx(-1.0, abs=0.02)
    assert slope(ERR["R"]) == pytest.approx(-1.0, abs=0.02)
    assert slope(ERR["M"]) == pytest.approx(-2.0, abs=0.02)
    assert slope(ERR["T"]) == pytest.approx(-2.0, abs=0.02)
    assert FACTS["ratio_d"] == pytest.approx(4.0, abs=0.05)                       # n -> 2n


def test_simpson_is_two_thirds_mid_plus_one_third_trap():
    assert Ssum(8) == pytest.approx((2 * Msum(4) + Tsum(4)) / 3, abs=1e-14)
    g = lambda x: x ** 3 - 2 * x ** 2 + x                                         # cubic: exact
    G = lambda x: x ** 4 / 4 - 2 * x ** 3 / 3 + x ** 2 / 2
    from midpoint_video import mathcore as mc
    old = mc.f; mc.f = g                                   # swap the integrand temporarily
    try:
        assert mc.Ssum(4) == pytest.approx(G(2) - G(0), abs=1e-12)
    finally:
        mc.f = old


def test_left_right_bracket_for_monotone_curve():
    assert Lsum(4) > EXACT > Rsum(4)


def test_endpoint_singularity_converges_slowly():
    assert 1.98 < M_endpoint_singularity(1000) < 2.0


def test_narration_script_is_consistent():
    from midpoint_video.narration import load_script
    s = load_script("narration.json")                      # raises KeyError if a {placeholder} has no fact
    lines = [ln["spoken"] for sc in s["scenes"] for ln in sc["lines"]]
    assert len(lines) >= 40
    assert not any("minus minus" in t or "minus -" in t for t in lines)
    assert not any("{" in t or "}" in t for t in lines)
