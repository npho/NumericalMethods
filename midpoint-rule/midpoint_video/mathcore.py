"""Pure-numpy math for the video. No plotting, no audio.

Everything shown on screen or spoken in the narration is computed here, so the
video cannot drift out of sync with the mathematics.
"""
import math
import numpy as np

# The integrand, its derivatives, and the interval [A_, B_].
f = lambda x: np.exp(-np.asarray(x, dtype=float) ** 2)
f1 = lambda x: -2 * np.asarray(x, dtype=float) * np.exp(-np.asarray(x, dtype=float) ** 2)
f2 = lambda x: (4 * np.asarray(x, dtype=float) ** 2 - 2) * np.exp(-np.asarray(x, dtype=float) ** 2)
A_, B_ = 0.0, 2.0
EXACT = 0.5 * math.sqrt(math.pi) * math.erf(2.0)      # = (sqrt(pi)/2) erf(2)


def Lsum(n): h = (B_ - A_) / n; return h * f(A_ + h * np.arange(n)).sum()
def Rsum(n): h = (B_ - A_) / n; return h * f(A_ + h * np.arange(1, n + 1)).sum()
def Msum(n): h = (B_ - A_) / n; return h * f(A_ + h * (np.arange(n) + .5)).sum()
def Tsum(n): return 0.5 * (Lsum(n) + Rsum(n))
def Ssum(n):                                            # composite Simpson, n even
    h = (B_ - A_) / n; x = A_ + h * np.arange(n + 1); y = f(x)
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


# The four strips used in scenes 3-5 and the exact (true - midpoint) error on each.
STRIPS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
STRIP_ERR = []
for _lo, _hi in STRIPS:
    _x = np.linspace(_lo, _hi, 200001)
    STRIP_ERR.append(float(np.trapezoid(f(_x), _x) - (_hi - _lo) * f((_lo + _hi) / 2)))

# Convergence study (scene 6).
NS = 2 ** np.arange(1, 12)                              # 2 .. 2048
ERR = {k: np.array([abs(EXACT - fn(int(n))) for n in NS]) for k, fn in
       [('L', Lsum), ('R', Rsum), ('M', Msum), ('T', Tsum)]}


def slope(e, lo=3):
    """Fitted log-log slope of an error curve over n = NS[lo:]."""
    return float(np.polyfit(np.log(NS[lo:]), np.log(e[lo:]), 1)[0])


def M_endpoint_singularity(n=1000):
    """Midpoint rule on integral_0^1 x^(-1/2) dx = 2 (never touches x = 0)."""
    x = (np.arange(n) + .5) / n
    return float((1.0 / n) * (1.0 / np.sqrt(x)).sum())


def facts():
    """Numbers quoted in narration.json via {placeholders}."""
    n = 4
    L4, R4, M4, T4 = Lsum(n), Rsum(n), Msum(n), Tsum(n)
    errL, errR, errM = abs(EXACT - L4), abs(EXACT - R4), abs(EXACT - M4)
    ratio = 0.5 * (errL + errR) / errM
    fm = f(0.25 + 0.5 * np.arange(4))
    m, hh = 1.25, 0.5
    xs = np.linspace(1.0, 1.5, 200001)
    sliver = float(np.trapezoid(f(xs) - (f(m) + f1(m) * (xs - m)), xs))
    pred_strip = float(f2(m)) * hh ** 3 / 24
    i16 = int(np.where(NS == 16)[0][0])
    return dict(
        L4=L4, R4=R4, M4=M4, T4=T4, exact=EXACT, errL=errL, errR=errR, errM=errM,
        ratio=ratio, ratio10=int(round(ratio, -1)),
        fm1=float(fm[0]), fm2=float(fm[1]), fm3=float(fm[2]), fm4=float(fm[3]), S=float(fm.sum()),
        sliver=sliver, pred_strip=pred_strip,
        net=EXACT - M4, net_abs=abs(EXACT - M4),
        pred_net=0.5 ** 2 / 24 * float(f1(2) - f1(0)),
        pred_net_abs=abs(0.5 ** 2 / 24 * float(f1(2) - f1(0))),
        ratio_d=float(ERR['M'][i16] / ERR['M'][i16 + 1]),
        ET_EM=(EXACT - Tsum(1024)) / (EXACT - Msum(1024)),
        M1000=M_endpoint_singularity(1000),
    )


FACTS = facts()
