"""Scene drawing: a 3Blue1Brown x Tufte hybrid.

3Blue1Brown: dark canvas, saturated accent colours, geometry that builds up.
Tufte: serif type, range-frame axes, direct labels (no legends), margin
sidenotes, small multiples, hairline tables, high data-ink ratio.

Every scene is a pure function of local time `t` (seconds).  Event times come
from the narration timeline via `Q(scene, key, frac, off)` = the moment that is
`frac` of the way through spoken line `key`, plus `off` seconds.
"""
import glob, math, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from .mathcore import (f, f1, f2, A_, B_, EXACT, Lsum, Rsum, Msum, Tsum, STRIPS, STRIP_ERR,
                       NS, ERR, slope, M_endpoint_singularity)

# ------------------------------------------------------------------ look
BG = '#12141a'; FG = '#ece9e1'; MUTED = '#8b909a'; SOFT = '#b9bdc6'
BLUE = '#58C4DD'; YELLOW = '#FFD35A'; CORAL = '#FC6255'; GREEN = '#83C167'
FONT = 'TeX Gyre Pagella'              # Palatino clone, bundled in the repo's fonts/ (GUST Font License)
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'fonts')
FPS = 30
VIS_LEAD = 0.15                         # visuals appear this much before the voice

W, H, DPI, fig = 1280, 720, 100.0, None


def configure(width=1280):
    """Set output resolution (16:9). Figure size is fixed in inches; DPI scales everything."""
    global W, H, DPI, fig
    for path in glob.glob(os.path.join(FONT_DIR, '*.otf')):
        font_manager.fontManager.addfont(path)
    try:
        font_manager.findfont(FONT, fallback_to_default=False)
    except ValueError as e:
        raise SystemExit(f"Font '{FONT}' not found in {FONT_DIR} or on the system.") from e
    plt.rcParams.update({
        'font.family': FONT,
        'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic',
        'mathtext.bf': FONT + ':bold', 'mathtext.fallback': 'stix', 'axes.unicode_minus': True,
    })
    W, H = int(width), int(width) * 9 // 16
    DPI = W / 12.8
    if fig is not None: plt.close(fig)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=DPI)


# ------------------------------------------------------------------ timeline access
TL = None


def set_timeline(tl):
    global TL
    TL = tl


def Q(scene, key, frac=0.0, off=0.0):
    c = TL['scenes'][scene]['cues'][key]
    t = c['start'] + frac * c['dur'] + off
    if frac == 0.0 and off == 0.0: t -= VIS_LEAD
    return max(0.05, t)


def Qd(scene, key): return TL['scenes'][scene]['cues'][key]['dur']


# ------------------------------------------------------------------ helpers
def sgn(x, nd=4, unit=1.0):
    return ("+" if x >= 0 else "−") + f"{abs(x) / unit:.{nd}f}"


def neg(s): return s.replace('-', '−')
def clamp(x): return min(1.0, max(0.0, x))
def ease(x): x = clamp(x); return x * x * (3 - 2 * x)
def A(t, t0, d): return ease((t - t0) / d)


def new_frame():
    fig.clf(); fig.patch.set_facecolor(BG)


def T(x, y, s, size=15, color=FG, alpha=1.0, ha='left', va='baseline', style='normal', weight='normal', **kw):
    if alpha <= 0.004: return None
    return fig.text(x, y, s, fontsize=size, color=color, alpha=min(1, alpha), ha=ha, va=va,
                    style=style, fontweight=weight, **kw)


def TA(ax, x, y, s, size=15, color=FG, alpha=1.0, ha='left', va='baseline', style='normal', **kw):
    if alpha <= 0.004: return None
    return ax.text(x, y, s, fontsize=size, color=color, alpha=min(1, alpha), ha=ha, va=va, style=style,
                   clip_on=False, **kw)


def hline(y, x0=0.06, x1=0.94, alpha=0.55, lw=0.6, color=MUTED):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, lw=lw, alpha=alpha))


def header(num, title):
    T(0.06, 0.915, str(num), size=22, color=YELLOW)
    T(0.092, 0.915, title, size=22)
    hline(0.893)


def wrap(s, width):
    """Word-wrap that never breaks inside $math$."""
    toks = re.findall(r'\$[^$]*\$[^\s$]*|[^\s$]+', s)
    lines, cur, curlen = [], [], 0
    for tk in toks:
        L = len(tk) - (2 if '$' in tk else 0)
        if cur and curlen + 1 + L > width:
            lines.append(' '.join(cur)); cur, curlen = [tk], L
        else:
            cur.append(tk); curlen += (1 if cur[:-1] else 0) + L
    if cur: lines.append(' '.join(cur))
    return '\n'.join(lines)


def margin(t, items, top=0.835, x=0.715, size=15, width=29, gap=0.030):
    """Tufte-style sidenotes. Layout is fixed in advance so nothing jumps."""
    y = top
    pitch = size * 100 / 72 * 1.42 / 720
    for it in items:
        kind = it.get('kind', 'p')
        a = A(t, it['t'], 0.9) * (1 - A(t, it.get('out', 1e9), 0.6))
        if kind == 'eq':
            hgt = it.get('h', 0.075)
            T(x, y - hgt * 0.62, it['text'], size=it.get('size', 19), alpha=a, color=it.get('color', FG))
            y -= hgt + gap * 0.6
        else:
            s = wrap(it['text'], width)
            n = s.count('\n') + 1
            T(x, y, s, size=size, alpha=a * (0.94 if kind == 'p' else 0.9), va='top',
              color=it.get('color', FG if kind == 'p' else SOFT), style='normal' if kind == 'p' else 'italic',
              linespacing=1.3)
            y -= n * pitch + gap


def style_ax(ax, xr, yr, xticks, yticks, xlabels=None, ylabels=None, size=13):
    """Tufte 'range frame': axes only span the data, offset outward, no box, no grid."""
    ax.set_facecolor('none')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_position(('outward', 6))
    ax.set_xlim(*xr); ax.set_ylim(*yr)
    ax.set_xticks(xticks); ax.set_yticks(yticks)
    if xlabels is not None: ax.set_xticklabels(xlabels)
    if ylabels is not None: ax.set_yticklabels(ylabels)
    ax.spines['bottom'].set_bounds(min(xticks), max(xticks))
    ax.spines['left'].set_bounds(min(yticks), max(yticks))
    ax.tick_params(colors=MUTED, labelsize=size, length=3.5, width=0.8, pad=4)
    ax.minorticks_off()


AX_MAIN = [0.075, 0.15, 0.60, 0.68]
XS = np.linspace(0, 2, 500)


def main_axes():
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, (0, 2.05), (0, 1.08), [0, .5, 1, 1.5, 2], [0, .5, 1],
             ['0', '0.5', '1', '1.5', '2'], ['0', '0.5', '1'])
    return ax


def rects(ax, heights, growth=lambda i: 1.0, alpha=0.30, ec=BLUE, fc=BLUE, lw=1.3, ecalpha=1.0):
    n = len(heights); h = (B_ - A_) / n
    for i in range(n):
        g = growth(i)
        if g <= 0.001: continue
        ax.add_patch(Rectangle((A_ + i * h, 0), h, heights[i] * g, facecolor=to_rgba(fc, alpha),
                               edgecolor=to_rgba(ec, ecalpha), lw=lw, zorder=2))


# ================================================================== SCENES
def scene0(t):
    q = lambda k, fr=0.0, off=0.0: Q(0, k, fr, off)
    t0 = q('title', 0, 0.15)
    T(0.07, 0.585, "The Midpoint Rule", size=44, alpha=A(t, t0, 1.2))
    T(0.07, 0.505, "Estimating an area when the\nantiderivative won't help.", size=21, color=SOFT,
      style='italic', alpha=A(t, t0 + 1.0, 1.2), va='top', linespacing=1.3)
    hline(0.685, 0.07, 0.46, alpha=0.7 * A(t, t0 + 0.6, 1.0))
    T(0.07, 0.24, "Numerical integration · part 1", size=14, color=MUTED, alpha=A(t, t0 + 2.2, 1.0))
    ax = fig.add_axes([0.54, 0.20, 0.40, 0.52])
    style_ax(ax, (0, 2.05), (0, 1.08), [0, 2], [0], ['0', '2'], ['0'])
    ax.tick_params(axis='y', length=0)
    p = A(t, t0 + 1.4, 3.0)
    k = max(2, int(len(XS) * p))
    ax.fill_between(XS[:k], f(XS[:k]), color=BLUE, alpha=0.18 * A(t, t0 + 3.0, 1.5), lw=0)
    ax.plot(XS[:k], f(XS[:k]), color=FG, lw=2.6)
    hts = f(0.25 + 0.5 * np.arange(4))
    r0 = q('rects', 0.25)
    rects(ax, hts, growth=lambda i: A(t, r0 + 0.35 * i, 0.9), alpha=0.0, lw=1.2, ecalpha=0.9)


def scene1(t):
    q = lambda k, fr=0.0, off=0.0: Q(1, k, fr, off)
    header(1, "The problem")
    ax = main_axes()
    ax.fill_between(XS, f(XS), color=BLUE, alpha=0.20, lw=0)
    ax.plot(XS, f(XS), color=FG, lw=2.6, zorder=3)
    TA(ax, 0.84, 0.80, r'$f(x)=e^{-x^2}$', size=21, alpha=A(t, q('area', 0.35), 1.0))
    ax.text(0.07, 0.17, r'$A=\int_0^2 e^{-x^2}\,dx$', fontsize=21, color=BLUE,
            alpha=A(t, q('area', 0.6), 1.0) * (1 - A(t, q('strips', 0.25), 0.8)), clip_on=False)
    for i, xv in enumerate(np.arange(5) * 0.5):                       # strip dividers
        a = A(t, q('strips', 0.35) + 0.35 * i, 0.6)
        if a > 0.01:
            ax.plot([xv, xv], [0, f(xv)], color=FG, lw=1.0, alpha=0.55 * a, zorder=2)
    a = A(t, q('strips', 0.72), 0.8)                                  # width marker
    if a > 0.01:
        ax.annotate('', xy=(0, 0.045), xytext=(0.5, 0.045),
                    arrowprops=dict(arrowstyle='<->', color=YELLOW, lw=1.3, shrinkA=0, shrinkB=0, alpha=a))
        TA(ax, 0.25, 0.075, r'$h=0.5$', size=16, color=YELLOW, alpha=a, ha='center')
    a = A(t, q('rect', 0.62), 0.8)                                    # which height?
    if a > 0.01:
        ax.plot([1.25, 1.25], [f(1.5), f(1.0)], color=YELLOW, lw=1.6, alpha=a, zorder=4)
        for yv in (f(1.5), f(1.0)):
            ax.plot([1.21, 1.29], [yv, yv], color=YELLOW, lw=1.6, alpha=a, zorder=4)
        TA(ax, 1.32, 0.30, 'which height?', size=17, color=YELLOW, alpha=a, style='italic')
    margin(t, [
        dict(t=q('area'), text="We want the area under this curve, from $a=0$ to $b=2$."),
        dict(t=q('noanti'), text="$e^{-x^2}$ has no elementary antiderivative, so there is no closed form to find."),
        dict(t=q('samples'), kind='aside', text="Or $f$ is only measured samples: a table of numbers, no formula."),
        dict(t=q('strips'), text="Cut $[a,b]$ into $n$ strips of equal width $h=(b-a)/n$."),
        dict(t=q('rect'), text="Over each strip, swap the curve for one flat rectangle. The only question: how tall?"),
    ])


def scene2(t):
    q = lambda k, fr=0.0, off=0.0: Q(2, k, fr, off)
    header(2, "Three ways to choose the height")
    n = 4; h = 0.5
    modes = [
        ("Left endpoint", "sample at the left edge", lambda i: i * h, Lsum(n), r'$L_4$', 'left', SOFT),
        ("Right endpoint", "sample at the right edge", lambda i: (i + 1) * h, Rsum(n), r'$R_4$', 'right', SOFT),
        ("Midpoint", "sample at the centre", lambda i: (i + .5) * h, Msum(n), r'$M_4$', 'mid', YELLOW),
    ]
    xl = [0.06, 0.375, 0.69]; pw = 0.255
    for j, (name, sub, xf, val, sym, key, col) in enumerate(modes):
        t0 = q(key)
        ax = fig.add_axes([xl[j], 0.37, pw, 0.40])
        style_ax(ax, (0, 2.05), (0, 1.08), [0, 2], [0, 1], ['0', '2'], ['0', '1'], size=12)
        ax.plot(XS, f(XS), color=FG, lw=2.2, zorder=3)
        hts = [float(f(xf(i))) for i in range(n)]
        g0 = q(key, 0.12)
        rects(ax, hts, growth=lambda i, g0=g0: A(t, g0 + 0.3 * i, 0.8),
              alpha=0.30, ec=BLUE if j < 2 else YELLOW, fc=BLUE, lw=1.3)
        for i in range(n):
            a = A(t, g0 + 0.3 * i, 0.8)
            if a > 0.01:
                ax.plot([xf(i)], [float(f(xf(i)))], 'o', color=col if j == 2 else FG, ms=6.5, alpha=a, zorder=5)
        T(xl[j], 0.84, name, size=19, color=col if j == 2 else FG, alpha=A(t, t0, 0.8))
        T(xl[j], 0.805, sub, size=13, color=MUTED, style='italic', alpha=A(t, t0 + 0.2, 0.8))
        err = EXACT - val
        T(xl[j], 0.255, sym + r'$\,=\,$' + f'${val:.4f}$', size=19, alpha=A(t, q(key, 0.45), 0.8),
          color=YELLOW if j == 2 else FG)
        T(xl[j], 0.205, "error  " + sgn(err), size=16, alpha=A(t, q(key, 0.62), 0.8),
          color=YELLOW if j == 2 else CORAL)
    errLR = 0.5 * (abs(EXACT - Lsum(n)) + abs(EXACT - Rsum(n)))
    ratio = errLR / abs(EXACT - Msum(n))
    T(0.06, 0.125, f"error = true − estimate, true value {EXACT:.5f}.  Left and right bracket the truth, "
                   f"each off by about {errLR:.2f}.",
      size=15, color=SOFT, alpha=A(t, q('foot1'), 1.0))
    T(0.06, 0.080, f"The centre sample is off by {abs(EXACT - Msum(n)):.4f}: about {int(round(ratio, -1))}× smaller, "
                   f"from the same four function evaluations.",
      size=15, color=YELLOW, alpha=A(t, q('foot2'), 1.0))


def scene3(t):
    q = lambda k, fr=0.0, off=0.0: Q(3, k, fr, off)
    header(3, "The recipe")
    ax = main_axes()
    n = 4; h = 0.5
    ms = A_ + (np.arange(n) + .5) * h
    fm = f(ms)
    ax.plot(XS, f(XS), color=FG, lw=2.4, zorder=3)
    row_t0 = [q(f'r{i + 1}', 0.12) for i in range(n)]
    rects(ax, list(fm), growth=lambda i: A(t, row_t0[i], 1.0), alpha=0.36, ec=YELLOW, fc=BLUE, lw=1.4)
    for i in range(n):
        a = A(t, row_t0[i] - 0.4, 0.6)
        if a > 0.01:
            ax.plot([ms[i], ms[i]], [0, fm[i]], color=YELLOW, lw=1.2, ls=(0, (2, 2)), alpha=a, zorder=4)
            ax.plot([ms[i]], [fm[i]], 'o', color=YELLOW, ms=7.5, alpha=a, zorder=5)
            TA(ax, ms[i], -0.058, rf'$m_{i + 1}$', size=15, color=YELLOW, alpha=a, ha='center', va='center')
            TA(ax, ms[i], fm[i] + 0.05, f'{fm[i]:.3f}', size=14, color=YELLOW, alpha=a, ha='center')
    X = 0.715
    T(X, 0.850, "step", size=13, color=MUTED, style='italic', alpha=A(t, q('h'), 0.8))
    T(X, 0.790, r'$h=\frac{b-a}{n}=0.5$', size=20, alpha=A(t, q('h', 0.3), 1.0))
    T(X, 0.728, "centre of strip $i$", size=13, color=MUTED, style='italic', alpha=A(t, q('mi'), 0.8))
    T(X, 0.676, r'$m_i=a+(i-\frac{1}{2})\,h$', size=20, alpha=A(t, q('mi', 0.3), 1.0))
    T(X, 0.608, "add up the heights", size=13, color=MUTED, style='italic', alpha=A(t, q('Mn'), 0.8))
    T(X, 0.545, r'$M_n=h\sum_{i=1}^{n}f(m_i)$', size=21, color=YELLOW, alpha=A(t, q('Mn', 0.4), 1.0))
    ty = 0.425
    th = q('r1', 0, -0.4)
    T(X, ty, r'$i$', size=14, color=MUTED, alpha=A(t, th, 0.8))
    T(X + 0.045, ty, r'$m_i$', size=14, color=MUTED, alpha=A(t, th, 0.8))
    T(X + 0.115, ty, r'$f(m_i)$', size=14, color=MUTED, alpha=A(t, th, 0.8))
    hline(ty - 0.012, X, 0.94, alpha=0.5 * A(t, th, 0.8))
    for i in range(n):
        yy = ty - 0.046 * (i + 1)
        a = A(t, row_t0[i], 0.8)
        T(X, yy, f'{i + 1}', size=16, alpha=a)
        T(X + 0.045, yy, f'{ms[i]:.2f}', size=16, alpha=a)
        T(X + 0.115, yy, f'{fm[i]:.4f}', size=16, alpha=a)
    S = fm.sum()
    yy = ty - 0.046 * 5 + 0.002
    ts = q('sum')
    hline(yy + 0.032, X, 0.94, alpha=0.5 * A(t, ts, 0.8))
    T(X, yy, r'$\Sigma$', size=16, alpha=A(t, ts, 0.8))
    T(X + 0.115, yy, f'{S:.4f}', size=16, alpha=A(t, ts, 0.8))
    T(X, yy - 0.066, rf'$M_4=0.5\times{S:.4f}$', size=18, alpha=A(t, q('sum', 0.45), 1.0), color=YELLOW)
    T(X, yy - 0.120, rf'$=\,{Msum(4):.4f}$', size=21, alpha=A(t, q('sum', 0.75), 1.0), color=YELLOW)
    T(0.06, 0.055, f"True value {EXACT:.4f}.  Four function evaluations, one multiplication by $h$.",
      size=15, color=SOFT, alpha=A(t, q('foot'), 1.0))


def scene4(t):
    q = lambda k, fr=0.0, off=0.0: Q(4, k, fr, off)
    header(4, "Why it is so accurate")
    m, lo, hi = 1.25, 1.0, 1.5
    hh = hi - lo; fm = float(f(m)); s_full = float(f1(m))
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, (0.93, 1.57), (0, 0.42), [1.0, 1.25, 1.5], [0, 0.2, 0.4],
             ['1.0', '$m$', '1.5'], ['0', '0.2', '0.4'])
    xs = np.linspace(0.93, 1.57, 400)
    xstrip = np.linspace(lo, hi, 400)
    ax.add_patch(Rectangle((lo, 0), hh, fm, facecolor=to_rgba(BLUE, 0.28), edgecolor=to_rgba(BLUE, 1), lw=1.4, zorder=2))
    for xv in (lo, hi):
        ax.plot([xv, xv], [0, float(f(xv))], color=FG, lw=1.0, alpha=0.5, zorder=2)
    ax.plot(xs, f(xs), color=FG, lw=2.6, zorder=6)
    ax.plot([m], [fm], 'o', color=YELLOW, ms=8, zorder=8)
    TA(ax, 1.25, 0.055, r'area $=h\,f(m)$', size=18, color=BLUE, ha='center', alpha=A(t, q('zoom', 0.6), 1.0))
    t_sl = q('sliver', 0, -0.3)                                      # phase 2 starts here
    p = A(t, q('tilt', 0.30), 4.0)                                   # rotate flat top -> tangent
    s = s_full * p
    yt = lambda x: fm + s * (np.asarray(x) - m)
    wedge_a = 0.45 * p * (1 - 0.65 * A(t, t_sl, 1.2))
    if p > 0.003:
        ax.add_patch(Polygon([(lo, fm), (m, fm), (lo, float(yt(lo)))], closed=True,
                             facecolor=to_rgba(GREEN, wedge_a), edgecolor='none', zorder=3))
        ax.add_patch(Polygon([(m, fm), (hi, fm), (hi, float(yt(hi)))], closed=True,
                             facecolor=to_rgba(CORAL, wedge_a), edgecolor='none', zorder=3))
        ax.plot(xstrip, yt(xstrip), color=GREEN, lw=2.2, zorder=7)
        ax.plot([lo, hi], [fm, fm], color=BLUE, lw=1.6, alpha=0.9 * (1 - p), zorder=7)
        sig = A(t, q('equal', 0.1), 1.0) * (1 - A(t, t_sl, 1.0))
        TA(ax, lo + (m - lo) * 0.36, fm + 0.03, '+', size=30, color=GREEN, alpha=sig, ha='center', va='center')
        TA(ax, hi - (hi - m) * 0.30, float(f(hi)) + 0.045, '−', size=30, color=CORAL, alpha=sig, ha='center', va='center')
    else:
        ax.plot([lo, hi], [fm, fm], color=BLUE, lw=1.6, zorder=7)
    sl = A(t, q('sliver', 0.4), 1.5)                                  # the error sliver
    if sl > 0.01:
        yy = yt(xstrip); cc = f(xstrip)
        ax.fill_between(xstrip, yy, cc, where=cc >= yy, facecolor=to_rgba(YELLOW, 0.85 * sl), edgecolor='none', zorder=5)
        ax.annotate('the error', xy=(1.4, float(f(1.4))), xytext=(1.40, 0.33),
                    color=YELLOW, fontsize=17, alpha=sl, ha='left', style='italic',
                    arrowprops=dict(arrowstyle='-', color=YELLOW, lw=0.9, alpha=0.8 * sl, shrinkA=2, shrinkB=2))
    rect_area = hh * fm
    trap_area = hh * 0.5 * (float(yt(lo)) + float(yt(hi)))
    out = t_sl - 0.2
    margin(t, [
        dict(t=q('zoom'), out=out, text="Zoom in on one strip, $[1.0,\\ 1.5]$. The rectangle is as tall as the curve at the centre, $m=1.25$."),
        dict(t=q('tilt'), out=out, text="Now tilt that flat top into the tangent line at $m$."),
        dict(t=q('equal'), out=out, text="The wedge gained on the left mirrors the wedge lost on the right: equal areas."),
        dict(t=q('same'), out=out, text=f"So rectangle and trapezoid have the same area: {rect_area:.5f} and {trap_area:.5f}."),
    ])
    sliver = float(np.trapezoid(f(xstrip) - (fm + s_full * (xstrip - m)), xstrip))
    pred = float(f2(m)) * hh ** 3 / 24
    margin(t, [
        dict(t=q('sliver'), text="What the tangent misses is the yellow sliver. That is the entire error."),
        dict(t=q('formula', 0.3), kind='eq', h=0.085, size=19,
             text=r'$\int_{-h/2}^{h/2}\frac{f^{\,\prime\prime}(m)}{2}\,u^2\,du=\frac{f^{\,\prime\prime}(m)\,h^3}{24}$'),
        dict(t=q('check'), kind='aside', text=f"Here: formula {pred:.5f}, true sliver {sliver:.5f}. Close, not equal: higher-order terms."),
        dict(t=q('bend'), text="The curve bends up ($f''>0$), so the rectangle is too short. For a straight line $f''=0$: no sliver, no error."),
    ])


def scene5(t):
    q = lambda k, fr=0.0, off=0.0: Q(5, k, fr, off)
    header(5, "Add up the slivers")
    ax = fig.add_axes([0.075, 0.245, 0.60, 0.59])
    style_ax(ax, (0, 2.05), (0, 1.08), [0, .5, 1, 1.5, 2], [0, .5, 1],
             ['0', '0.5', '1', '1.5', '2'], ['0', '0.5', '1'])
    ax.plot(XS, f(XS), color=FG, lw=2.4, zorder=6)
    for i, (lo, hi) in enumerate(STRIPS):
        a = A(t, q('apply', 0.10 + 0.19 * i), 1.0)
        m = 0.5 * (lo + hi)
        ax.plot([lo, lo], [0, float(f(lo))], color=FG, lw=1.0, alpha=0.4, zorder=2)
        if a > 0.01:
            ax.add_patch(Rectangle((lo, 0), hi - lo, float(f(m)) * a, facecolor=to_rgba(BLUE, 0.18),
                                   edgecolor=to_rgba(BLUE, 0.7), lw=1.1, zorder=2))
            xx = np.linspace(lo, hi, 300)
            yt = float(f(m)) + float(f1(m)) * (xx - m)
            ax.plot(xx, yt, color=GREEN, lw=1.6, alpha=a, zorder=7)
            cc = f(xx)
            ax.fill_between(xx, yt, cc, where=cc >= yt, facecolor=to_rgba(YELLOW, 0.95 * a), edgecolor='none', zorder=5)
            ax.fill_between(xx, yt, cc, where=cc < yt, facecolor=to_rgba(CORAL, 0.95 * a), edgecolor='none', zorder=5)
    ax.plot([2, 2], [0, float(f(2))], color=FG, lw=1.0, alpha=0.4, zorder=2)
    a = A(t, q('bars', 0.5), 1.0)                                     # inflection hint
    TA(ax, 0.707, 0.98, 'bends down ← | → bends up', size=14, color=MUTED, style='italic', alpha=a, ha='center')
    ax.plot([0.7071, 0.7071], [0.66, 0.93], color=MUTED, lw=0.8, ls=(0, (3, 3)), alpha=0.8 * a, zorder=1)
    bx = fig.add_axes([0.725, 0.335, 0.215, 0.23])                    # sparkbars: error per strip
    bx.set_facecolor('none')
    for s_ in ('top', 'right', 'bottom', 'left'): bx.spines[s_].set_visible(False)
    bx.set_xlim(-0.6, 3.6); bx.set_ylim(-11, 7)
    bx.set_xticks([]); bx.set_yticks([])
    b0 = q('bars', 0.05)
    bx.axhline(0, color=MUTED, lw=0.8, alpha=0.7 * A(t, b0, 1.0))
    for i, e in enumerate(STRIP_ERR):
        a = A(t, b0 + 0.9 * i, 0.9)
        v = e * 1e3
        bx.bar(i, v * a, width=0.62, color=YELLOW if v >= 0 else CORAL, alpha=0.95)
        if a > 0.5:
            bx.text(i, v + (0.7 if v >= 0 else -0.7), sgn(e, 2, 1e-3), ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=13, color=FG, alpha=(a - 0.5) * 2)
    T(0.725, 0.625, "error per strip, true − midpoint", size=13, color=MUTED, style='italic', alpha=A(t, q('bars'), 1.0))
    T(0.725, 0.597, r"in units of $10^{-3}$", size=13, color=MUTED, style='italic', alpha=A(t, q('bars'), 1.0))
    margin(t, [
        dict(t=q('apply'), text="Do this on every strip. Yellow: curve above its tangent, rectangle too short. Coral: below, too tall."),
    ], top=0.835)
    net = EXACT - Msum(4)
    T(0.725, 0.275, "net, true − $M_4$:", size=15, color=SOFT, alpha=A(t, q('net', 0.3), 1.0))
    T(0.725, 0.225, neg(f"{net:.5f}"), size=22, color=CORAL, alpha=A(t, q('net', 0.3), 1.0))
    T(0.725, 0.180, "the estimate overshoots:\nthe bends partly cancel.", size=14, color=SOFT, style='italic',
      alpha=A(t, q('net', 0.7), 1.0), va='top', linespacing=1.3)
    pred = 0.5 ** 2 / 24 * float(f1(2) - f1(0))
    T(0.06, 0.118,
      r"$\sum_i\frac{f^{\,\prime\prime}(m_i)\,h^3}{24}\;\approx\;\frac{h^2}{24}\sum_i f^{\,\prime\prime}(m_i)\,h\;\approx\;"
      r"\frac{h^2}{24}\,[\,f^{\,\prime}(b)-f^{\,\prime}(a)\,]$",
      size=19, alpha=A(t, q('telescope', 0.1), 1.2))
    T(0.06, 0.040, neg(f"Prediction h²/24 · (f′(2) − f′(0)) = {pred:.5f};   actual {net:.5f}.  "
                       "The bends telescope into the slopes at the two ends."),
      size=14.5, color=SOFT, alpha=A(t, q('predict'), 1.0))


def scene6(t):
    q = lambda k, fr=0.0, off=0.0: Q(6, k, fr, off)
    header(6, "How fast does the error shrink?")
    ax = fig.add_axes(AX_MAIN)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_facecolor('none')
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(MUTED); ax.spines[s_].set_linewidth(0.8); ax.spines[s_].set_position(('outward', 6))
    ax.set_xlim(1.7, 2048 * 11); ax.set_ylim(2e-10, 1.0)
    xt = [2, 8, 32, 128, 512, 2048]
    ax.set_xticks(xt); ax.set_xticklabels([str(v) for v in xt])
    yt = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
    ax.set_yticks(yt); ax.set_yticklabels([r'$10^{%d}$' % int(round(math.log10(v))) for v in yt])
    ax.spines['bottom'].set_bounds(2, 2048); ax.spines['left'].set_bounds(1e-9, 1e-1)
    ax.tick_params(colors=MUTED, labelsize=13, length=3.5, width=0.8, pad=4)
    ax.tick_params(which='minor', length=0)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    T(0.075, 0.845, r"absolute error $|\,\mathrm{true}-\mathrm{estimate}\,|$", size=14, color=MUTED, style='italic',
      alpha=A(t, q('setup', 0.5), 1.0))
    T(0.675 - 0.01, 0.105, r"strips $n$", size=14, color=MUTED, style='italic', ha='right',
      alpha=A(t, q('setup', 0.5), 1.0))

    def npts(p): return 0 if p <= 0.001 else max(1, int(round(1 + p * (len(NS) - 1))))
    kLR = npts(A(t, q('reveal', 0.05), 0.85 * Qd(6, 'reveal')))     # left/right first,
    kTM = npts(A(t, q('mid', 0.05), 0.45 * Qd(6, 'mid')))           # then trapezoid + midpoint
    for key, col, ls, lw, k in [('R', SOFT, '--', 1.4, kLR), ('L', SOFT, '-', 1.8, kLR),
                                ('T', GREEN, '-', 2.2, kTM), ('M', YELLOW, '-', 2.8, kTM)]:
        if k: ax.plot(NS[:k], ERR[key][:k], color=col, ls=ls, lw=lw, marker='o', ms=4.5, zorder=4)
    end = NS[-1] * 1.35
    sL, sT, sM = slope(ERR['L']), slope(ERR['T']), slope(ERR['M'])
    a = A(t, q('reveal', 0.85), 1.0)
    if a > 0.01:
        ax.text(end, ERR['L'][-1], neg(f"left, right\nslope {sL:.2f}"), color=SOFT, fontsize=15, alpha=a,
                va='center', ha='left', linespacing=1.2)
    a = A(t, q('mid', 0.6), 1.0)
    if a > 0.01:
        ax.text(end, ERR['T'][-1] * 1.35, neg(f"trapezoid\nslope {sT:.2f}"), color=GREEN, fontsize=15, alpha=a,
                va='bottom', ha='left', linespacing=1.2)
        ax.text(end, ERR['M'][-1] / 1.35, neg(f"midpoint\nslope {sM:.2f}"), color=YELLOW, fontsize=16, alpha=a,
                va='top', ha='left', linespacing=1.2, fontweight='bold')
    a2 = A(t, q('mid', 0.8), 1.0)                                    # the n -> 2n bracket
    if a2 > 0.01:
        i16 = int(np.where(NS == 16)[0][0]); i32 = i16 + 1
        e16, e32 = ERR['M'][i16], ERR['M'][i32]
        ax.plot([16, 32, 32], [e16, e16, e32], color=FG, lw=1.0, alpha=0.9 * a2, zorder=6)
        ax.text(36, e16 * 4.5, f"n → 2n:  error ÷{e16 / e32:.2f}", color=FG, fontsize=15,
                alpha=a2, ha='left', va='center')
        ax.plot([32, 36], [e16, e16 * 3.2], color=FG, lw=0.8, alpha=0.7 * a2)
    margin(t, [
        dict(t=q('reveal'), text="Left, right: error $\\propto h$. Halve $h$ and the error halves."),
        dict(t=q('mid'), text="Midpoint, trapezoid: error $\\propto h^2$. Halve $h$ and the error drops about fourfold."),
        dict(t=q('half'), text=neg(f"Midpoint's error is half the trapezoid's, with the opposite sign: $E_T/E_M\\to-2$ "
                                   f"({(EXACT - Tsum(1024)) / (EXACT - Msum(1024)):.4f} at $n=1024$).")),
        dict(t=q('cost'), kind='aside', text="And $M_n$ costs $n$ evaluations of $f$; $T_n$ costs $n+1$."),
    ], width=29)


def scene7(t):
    q = lambda k, fr=0.0, off=0.0: Q(7, k, fr, off)
    header(7, "Where this leaves us")
    cols = [0.06, 0.30, 0.52, 0.715]
    hy = 0.775
    th = q('row1', 0, -0.7)
    for x, s in zip(cols, ["rule", "samples f at", "exact for", "error, true − estimate"]):
        T(x, hy, s, size=14, color=MUTED, style='italic', alpha=A(t, th + 0.2, 0.8))
    hline(0.815, alpha=0.7 * A(t, th, 0.8)); hline(hy - 0.025, alpha=0.5 * A(t, th + 0.2, 0.8))
    rows = [
        ("Left / right", "one end of each strip", "constants", r"$O(h)$", FG),
        ("Midpoint", "centre of each strip", "lines", r"$\approx+\frac{h^2}{24}\,[\,f^{\,\prime}(b)-f^{\,\prime}(a)\,]$", YELLOW),
        ("Trapezoid", "both ends, averaged", "lines", r"$\approx-\frac{h^2}{12}\,[\,f^{\,\prime}(b)-f^{\,\prime}(a)\,]$", GREEN),
        ("Simpson", r"$(2M_n+T_n)/3$", "cubics", r"$O(h^4)$", FG),
    ]
    for i, (a_, b_, c_, d_, col) in enumerate(rows):
        yy = hy - 0.105 - 0.105 * i
        al = A(t, q(f'row{i + 1}', 0.05), 0.9)
        T(cols[0], yy, a_, size=20, color=col, alpha=al)
        T(cols[1], yy, b_, size=17, color=SOFT, alpha=al)
        T(cols[2], yy, c_, size=17, color=SOFT, alpha=al)
        T(cols[3], yy - 0.005, d_, size=18, color=col if col != FG else SOFT, alpha=al)
    hline(0.775 - 0.105 - 0.105 * 3 - 0.05, alpha=0.7 * A(t, q('row4', 0.9), 0.8))
    M1000 = M_endpoint_singularity(1000)
    T(0.06, 0.215, "Assumes $f$ is smooth. Because $M_n$ never evaluates $f$ at $a$ or $b$, it survives endpoint blow-ups,",
      size=15, color=SOFT, alpha=A(t, q('caveat'), 1.0))
    T(0.06, 0.172, rf"but slowly: $\int_0^1 x^{{-1/2}}dx=2$, yet $M_{{1000}}={M1000:.3f}$.",
      size=15, color=SOFT, alpha=A(t, q('caveat', 0.45), 1.0))
    T(0.06, 0.092, r"Next: the $2{:}1$ blend of midpoint and trapezoid is Simpson's rule.", size=19, color=YELLOW,
      alpha=A(t, q('next'), 1.2))


SCENES = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7]
FADE_IN_FIRST, FADE_OUT_LAST = 0.9, 1.4


def fade_val(idx, t):
    v = 1.0
    if idx == 0: v = min(v, clamp(t / FADE_IN_FIRST))
    if idx == len(SCENES) - 1: v = min(v, clamp((TL['scenes'][idx]['duration'] - t) / FADE_OUT_LAST))
    return v


def render_frame(idx, t, t_real=None):
    """RGB bytes for scene `idx` at local time t (t_real drives fades when t is clamped)."""
    new_frame()
    SCENES[idx](t)
    fv = fade_val(idx, t if t_real is None else t_real)
    if fv < 0.999:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor='none',
                                 alpha=1 - fv, zorder=100))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
