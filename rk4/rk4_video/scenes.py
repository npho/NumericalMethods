"""Scene drawing for the Runge-Kutta video: a 3Blue1Brown x Tufte hybrid.

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

from . import mathcore as M
from .mathcore import FACTS as F

# ------------------------------------------------------------------ look
BG = '#12141a'; FG = '#ece9e1'; MUTED = '#8b909a'; SOFT = '#b9bdc6'
BLUE = '#58C4DD'; YELLOW = '#FFD35A'; CORAL = '#FC6255'; GREEN = '#83C167'
FONT = 'TeX Gyre Pagella'              # Palatino clone, bundled in ../fonts (GUST Font License)
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts')
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
    T(0.092 if num < 10 else 0.108, 0.915, title, size=22)
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




# ------------------------------------------------------------------ extra helpers for this video
AX_MAIN = [0.075, 0.15, 0.60, 0.68]


def lin(t, t0, t1): return clamp((t - t0) / max(1e-9, t1 - t0))


def sci(x, nd=1):
    """x as a mathtext 'a x 10^b' string."""
    if x == 0: return r'$0$'
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(m, nd) >= 10: m /= 10; e += 1
    return rf'${m:.{nd}f}\times10^{{{e}}}$'


def phase_ax(rect, lim, ticks, size=12):
    ax = fig.add_axes(rect)
    style_ax(ax, (-lim, lim), (-lim, lim), ticks, ticks, size=size)
    ax.set_aspect('equal', adjustable='box')
    return ax


def trail(ax, xs, ys, p, color=BLUE, lw=1.8, ms=4.0, dots=0, tip=True, alpha=1.0, z=4, ls='-'):
    """Draw the first fraction p of a polyline (with a tip marker and optional sample dots)."""
    n = len(xs); k = int(round(clamp(p) * (n - 1))) + 1
    if k < 2: return
    ax.plot(xs[:k], ys[:k], color=color, lw=lw, alpha=alpha, zorder=z, ls=ls, solid_joinstyle='round')
    if dots: ax.plot(xs[:k:dots], ys[:k:dots], 'o', color=color, ms=ms, alpha=alpha, zorder=z + 1)
    if tip: ax.plot([xs[k - 1]], [ys[k - 1]], 'o', color=color, ms=ms + 2.5, alpha=alpha, zorder=z + 2)


def bez(p0, c, p1, n=60):
    s = np.linspace(0, 1, n)
    return ((1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * c[0] + s ** 2 * p1[0],
            (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * c[1] + s ** 2 * p1[1])


def arc(ax, p0, c, p1, prog, color, lw=2.2, alpha=1.0, z=4, ls='-'):
    if prog <= 0.01: return
    X, Y = bez(p0, c, p1)
    k = max(2, int(len(X) * prog))
    ax.plot(X[:k], Y[:k], color=color, lw=lw, alpha=alpha, zorder=z, solid_capstyle='round', ls=ls)
    if prog > 0.97:
        ax.annotate('', xy=(X[-1], Y[-1]), xytext=(X[-5], Y[-5]), zorder=z,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0,
                                    mutation_scale=15))


def fmt(x, nd=4):
    return (f"{x:.{nd}f}").replace('-', '−')




# ------------------------------------------------------------------ extra helpers for this video
AX_MAIN = [0.075, 0.15, 0.60, 0.68]
EXACT_C = SOFT


def lin(t, t0, t1): return clamp((t - t0) / max(1e-9, t1 - t0))


def poly_prog(ax, xs, ys, p, color=BLUE, lw=2.0, dots=True, ms=5.0, z=4, alpha=1.0, tip=True):
    """Draw the first fraction p of a polyline (continuous along the segments), dots at the vertices."""
    n = len(xs) - 1
    if p <= 0.0 or n < 1: return
    u = clamp(p) * n; k = min(n, int(u)); fr = u - k
    X = list(xs[:k + 1]); Y = list(ys[:k + 1])
    if k < n:
        X.append(xs[k] + fr * (xs[k + 1] - xs[k])); Y.append(ys[k] + fr * (ys[k + 1] - ys[k]))
    ax.plot(X, Y, color=color, lw=lw, alpha=alpha, zorder=z, solid_joinstyle='round', clip_on=False)
    if dots: ax.plot(xs[:k + 1], ys[:k + 1], 'o', color=color, ms=ms, alpha=alpha, zorder=z + 1, clip_on=False)
    if tip: ax.plot([X[-1]], [Y[-1]], 'o', color=color, ms=ms + 2.5, alpha=alpha, zorder=z + 2, clip_on=False)


def pw(v, nd):
    """Mathtext power of ten label for log axes: 10^v."""
    return rf'$10^{{{v}}}$'


def slope_ticks(ax, tr, yr, nt, ny, L=0.17, f=lambda t, y: y):
    """Slope-field ticks, drawn at the true on-screen angle. Returns list of (t0, y0, t1, y1, tc)."""
    pos = ax.get_position()
    sx = pos.width * 12.8 / (tr[1] - tr[0]); sy = pos.height * 7.2 / (yr[1] - yr[0])
    out = []
    for ti in np.linspace(tr[0], tr[1], nt):
        for yi in np.linspace(yr[0], yr[1], ny):
            m = f(ti, yi) * sy / sx
            dx = 0.5 * L / math.sqrt(1 + m * m); dy = m * dx
            out.append((ti - dx / sx, yi - dy / sy, ti + dx / sx, yi + dy / sy, ti))
    return out


def fmt(x, nd=4):
    return (f"{x:.{nd}f}").replace('-', '−')


def hair_table(t, x0s, y0, pitch, rows, size=14, header=None, header_t=0.0, aligns=None, colors=None):
    """Hairline table in the margin. rows = list of (t_appear, [cells])."""
    if header:
        a = A(t, header_t, 0.8)
        for x, s, al in zip(x0s, header, aligns):
            T(x, y0, s, size=size - 1, color=MUTED, alpha=a, ha=al)
        hline(y0 - 0.013, x0s[0] - 0.005, x0s[-1] + 0.01, alpha=0.5 * a, lw=0.5)
    for i, (ta, cells) in enumerate(rows):
        a = A(t, ta, 0.7)
        for j, (x, s, al) in enumerate(zip(x0s, cells, aligns)):
            col = FG if colors is None or colors[j] is None else colors[j]
            T(x, y0 - 0.047 - i * pitch, s, size=size, alpha=a, ha=al, color=col)




# ------------------------------------------------------------------ small helpers for this video
KC = [BLUE, YELLOW, GREEN, CORAL]                       # colours of k1..k4
KN = ['k_1', 'k_2', 'k_3', 'k_4']
E = math.e


def tick(ax, t0, y0, m, half=0.17, color=FG, lw=3.0, alpha=1.0, z=6):
    """Short line of slope m through (t0, y0), in data coordinates (so it matches the curve)."""
    ax.plot([t0 - half, t0 + half], [y0 - m * half, y0 + m * half], color=color, lw=lw, alpha=alpha, zorder=z,
            solid_capstyle='round')


def axes_ty(xr=(0, 1.4), yr=(0, 3.2), xt=(0, 0.5, 1), yt=(0, 1, 2, 3), rect=None, size=13):
    ax = fig.add_axes(rect or AX_MAIN)
    style_ax(ax, xr, yr, list(xt), list(yt), xlabels=[f"{v:g}" for v in xt], size=size)
    ax.set_xlabel('time  $t$', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('$y$', fontsize=15, color=MUTED, labelpad=4, rotation=0)
    return ax


def exact_curve(ax, t1=1.45, alpha=0.55, label=True):
    tt = np.linspace(0, t1, 200)
    ax.plot(tt, np.exp(tt), color=FG, lw=1.5, ls=(0, (4, 3)), alpha=alpha, zorder=2)
    if label: TA(ax, 1.22, 2.9, "$e^{t}$", size=16, color=SOFT, alpha=0.9)


def seg(ax, p0, p1, prog, color, lw=2.6, alpha=1.0, z=5, ls='-'):
    if prog <= 0: return
    x = p0[0] + prog * (p1[0] - p0[0]); y = p0[1] + prog * (p1[1] - p0[1])
    ax.plot([p0[0], x], [p0[1], y], color=color, lw=lw, alpha=alpha, zorder=z, ls=ls, solid_capstyle='round')


def dot(ax, p, color, ms=9, alpha=1.0, z=8, mec=None):
    ax.plot([p[0]], [p[1]], 'o', color=color, ms=ms, alpha=alpha, zorder=z, mec=mec or color)


# ================================================================== SCENES
def scene0(t):
    q = lambda k, fr=0.0, off=0.0: Q(0, k, fr, off)
    t0 = q('hook', 0, 0.1)
    T(0.07, 0.585, "Runge–Kutta", size=42, alpha=A(t, t0, 1.2))
    T(0.07, 0.505, "Four slopes, one step.", size=21, color=SOFT, style='italic', alpha=A(t, t0 + 1.0, 1.2), va='top')
    hline(0.685, 0.07, 0.55, alpha=0.7 * A(t, t0 + 0.6, 1.0))
    T(0.07, 0.24, "Numerical integration · part 4", size=14, color=MUTED, alpha=A(t, t0 + 2.2, 1.0))
    ax = fig.add_axes([0.62, 0.18, 0.32, 0.56])
    ax.set_facecolor('none'); ax.axis('off')
    ax.set_xlim(-0.1, 2.15); ax.set_ylim(-0.3, 8.2)
    tt = np.linspace(0, 2, 200)
    a = A(t, t0 + 0.4, 1.0)
    ax.plot(tt, np.exp(tt), color=SOFT, lw=1.4, ls=(0, (4, 3)), alpha=0.8 * a)
    poly_prog(ax, M.TAB_T, M.TAB_Y, lin(t, t0 + 0.8, t0 + 11.0), color=BLUE, lw=2.4, ms=6.0)
    ax.plot([0], [1], 'o', color=YELLOW, ms=8, zorder=8, alpha=a)


def scene1(t):
    q = lambda k, fr=0.0, off=0.0: Q(1, k, fr, off)
    header(1, "What Euler's step misses")
    ax = axes_ty()
    exact_curve(ax)
    dot(ax, (0, 1), YELLOW, 10)
    p = lin(t, q('euler', 0.25), q('euler', 0.7))
    seg(ax, (0, 1), (1, 2), p, BLUE, lw=3.0)
    if p > 0: dot(ax, (p, 1 + p), BLUE, 9, z=7)
    a = A(t, q('euler', 0.62), 0.8)
    if a > 0.01:
        dot(ax, (1, E), FG, 9, alpha=a)
        TA(ax, 0.94, E + 0.03, f"$e={E:.3f}$", size=17, ha='right', alpha=a)
        ax.plot([1, 1], [2, E], color=CORAL, lw=3.2, alpha=a, zorder=7, solid_capstyle='butt')
        TA(ax, 1.05, 2.35, f"miss $={F['one_err']['euler']:.3f}$", size=16, color=CORAL, alpha=a)
        TA(ax, 1.05, 1.97, "$y_1=2$", size=15, color=BLUE, alpha=a)
    # the true slopes along the curve
    for tt, tf in ((0.0, 0.0), (0.5, 0.35), (1.0, 0.7)):
        ai = A(t, q('idea', tf), 0.8)
        if ai > 0.01:
            tick(ax, tt, math.exp(tt), math.exp(tt), half=0.17, color=YELLOW, lw=3.0, alpha=ai)
            lx, ly, ha = {0.0: (0.04, 0.66, 'left'), 0.5: (0.5, 2.0, 'center'), 1.0: (0.97, 3.05, 'right')}[tt]
            TA(ax, lx, ly, f"slope {math.exp(tt):.2f}".replace('1.00', '1'), size=14, color=YELLOW, alpha=ai, ha=ha)
    margin(t, [
        dict(t=q('setup'), kind='eq', size=22, h=0.07, text=r"$y'=y,\quad y(0)=1$"),
        dict(t=q('euler'), text="Euler: $y_1=y_0+h\\,f(t_0,y_0)=1+1\\cdot1=2$."),
        dict(t=q('idea'), kind='aside', text="The true slope grows along the step, from 1 to 2.72. "
                                             "Euler uses only the first."),
    ])


def scene2(t):
    q = lambda k, fr=0.0, off=0.0: Q(2, k, fr, off)
    header(2, "Look in the middle")
    ax = axes_ty()
    exact_curve(ax)
    dot(ax, (0, 1), YELLOW, 10)
    p = lin(t, q('half', 0.25), q('half', 0.6))
    seg(ax, (0, 1), (0.5, 1.5), p, BLUE, lw=3.0)
    a1 = A(t, q('half', 0.62), 0.8)
    if a1 > 0.01:
        dot(ax, (0.5, 1.5), BLUE, 9, alpha=a1, z=7)
        tick(ax, 0.5, 1.5, 1.5, half=0.2, color=YELLOW, lw=3.2, alpha=a1)
        TA(ax, 0.55, 1.1, "slope here $=1.5$", size=15, color=YELLOW, alpha=a1)
    pf = lin(t, q('full', 0.12), q('full', 0.5))
    seg(ax, (0, 1), (1, 2.5), pf, YELLOW, lw=3.4, z=6)
    af = A(t, q('full', 0.5), 0.8)
    if af > 0.01:
        dot(ax, (1, 2.5), YELLOW, 10, alpha=af)
        TA(ax, 1.05, 2.5, f"$y_1={F['one']['midpoint']:.1f}$", size=17, color=YELLOW, alpha=af)
        dot(ax, (1, E), FG, 8, alpha=af)
        ax.plot([1, 1], [2.5, E], color=CORAL, lw=3.0, alpha=af, zorder=7)
        TA(ax, 1.05, 2.68, f"miss $={F['one_err']['midpoint']:.3f}$", size=15, color=CORAL, alpha=af)
        TA(ax, 0.94, E + 0.03, f"$e={E:.3f}$", size=16, ha='right', alpha=af)
    ag = A(t, q('gap', 0.25), 0.9)
    if ag > 0.01:
        dot(ax, (0.5, math.exp(0.5)), FG, 8, alpha=ag)
        ax.plot([0.5, 0.5], [1.5, math.exp(0.5)], color=CORAL, lw=2.6, alpha=ag, zorder=7)
        TA(ax, 0.40, 1.80, f"true height {math.exp(0.5):.3f}", size=14, color=CORAL, alpha=ag, ha='right')
    margin(t, [
        dict(t=q('half'), kind='eq', size=18, h=0.06, text=r"$k_1=f(t_n,y_n)$"),
        dict(t=q('half', 0.4), kind='eq', size=18, h=0.07, text=r"$k_2=f\!\left(t_n+\frac{h}{2},\,y_n+\frac{h}{2}k_1\right)$"),
        dict(t=q('full'), kind='eq', size=18, h=0.06, text=r"$y_{n+1}=y_n+h\,k_2$"),
        dict(t=q('gap'), kind='aside', text="The middle point was reached with the crude slope $k_1$, "
                                            "so it is slightly low: 1.5, not 1.649."),
    ])


def scene3(t):
    q = lambda k, fr=0.0, off=0.0: Q(3, k, fr, off)
    header(3, "Four slopes")
    ax = axes_ty()
    exact_curve(ax, alpha=0.4)
    k = F['k']
    pts = [(0, 1), (0.5, 1.5), (0.5, 1.75), (1, 2.75)]
    dot(ax, (0, 1), YELLOW, 10)
    # k1: slope at the start, shown at (0,1)
    a = A(t, q('k1', 0.2), 0.7)
    tick(ax, 0, 1, k[0], half=0.2, color=KC[0], alpha=a)
    TA(ax, 0.06, 0.72, "$k_1=1$", size=17, color=KC[0], alpha=a)
    # k2
    for i, key in ((1, 'k2'), (2, 'k3'), (3, 'k4')):
        pr = lin(t, q(key, 0.10), q(key, 0.45))
        src = (0, 1) if i != 3 else (0, 1)
        seg(ax, src, pts[i], pr, KC[i - 1], lw=2.2, alpha=0.85, ls=(0, (4, 3)))
        a = A(t, q(key, 0.45), 0.7)
        if a > 0.01:
            dot(ax, pts[i], KC[i], 8, alpha=a, z=7)
            tick(ax, pts[i][0], pts[i][1], k[i], half=0.2, color=KC[i], alpha=a)
    la = {'k2': (0.58, 1.08, "$k_2=1.5$", 1), 'k3': (0.30, 2.02, "$k_3=1.75$", 2), 'k4': (1.0, 3.06, "$k_4=2.75$", 3)}
    for key, (lx, ly, s, i) in la.items():
        TA(ax, lx, ly, s, size=17, color=KC[i], alpha=A(t, q(key, 0.5), 0.7), ha='left' if key != 'k4' else 'center')
    # weighted slope
    pa = lin(t, q('avg', 0.35), q('avg', 0.8))
    seg(ax, (0, 1), (1, 1 + F['k_avg']), pa, FG, lw=4.0, z=6)
    aa = A(t, q('avg', 0.8), 0.8)
    if aa > 0.01:
        dot(ax, (1, 1 + F['k_avg']), FG, 10, alpha=aa)
        dot(ax, (1, E), SOFT, 8, alpha=aa, z=7)
        TA(ax, 1.06, 2.45, f"average slope\n$={F['k_avg']:.4f}$", size=15, color=FG, alpha=aa, linespacing=1.2, va='top')
    margin(t, [
        dict(t=q('k1'), kind='eq', size=15, h=0.055, text=r"$k_1=f(t_n,\,y_n)$", color=KC[0]),
        dict(t=q('k2'), kind='eq', size=15, h=0.062, text=r"$k_2=f(t_n+\frac{h}{2},\,y_n+\frac{h}{2}k_1)$", color=KC[1]),
        dict(t=q('k3'), kind='eq', size=15, h=0.062, text=r"$k_3=f(t_n+\frac{h}{2},\,y_n+\frac{h}{2}k_2)$", color=KC[2]),
        dict(t=q('k4'), kind='eq', size=15, h=0.062, text=r"$k_4=f(t_n+h,\,y_n+h\,k_3)$", color=KC[3]),
        dict(t=q('avg'), kind='eq', size=16, h=0.075, text=r"$y_{n+1}=y_n+\frac{h}{6}\,(k_1+2k_2+2k_3+k_4)$"),
    ], top=0.835, x=0.705, gap=0.015)


def scene4(t):
    q = lambda k, fr=0.0, off=0.0: Q(4, k, fr, off)
    header(4, "One step, by the numbers")
    xs = [0.07, 0.30]
    k = F['k']
    rows = [(q('setup', 0.5), r"$k_1$", f"{k[0]:g}", KC[0]), (q('ks', 0.12), r"$k_2$", f"{k[1]:g}", KC[1]),
            (q('ks', 0.45), r"$k_3$", f"{k[2]:g}", KC[2]), (q('ks', 0.78), r"$k_4$", f"{k[3]:g}", KC[3]),
            (q('avg', 0.1), r"$\frac{1}{6}(k_1+2k_2+2k_3+k_4)$", f"{F['k_avg']:.4f}", FG),
            (q('avg', 0.5), r"$y_1=1+h\cdot$average", f"{F['one']['rk4']:.4f}", FG)]
    for i, (ta, name, val, col) in enumerate(rows):
        a = A(t, ta, 0.7); y = 0.78 - i * 0.092 - (0.02 if i >= 4 else 0)
        T(0.07, y, name, size=19, color=col, alpha=a)
        T(0.44, y, val, size=21, color=col, alpha=a, ha='right')
        hline(y - 0.03, 0.07, 0.45, alpha=0.3 * a, lw=0.5)
    T(0.07, 0.16, "$y'=y$, $y(0)=1$, $h=1$.  All $k$ values are exact fractions: $1,\\ \\frac{3}{2},\\ \\frac{7}{4},\\ \\frac{11}{4}$.",
      size=13, color=MUTED, style='italic', alpha=A(t, q('ks', 0.5), 1.0))
    # error dot plot (log scale)
    ax = fig.add_axes([0.58, 0.20, 0.36, 0.50])
    ax.set_xscale('log')
    style_ax(ax, (4e-3, 1.5), (-0.6, 2.6), [1e-2, 1e-1, 1], [0, 1], xlabels=[pw(-2, 0), pw(-1, 0), pw(0, 0)], size=13)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel('error of one step of size 1', fontsize=14, color=MUTED, labelpad=6)
    ae = A(t, q('compare', 0.05), 0.9)
    for yy, m, col, lab in ((2, 'euler', BLUE, 'Euler'), (1, 'midpoint', YELLOW, 'midpoint'), (0, 'rk4', GREEN, 'Runge–Kutta 4')):
        ev = F['one_err'][m]
        am = A(t, q('compare', 0.05 + 0.25 * (2 - yy)), 0.8)
        if am > 0.01:
            ax.plot([4e-3, ev], [yy, yy], color=col, lw=1.0, alpha=0.5 * am, zorder=2)
            ax.plot([ev], [yy], 'o', color=col, ms=11, alpha=am, zorder=4)
            TA(ax, 4.3e-3, yy + 0.22, lab, size=15, color=col, alpha=am)
            TA(ax, ev * 1.35, yy - 0.08, f"{ev:.3f}", size=16, color=col, alpha=am)


def scene5(t):
    q = lambda k, fr=0.0, off=0.0: Q(5, k, fr, off)
    header(5, "Where the weights come from")
    ax = fig.add_axes([0.075, 0.15, 0.60, 0.66])
    style_ax(ax, (0, 1), (0, 3.2), [0, 0.5, 1], [0, 1, 2, 3], xlabels=['$t_n$', '$t_n+\\frac{h}{2}$', '$t_n+h$'], size=13)
    ax.set_ylabel(r"slope  $f(t)$", fontsize=14, color=MUTED, labelpad=4)
    tt = np.linspace(0, 1, 200)
    a0 = A(t, q('setup', 0.3), 0.9)
    ax.plot(tt, np.exp(tt), color=FG, lw=2.0, alpha=0.9 * a0, zorder=4)
    TA(ax, 0.04, 2.85, "$f(t)=e^{t}$, no $y$ in it", size=15, color=SOFT, alpha=a0)
    ax.fill_between(tt, 0, np.exp(tt), color=BLUE, alpha=0.10 * a0, lw=0)
    asm = A(t, q('simpson', 0.15), 0.9)
    if asm > 0.01:
        # parabola through the three samples (Simpson's rule integrates exactly this)
        c = np.polyfit([0, 0.5, 1], [1, math.exp(0.5), E], 2)
        ax.plot(tt, np.polyval(c, tt), color=YELLOW, lw=2.2, ls=(0, (5, 3)), alpha=asm, zorder=5)
        for xx, yy, nm, w in ((0, 1, r"$k_1$", 1), (0.5, math.exp(0.5), r"$k_2=k_3$", 4), (1, E, r"$k_4$", 1)):
            ax.plot([xx, xx], [0, yy], color=YELLOW, lw=1.0, alpha=0.6 * asm)
            dot(ax, (xx, yy), YELLOW, 8, alpha=asm)
            TA(ax, xx + (0.03 if xx < 1 else -0.03), yy + 0.18, nm, size=15, color=YELLOW, alpha=asm,
               ha='left' if xx < 1 else 'right')
            TA(ax, xx + (0.05 if xx == 0 else (-0.05 if xx == 1 else 0)), 0.25, f"weight  {w}/6", size=14, color=SOFT, alpha=asm,
               ha='left' if xx == 0 else ('right' if xx == 1 else 'center'))
    mg = [
        dict(t=q('setup'), text="If the slope depends only on time, the equation is just an integral: "
                                "$y_{n+1}-y_n=\\int f\\,dt$."),
        dict(t=q('simpson'), kind='eq', size=17, h=0.07, text=r"$k_2=k_3$"),
        dict(t=q('simpson', 0.3), kind='eq', size=16, h=0.075, text=r"$\frac{h}{6}\,(k_1+4k_2+k_4)$"),
        dict(t=q('simpson', 0.6), text="Simpson's rule: the area under the parabola through three samples."),
        dict(t=q('meaning'), kind='aside', text=f"Here: Simpson {F['simp_val']:.5f}, one RK4 step "
                                                f"{F['simp_rk4']:.5f}, exact {F['simp_exact']:.5f}."),
    ]
    margin(t, mg, top=0.835, x=0.705, width=31)


def scene6(t):
    q = lambda k, fr=0.0, off=0.0: Q(6, k, fr, off)
    header(6, "Four steps of one half")
    ax = fig.add_axes([0.075, 0.15, 0.56, 0.68])
    style_ax(ax, (0, 2), (0, 8), [0, 0.5, 1, 1.5, 2], [0, 2, 4, 6, 8], xlabels=['0', '0.5', '1', '1.5', '2'], size=13)
    ax.set_xlabel('time  $t$', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('$y$', fontsize=15, color=MUTED, labelpad=4, rotation=0)
    tt = np.linspace(0, 2, 200)
    ax.plot(tt, np.exp(tt), color=FG, lw=1.5, ls=(0, (4, 3)), alpha=0.6, zorder=2)
    TA(ax, 1.55, 6.4, "$e^{t}$", size=16, color=SOFT, ha='right')
    ae = A(t, q('compare', 0.4), 0.9)
    if ae > 0.01:
        ax.plot(M.TAB_T, M.TAB_EULER, 'o-', color=BLUE, lw=1.6, ms=4, alpha=0.55 * ae, zorder=3)
        TA(ax, 1.62, 3.5, f"Euler, same step:\nshort by {F['tab_euler_pct']:.0f}%", size=14, color=BLUE, alpha=ae,
           linespacing=1.2)
    pp = lin(t, q('table', 0.0), q('table', 0.80)) if t > q('setup', 0.4) else 0.0
    poly_prog(ax, M.TAB_T, M.TAB_Y, max(pp, 0.001 if t > q('setup', 0.4) else 0.0), color=GREEN, lw=2.8, ms=6.0)
    ax.plot([0], [1], 'o', color=YELLOW, ms=10, zorder=8)
    ac = A(t, q('compare', 0.1), 0.9)
    if ac > 0.01:
        TA(ax, 1.95, 7.6, f"off by only {F['tab_err']:.3f}", size=16, color=GREEN, alpha=ac, ha='right')
    xs = [0.705, 0.775, 0.865, 0.945]
    rows = []
    for i in range(5):
        ta = q('table', 0.80 * i / 4.0 + 0.02) if i > 0 else q('setup', 0.6)
        rows.append((ta, [str(i), f"{M.TAB_T[i]:g}", f"{M.TAB_Y[i]:.4f}", f"{math.exp(M.TAB_T[i]):.3f}"]))
    hair_table(t, xs, 0.80, 0.062, rows, size=15, header=['$n$', '$t_n$', '$y_n$', '$e^{t_n}$'],
               header_t=q('setup', 0.5), aligns=['right'] * 4, colors=[SOFT, SOFT, GREEN, SOFT])
    margin(t, [
        dict(t=q('table'), kind='eq', size=16, h=0.06, text=r"$y_{n+1}=1.6484\,y_n$"),
        dict(t=q('table', 0.3), text="For $y'=y$ one step multiplies by $1+h+\\frac{h^2}{2}+\\frac{h^3}{6}+\\frac{h^4}{24}$."),
    ], top=0.455, x=0.705, width=34)


def scene7(t):
    q = lambda k, fr=0.0, off=0.0: Q(7, k, fr, off)
    header(7, "How the error shrinks")
    panels = [([0.075, 0.24, 0.40, 0.56], "smooth:  $y'=y$", M.ORD_N, M.ORD_LIN, (1e-13, 2.0), [-12, -9, -6, -3, 0], (3e-3, 1.6),
               {'euler': 1, 'midpoint': 2, 'rk4': 4}),
              ([0.545, 0.24, 0.40, 0.56], "kink at 0:  $y'=\\sqrt{t}$", M.ORD_N_SQRT, M.ORD_SQRT, (1e-7, 1.0), [-6, -4, -2, 0], (6e-4, 1.6),
               {'euler': 1, 'midpoint': 1.5, 'rk4': 1.5})]
    cols = {'euler': BLUE, 'midpoint': YELLOW, 'rk4': GREEN}
    names = {'euler': 'Euler', 'midpoint': 'midpoint', 'rk4': 'RK4'}
    for pi, (rect, ttl, Nn, errs, yr, yexp, xr, sl) in enumerate(panels):
        ax = fig.add_axes(rect)
        ax.set_xscale('log'); ax.set_yscale('log')
        xe = [-2, -1, 0] if pi == 0 else [-3, -2, -1, 0]
        style_ax(ax, xr, yr, [10.0 ** e for e in xe], [10.0 ** e for e in yexp],
                 xlabels=[pw(e, 0) for e in xe], ylabels=[pw(e, 0) for e in yexp], size=12)
        ax.set_xlabel('step size  $h$', fontsize=13, color=MUTED, labelpad=5)
        if pi == 0: ax.set_ylabel('error at  $t=1$', fontsize=13, color=MUTED, labelpad=3)
        ta = q('setup', 0.1) if pi == 0 else q('caveat', 0.1)
        TA(ax, 0.0 + (3.4e-3 if pi == 0 else 7e-4), yr[1] * 0.9, ttl, size=16, color=FG, alpha=A(t, ta, 0.8), va='top')
        H = 1.0 / np.asarray(Nn)
        for m in ('euler', 'midpoint', 'rk4'):
            if pi == 0:
                tm = {'euler': q('setup', 0.2), 'midpoint': q('setup', 0.55), 'rk4': q('rk4', 0.05)}[m]
                dur = 2.2
            else:
                tm = q('caveat', {'euler': 0.15, 'midpoint': 0.3, 'rk4': 0.45}[m]); dur = 1.8
            p = lin(t, tm, tm + dur)
            n = len(H) - 1; u = p * n; k = min(n, int(u))
            if p > 0:
                ax.plot(H[:k + 1], errs[m][:k + 1], 'o-', color=cols[m], lw=2.0, ms=4.5, zorder=5)
                if k < n:
                    fr = u - k
                    hx = math.exp(math.log(H[k]) + fr * (math.log(H[k + 1]) - math.log(H[k])))
                    ey = math.exp(math.log(errs[m][k]) + fr * (math.log(errs[m][k + 1]) - math.log(errs[m][k])))
                    ax.plot([H[k], hx], [errs[m][k], ey], color=cols[m], lw=2.0, zorder=5)
            if p >= 0.999:
                am = A(t, tm + dur, 0.7)
                if pi == 1 and m == 'rk4':
                    continue
                lab_s = f"{names[m]}: slope {sl[m]:g}" if not (pi == 1 and m == 'midpoint') else "midpoint and RK4: slope 1.5"
                ref = errs['rk4'][-1] if (pi == 1 and m == 'midpoint') else errs[m][-1]
                TA(ax, H[-1] * 1.5, ref * (0.15 if pi == 0 else 0.3), lab_s, size=14, color=cols[m], alpha=am, ha='left')
    T(0.075, 0.09, f"Logistic equation gives the same slopes ({F['sl_log_euler']:.2f}, {F['sl_log_midpoint']:.2f}, "
                    f"{F['sl_log_rk4']:.2f}).  Slopes measured over the finest step pair.", size=13, color=MUTED, style='italic',
      alpha=A(t, q('rk4', 0.6), 1.0))
    a = A(t, q('why', 0.1), 0.9)
    T(0.5, 0.855, "local error  $\\approx\\ h^5/120$  per step  →  $h^4$ over the interval", size=17, ha='center', color=FG,
      alpha=a * (1 - A(t, q('caveat', 0.0), 0.7)))


def scene8(t):
    q = lambda k, fr=0.0, off=0.0: Q(8, k, fr, off)
    header(8, "What accuracy costs")
    ax = fig.add_axes(AX_MAIN)
    ax.set_xscale('log'); ax.set_yscale('log')
    style_ax(ax, (0.8, 2e11), (1e-12, 3.0), [1, 1e2, 1e4, 1e6, 1e8, 1e10], [1e-12, 1e-9, 1e-6, 1e-3, 1],
             xlabels=[pw(e, 0) for e in (0, 2, 4, 6, 8, 10)], ylabels=[pw(e, 0) for e in (-12, -9, -6, -3, 0)], size=13)
    ax.set_xlabel('slope evaluations  (cost)', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('error at  $t=1$', fontsize=14, color=MUTED, labelpad=4)
    cols = {'euler': BLUE, 'midpoint': YELLOW, 'rk4': GREEN}
    names = {'euler': "Euler: 1 per step", 'midpoint': "midpoint: 2 per step", 'rk4': "RK4: 4 per step"}
    lab = {'euler': (3e3, 2.0e-1), 'midpoint': (1.5e3, 4.0e-6), 'rk4': (1.2, 6.0e-12)}
    t_start = {'euler': q('setup', 0.30), 'midpoint': q('setup', 0.50), 'rk4': q('setup', 0.70)}
    for m in ('euler', 'midpoint', 'rk4'):
        ev = M.WORK_N[m] * M.EVALS[m]; er = M.WORK_ERR[m]
        ok = er > 3e-13
        ev, er = ev[ok].astype(float), er[ok]
        p = lin(t, t_start[m], t_start[m] + 2.6)
        n = len(ev) - 1; u = p * n; k = min(n, int(u))
        if p > 0:
            ax.plot(ev[:k + 1], er[:k + 1], '-', color=cols[m], lw=2.4, zorder=5)
            if k < n:
                fr = u - k
                ex = math.exp(math.log(ev[k]) + fr * (math.log(ev[k + 1]) - math.log(ev[k])))
                ey = math.exp(math.log(er[k]) + fr * (math.log(er[k + 1]) - math.log(er[k])))
                ax.plot([ev[k], ex], [er[k], ey], color=cols[m], lw=2.4, zorder=5)
        lx, ly = lab[m]
        TA(ax, lx, ly, names[m], size=15, color=cols[m], alpha=A(t, t_start[m] + 1.5, 0.8))
    # equal budget of 100 evaluations
    ab = A(t, q('budget', 0.0), 0.8)
    ax.axvline(100, color=FG, lw=1.0, ls=(0, (2, 3)), alpha=0.6 * ab)
    TA(ax, 100, 1.2, "budget: 100", size=14, color=SOFT, alpha=ab, ha='center')
    for i, m in enumerate(('euler', 'midpoint', 'rk4')):
        am = A(t, q('budget', 0.22 + 0.28 * i), 0.8)
        if am > 0.01:
            v = F['bud'][m]['lin']
            ax.plot([100], [v], 'o', color=cols[m], ms=10, alpha=am, zorder=8, mec=BG, mew=1.2)
            TA(ax, 160, v * 0.95, sci(v, 1) if v < 1e-2 else f"{v:.3f}", size=15, color=cols[m], alpha=am, va='center')
    # target 1e-10
    at = A(t, q('target', 0.0), 0.8)
    ax.axhline(1e-10, color=FG, lw=1.0, ls=(0, (2, 3)), alpha=0.6 * at)
    TA(ax, 1.0, 1.7e-10, "error $=10^{-10}$", size=14, color=SOFT, alpha=at)
    txt = {'euler': "13.6 billion", 'midpoint': f"{F['ev_midpoint']:,}", 'rk4': f"{F['ev_rk4']}"}
    for i, m in enumerate(('euler', 'midpoint', 'rk4')):
        am = A(t, q('target', 0.25 + 0.25 * i), 0.8)
        if am > 0.01:
            ax.plot([F['ev_' + m]], [1e-10], 'o', color=cols[m], ms=10, alpha=am, zorder=8, mec=BG, mew=1.2)
            if m == 'euler': TA(ax, F['ev_' + m], 3.5e-11, txt[m], size=16, color=cols[m], alpha=am, ha='center', va='top')
            else: TA(ax, F['ev_' + m] / 1.5, 4.5e-11, txt[m], size=16, color=cols[m], alpha=am, ha='right', va='top')
    margin(t, [
        dict(t=q('setup'), text="Cost = number of times the equation's slope $f$ is evaluated. "
                                "RK4 pays four per step."),
        dict(t=q('budget'), kind='eq', size=16, h=0.06, text=r"100 evaluations $=$"),
        dict(t=q('budget', 0.1), text="Euler: 100 steps · midpoint: 50 steps · RK4: 25 steps."),
        dict(t=q('note'), kind='aside', text="One smooth problem, one end time. Evaluations are counted, not timed. "
                                             "The Euler figure comes from a closed form; the tests check it."),
    ], top=0.835, x=0.705, width=31)


def scene9(t):
    q = lambda k, fr=0.0, off=0.0: Q(9, k, fr, off)
    header(9, "How big a step is safe?")
    ax = fig.add_axes([0.075, 0.15, 0.55, 0.68])
    style_ax(ax, (0, 4), (0, 2.2), [0, 1, 2, 3, 4], [0, 0.5, 1, 1.5, 2], size=13)
    ax.set_xlabel(r'$\lambda h$', fontsize=16, color=MUTED, labelpad=5)
    ax.set_ylabel('size of the factor per step', fontsize=14, color=MUTED, labelpad=4)
    a0 = A(t, q('setup', 0.4), 0.9)
    ax.axhline(1, color=MUTED, lw=0.9, ls=(0, (2, 2)), alpha=0.8 * a0)
    TA(ax, 3.95, 1.05, "1: no growth", size=14, color=SOFT, alpha=a0, ha='right')
    z = M.STAB_Z
    cols = {'euler': BLUE, 'midpoint': YELLOW, 'rk4': GREEN}
    nm = {'euler': 'Euler', 'midpoint': 'midpoint', 'rk4': 'RK4'}
    tm = {'euler': q('setup', 0.5), 'midpoint': q('setup', 0.7), 'rk4': q('limits', 0.1)}
    for m in ('euler', 'midpoint', 'rk4'):
        p = lin(t, tm[m], tm[m] + 2.0)
        k = int(p * (len(z) - 1))
        if k > 1:
            y = M.amp(m, z[:k + 1])
            ax.plot(z[:k + 1], y, color=cols[m], lw=2.6 if m != 'midpoint' else 2.0, zorder=5 if m == 'rk4' else 4,
                    ls='-' if m != 'midpoint' else (0, (5, 2)))
    TA(ax, 0.1, 1.75, "Euler\n(midpoint: dashed,\nnearly the same)", size=14, color=BLUE,
       alpha=A(t, q('setup', 0.8), 0.8), linespacing=1.2, va='top')
    TA(ax, 3.4, 1.55, "RK4", size=17, color=GREEN, alpha=A(t, q('limits', 0.4), 0.8))
    al = A(t, q('limits', 0.35), 0.8)
    for zz, col, lab, dy in ((2.0, BLUE, "2", -0.13), (M.LIMIT['rk4'], GREEN, f"{M.LIMIT['rk4']:.2f}", -0.13)):
        ax.plot([zz], [1], 'o', color=col, ms=9, alpha=al, zorder=8, mec=BG, mew=1.1)
        TA(ax, zz, 1 + dy - 0.13, lab, size=16, color=col, alpha=al, ha='center')
    ax.axvspan(M.LIMIT['rk4'], 4, color=CORAL, alpha=0.07 * al, lw=0)
    # per-evaluation bars
    ap = A(t, q('per', 0.0), 0.9)
    T(0.705, 0.58, "stable  $\\lambda h$  per evaluation", size=15, color=SOFT, alpha=ap)
    hline(0.565, 0.705, 0.94, alpha=0.4 * ap, lw=0.5)
    for i, m in enumerate(('euler', 'midpoint', 'rk4')):
        a = A(t, q('per', 0.18 + 0.2 * i), 0.8)
        y = 0.49 - i * 0.095
        v = M.LIMIT_PER_EVAL[m]
        T(0.705, y + 0.027, nm[m], size=14, color=cols[m], alpha=a)
        fig.add_artist(Rectangle((0.705, y - 0.018), 0.10 * v, 0.032, transform=fig.transFigure, facecolor=cols[m],
                                 edgecolor='none', alpha=0.9 * a))
        T(0.705 + 0.10 * v + 0.012, y - 0.014, f"{v:.2f}".rstrip('0').rstrip('.') if m != 'rk4' else f"{v:.2f}", size=15,
          color=cols[m], alpha=a)
    margin(t, [
        dict(t=q('setup'), text="On $y'=-\\lambda y$ each step multiplies the solution by a fixed factor. "
                                "Above 1 it grows: unstable."),
    ], top=0.835, x=0.705, width=31)
    T(0.705, 0.17, "Four evaluations per step make\nthe stable step per evaluation smaller.",
      size=14, color=MUTED, style='italic', alpha=A(t, q('per', 0.7), 1.0), va='top', linespacing=1.25)


_SPR = {}


def spr_data():
    if not _SPR:
        tt, u = M.rk4_spring()
        _SPR['per'] = tt / (2 * math.pi)
        _SPR['rk'] = M.spr_energy(u) / M.spr_energy(u[0])
        x, v = M.leapfrog_spring()
        _SPR['lf'] = (x ** 2 + v ** 2)
    return _SPR


def scene10(t):
    q = lambda k, fr=0.0, off=0.0: Q(10, k, fr, off)
    header(10, "A long run on the spring")
    D = spr_data()
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, (0, 200), (0.5, 1.06), [0, 50, 100, 150, 200], [0.5, 0.75, 1.0], ylabels=['0.5', '0.75', '1'], size=13)
    ax.set_xlabel('periods', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('energy  $E/E_0$', fontsize=14, color=MUTED, labelpad=4)
    a0 = A(t, q('setup', 0.4), 0.9)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 2)), alpha=0.7 * a0)
    TA(ax, 199, 1.012, "exact: constant", size=13, color=SOFT, alpha=a0, ha='right')
    p1 = lin(t, q('rk4', 0.10), q('rk4', 0.90))
    k1 = int(p1 * (len(D['per']) - 1))
    if k1 > 1:
        ax.plot(D['per'][:k1 + 1:5], D['rk'][:k1 + 1:5], color=GREEN, lw=2.6, zorder=5)
        dot(ax, (D['per'][k1], D['rk'][k1]), GREEN, 8, z=7)
        TA(ax, 120, 0.795, "RK4", size=17, color=GREEN, alpha=A(t, q('rk4', 0.3), 0.8))
    if t > q('rk4', 0.92):
        a = A(t, q('rk4', 0.92), 0.8)
        TA(ax, 199, D['rk'][-1] - 0.06, f"{100 * D['rk'][-1]:.0f}% left", size=16, color=GREEN, alpha=a, ha='right')
    p2 = lin(t, q('lf', 0.05), q('lf', 0.85))
    k2 = int(p2 * (len(D['per']) - 1))
    if k2 > 1:
        ax.plot(D['per'][:k2 + 1], D['lf'][:k2 + 1], color=CORAL, lw=0.9, alpha=0.9, zorder=4)
        TA(ax, 100, 0.885, "leapfrog:  stays between 0.94 and 1", size=15, color=CORAL, alpha=A(t, q('lf', 0.3), 0.8))
    ae = A(t, q('setup', 0.7), 0.9)
    TA(ax, 4, 0.56, "Euler (×1.25 every step) is off the chart.", size=14, color=BLUE, style='italic', alpha=ae)
    margin(t, [
        dict(t=q('setup'), text=f"$\\ddot x=-x$, $h=0.5$, {M.SPR_PERIODS} periods = {M.SPR_N:,} steps."),
        dict(t=q('rk4'), kind='eq', size=15, h=0.07, text=r"$E_{n+1}/E_n=1-\frac{h^6}{72}+\frac{h^8}{576}$"),
        dict(t=q('rk4', 0.5), text=f"$={F['spr_rk4_factor']:.5f}$ per step: tiny, but it compounds."),
        dict(t=q('step'), kind='aside',
             text=f"At $h=0.25$ the same run keeps {100 * F['spr_rk4_h025']:.1f}% of the energy "
                  f"(computed from the factor)."),
    ], top=0.835, x=0.705, width=31)


def scene11(t):
    q = lambda k, fr=0.0, off=0.0: Q(11, k, fr, off)
    header(11, "Where Runge–Kutta sits")
    xs = [0.07, 0.285, 0.375, 0.50, 0.635, 0.80]
    al = ['left', 'center', 'center', 'center', 'center', 'center']
    hdr = ["method", "order", "slopes / step", "stable $\\lambda h<$", "stable $\\lambda h$ / evaluation", "error, 100 evaluations"]
    a = A(t, q('row1', 0.0), 0.9)
    for x, s, h in zip(xs, hdr, al): T(x, 0.82, s, size=14, color=MUTED, alpha=a, ha=h)
    hline(0.80, 0.07, 0.94, alpha=0.6 * a)
    b = F['bud']
    rows = [(q('row1', 0.15), "Euler", "1", "1", "2", "2", sci(b['euler']['lin'], 1) if b['euler']['lin'] < 1e-2 else f"{b['euler']['lin']:.3f}", BLUE),
            (q('row1', 0.45), "Midpoint", "2", "2", "2", "1", sci(b['midpoint']['lin'], 1), YELLOW),
            (q('row1', 0.80), "Runge–Kutta 4", "4", "4", f"{M.LIMIT['rk4']:.2f}", f"{M.LIMIT_PER_EVAL['rk4']:.2f}", sci(b['rk4']['lin'], 1), GREEN)]
    for i, (ta, name, order, sl, lim, per, err, col) in enumerate(rows):
        y = 0.725 - i * 0.085
        ar = A(t, ta, 0.8)
        vals = [name, order, sl, lim, per, err]
        for j, (x, v, h) in enumerate(zip(xs, vals, al)):
            T(x, y, v, size=21 if j == 0 else 19, color=col if j == 0 else FG, alpha=ar, ha=h)
        hline(y - 0.03, 0.07, 0.94, alpha=0.28 * ar, lw=0.5)
    lines = [(q('row2'), "Simpson's rule is the special case when the slope depends only on $t$: weights $1,\\ 4,\\ 1$ over 6."),
             (q('row3'), "On a kink ($\\sqrt{t}$) the order drops to 1.5. Stability limit: 2.79 against Euler's 2."),
             (q('row4'), "Adaptive solvers (RK45, the default in scipy's solve_ivp) compare a 4th and a 5th order answer to pick the step.")]
    for i, (ta, s) in enumerate(lines):
        T(0.07, 0.42 - i * 0.075, s, size=16, color=SOFT, alpha=A(t, ta, 0.9))
    ac = A(t, q('close', 0.1), 1.2)
    T(0.5, 0.115, "Euler follows one slope.  Runge–Kutta asks four questions and averages the answers.", size=21, ha='center',
      alpha=ac)


SCENES = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10, scene11]
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
