"""Scene drawing for the Euler-method video: a 3Blue1Brown x Tufte hybrid.

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


# ================================================================== SCENES
def scene0(t):
    q = lambda k, fr=0.0, off=0.0: Q(0, k, fr, off)
    t0 = q('hook', 0, 0.1)
    T(0.07, 0.585, "Euler's Method", size=42, alpha=A(t, t0, 1.2))
    T(0.07, 0.505, "Follow the slope, one small step at a time.", size=21, color=SOFT, style='italic',
      alpha=A(t, t0 + 1.0, 1.2), va='top')
    hline(0.685, 0.07, 0.55, alpha=0.7 * A(t, t0 + 0.6, 1.0))
    T(0.07, 0.24, "Numerical integration · part 3", size=14, color=MUTED, alpha=A(t, t0 + 2.2, 1.0))
    ax = fig.add_axes([0.62, 0.18, 0.32, 0.56])
    ax.set_facecolor('none'); ax.axis('off')
    ax.set_xlim(-0.1, 2.15); ax.set_ylim(-0.3, 8.2)
    tt = np.linspace(0, 2, 200)
    a = A(t, t0 + 0.4, 1.0)
    ax.plot(tt, np.exp(tt), color=SOFT, lw=1.4, ls=(0, (4, 3)), alpha=0.8 * a)
    poly_prog(ax, M.TAB_T, M.TAB_Y, lin(t, t0 + 0.8, t0 + 11.0), color=BLUE, lw=2.4, ms=5.5)
    ax.plot([0], [1], 'o', color=YELLOW, ms=8, zorder=8, alpha=a)


def scene1(t):
    q = lambda k, fr=0.0, off=0.0: Q(1, k, fr, off)
    header(1, "The equation gives a slope")
    tr, yr = (0, 3), (0, 6)
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, tr, yr, [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6], size=13)
    ax.set_xlabel('time  $t$', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('$y$', fontsize=15, color=MUTED, labelpad=4, rotation=0)
    segs = slope_ticks(ax, tr, yr, 13, 13)
    t_on = q('setup', 0.45)
    for (a0, b0, a1, b1, tc) in segs:
        al = 0.62 * A(t, t_on + 2.2 * tc / 3.0, 0.7)
        if al > 0.01: ax.plot([a0, a1], [b0, b1], color=SOFT, lw=1.5, alpha=al, solid_capstyle='round', zorder=2)
    a_s = A(t, q('start', 0.25), 0.7)
    if a_s > 0.01:
        ax.plot([0, 0.9], [1, 1.9], color=YELLOW, lw=3.0, alpha=a_s, zorder=5, solid_capstyle='round')
        ax.plot([0], [1], 'o', color=YELLOW, ms=11, zorder=7, alpha=a_s)
        TA(ax, 0.1, 1.55, "slope 1", size=16, color=YELLOW, alpha=A(t, q('start', 0.6), 0.7),
           bbox=dict(fc=BG, ec='none', pad=2.0, alpha=0.9))
        TA(ax, 0.12, 0.33, "start: $y(0)=1$", size=15, color=YELLOW, alpha=a_s,
           bbox=dict(fc=BG, ec='none', pad=2.0, alpha=0.9))
    a_j = A(t, q('judge', 0.3), 1.0)
    if a_j > 0.01:
        tt = np.linspace(0, math.log(6), 200)
        ax.plot(tt, np.exp(tt), color=FG, lw=1.6, ls=(0, (4, 3)), alpha=0.9 * a_j, zorder=4)
        TA(ax, 1.42, 5.4, "exact:  $e^{t}$", size=17, color=FG, alpha=a_j, ha='right',
           bbox=dict(fc=BG, ec='none', pad=2.0, alpha=0.9))
        TA(ax, 1.42, 4.8, "only to judge", size=14, color=SOFT, style='italic', alpha=a_j, ha='right',
           bbox=dict(fc=BG, ec='none', pad=2.0, alpha=0.9))
    margin(t, [
        dict(t=q('setup'), kind='eq', size=24, h=0.07, text=r"$y'=y$"),
        dict(t=q('setup', 0.3), text="At every point the equation says how steep the solution is there. "
                                    "It never says where the solution goes."),
        dict(t=q('start'), kind='eq', size=22, h=0.07, text=r"$y(0)=1$"),
        dict(t=q('judge'), kind='aside', text="Real problems rarely have a formula. The method must work "
                                              "from slopes alone."),
    ])


def scene2(t):
    q = lambda k, fr=0.0, off=0.0: Q(2, k, fr, off)
    header(2, "One step: follow the tangent")
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, (0, 1.5), (0, 3), [0, 0.5, 1, 1.5], [0, 1, 2, 3], xlabels=['0', '0.5', '1', '1.5'], size=13)
    ax.set_xlabel('time  $t$', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('$y$', fontsize=15, color=MUTED, labelpad=4, rotation=0)
    tt = np.linspace(0, math.log(3.0), 200)
    ax.plot(tt, np.exp(tt), color=FG, lw=1.5, ls=(0, (4, 3)), alpha=0.55, zorder=2)
    TA(ax, 1.08, 2.88, "$e^{t}$", size=16, color=SOFT, alpha=0.9)
    ax.plot([0], [1], 'o', color=YELLOW, ms=10, zorder=8)
    p = lin(t, q('tangent', 0.30), q('tangent', 0.85))
    if p > 0:
        ax.plot([0, p], [1, 1 + p], color=BLUE, lw=3.0, zorder=6, solid_capstyle='round')
        ax.plot([p], [1 + p], 'o', color=BLUE, ms=9, zorder=7)
    a = A(t, q('tangent', 0.55), 0.8)
    if a > 0.01:
        ax.plot([0, 1], [0.2, 0.2], color=MUTED, lw=1.0, alpha=a)
        for xx in (0, 1): ax.plot([xx, xx], [0.14, 0.26], color=MUTED, lw=1.0, alpha=a)
        TA(ax, 0.5, 0.27, "$h=1$", size=16, color=SOFT, alpha=a, ha='center')
        ax.plot([1, 1], [1, 2], color=MUTED, lw=1.0, ls=(0, (2, 2)), alpha=a)
        TA(ax, 1.03, 1.45, "rise $=h\\,y=1$", size=15, color=SOFT, alpha=a)
    a2 = A(t, q('miss', 0.18), 0.8)
    if a2 > 0.01:
        ax.plot([1], [math.e], 'o', color=FG, ms=9, zorder=8, alpha=a2)
        TA(ax, 0.94, math.e + 0.03, f"$e={math.e:.3f}$", size=17, ha='right', alpha=a2)
        ax.plot([1.0, 1.0], [2.0, math.e], color=CORAL, lw=3.2, alpha=a2, zorder=7, solid_capstyle='butt')
        TA(ax, 1.05, 2.35, f"miss $={F['one_err']:.3f}$", size=17, color=CORAL, alpha=A(t, q('miss', 0.45), 0.8))
        TA(ax, 1.05, 2.0, "$y_1=2$", size=15, color=BLUE, alpha=a2)
    margin(t, [
        dict(t=q('tangent'), text="At $(0,\\,1)$ the slope is $y'=y=1$. Walk along that line for a time $h$."),
        dict(t=q('formula'), kind='eq', size=19, h=0.075, text=r"$y_{n+1}=y_n+h\,f(t_n,y_n)$"),
        dict(t=q('formula', 0.5), kind='eq', size=19, h=0.07, text=r"$y_1=1+1\cdot1=2$"),
        dict(t=q('miss'), kind='aside', text="The tangent is the best straight-line guess, but the curve keeps "
                                             "turning upward after we leave the start."),
    ])


def scene3(t):
    q = lambda k, fr=0.0, off=0.0: Q(3, k, fr, off)
    header(3, "Four steps of one half")
    ax = fig.add_axes([0.075, 0.15, 0.56, 0.68])
    style_ax(ax, (0, 2), (0, 8), [0, 0.5, 1, 1.5, 2], [0, 2, 4, 6, 8], xlabels=['0', '0.5', '1', '1.5', '2'], size=13)
    ax.set_xlabel('time  $t$', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('$y$', fontsize=15, color=MUTED, labelpad=4, rotation=0)
    tt = np.linspace(0, 2, 200)
    ax.plot(tt, np.exp(tt), color=FG, lw=1.5, ls=(0, (4, 3)), alpha=0.6, zorder=2)
    TA(ax, 1.55, 6.3, "$e^{t}$", size=16, color=SOFT, ha='right')
    tp = lin(t, q('table', 0.0), q('table', 0.80))
    pp = tp if t >= q('table', 0.0) else 0.0
    poly_prog(ax, M.TAB_T, M.TAB_Y, max(pp, 0.001 if t > q('setup', 0.4) else 0.0), color=BLUE, lw=2.6, ms=6.0)
    ax.plot([0], [1], 'o', color=YELLOW, ms=10, zorder=8)
    a = A(t, q('under', 0.25), 0.9)
    if a > 0.01:
        y4 = M.TAB_Y[-1]
        ax.plot([2.04, 2.04], [y4, M.TAB_EXACT], color=CORAL, lw=3.2, alpha=a, solid_capstyle='butt', clip_on=False)
        TA(ax, 1.98, 7.0, f"short by {F['tab_pct']:.0f}%", size=17, color=CORAL, alpha=a, ha='right')
    # table
    xs = [0.705, 0.775, 0.865, 0.945]; al = ['right', 'right', 'right', 'right']
    rows = []
    for i in range(5):
        ta = q('table', 0.80 * i / 4.0 + 0.02) if i > 0 else q('setup', 0.6)
        cells = [str(i), f"{M.TAB_T[i]:g}", f"{M.TAB_Y[i]:.4f}".rstrip('0').rstrip('.') if i < 4 else f"{M.TAB_Y[i]:.4f}",
                 f"{math.exp(M.TAB_T[i]):.3f}"]
        rows.append((ta, cells))
    hair_table(t, xs, 0.80, 0.062, rows, size=15, header=['$n$', '$t_n$', '$y_n$', '$e^{t_n}$'],
               header_t=q('setup', 0.5), aligns=al, colors=[SOFT, SOFT, BLUE, SOFT])
    margin(t, [
        dict(t=q('table'), kind='eq', size=17, h=0.06, text=r"$y_{n+1}=y_n+0.5\,y_n=1.5\,y_n$"),
        dict(t=q('under'), kind='aside', text="Each tangent starts on the curve and ends below it, "
                                              "so every step falls short in the same direction."),
    ], top=0.455, x=0.705, width=34)


def scene4(t):
    q = lambda k, fr=0.0, off=0.0: Q(4, k, fr, off)
    header(4, "Smaller steps, closer to the curve")
    pos = [(0.075, 0.545), (0.375, 0.545), (0.075, 0.195), (0.375, 0.195)]
    cues = [0.24, 0.42, 0.60, 0.76]
    tt = np.linspace(0, 1, 100)
    for k, (h, (px, py)) in enumerate(zip(M.H_LIST, pos)):
        ax = fig.add_axes([px, py, 0.245, 0.265])
        style_ax(ax, (0, 1), (0, 3), [0, 0.5, 1], [0, 1, 2, 3], xlabels=['0', '0.5', '1'], size=12)
        a = A(t, q('multi', 0.02 + 0.03 * k), 0.8)
        ax.plot(tt, np.exp(tt), color=FG, lw=1.3, ls=(0, (4, 3)), alpha=0.55 * a)
        n = int(round(1 / h))
        tk, yk = M.euler(M.f_lin, 0.0, 1.0, h, n)
        poly_prog(ax, tk, yk, lin(t, q('multi', cues[k]), q('multi', cues[k], 1.3)),
                  color=BLUE, lw=2.2, ms=4.5 if n <= 10 else 3.0, dots=n <= 10)
        TA(ax, 0.04, 2.75, f"$h={h:g}$", size=17, color=FG, alpha=a)
        av = A(t, q('multi', cues[k], 1.5), 0.7)
        TA(ax, 0.62, 0.30, f"$y(1)={M.H_VAL[k][-1]:.4f}$", size=15, color=BLUE, alpha=av)
        ax.plot([1], [math.e], 'o', color=FG, ms=4, alpha=0.6 * a)
    T(0.075, 0.07, "Same problem, same end time $t=1$. Dashed: $e^{t}$, ending at $e=2.7183$.", size=14, color=MUTED,
      style='italic', alpha=A(t, q('multi', 0.1), 1.0))
    # limit table in the margin
    a_h = A(t, q('limit'), 0.8)
    nl = [1, 2, 4, 10, 100, 1000]
    rows = []
    for i, nn in enumerate(nl):
        rows.append((q('limit', 0.40 + 0.09 * i), [str(nn), f"{(1 + 1 / nn) ** nn:.4f}"]))
    rows.append((q('limit', 0.96), [r"$\to\infty$", f"{math.e:.4f}"]))
    margin(t, [
        dict(t=q('limit'), kind='eq', size=19, h=0.07, text=r"$y_n=(1+h)^n$"),
        dict(t=q('limit', 0.12), kind='eq', size=19, h=0.075, text=r"$y(1)=\left(1+\frac{1}{n}\right)^n$"),
    ], top=0.835, x=0.715)
    hair_table(t, [0.745, 0.89], 0.60, 0.055, rows, size=15, header=['$n$ steps', r'$y(1)$'], header_t=q('limit', 0.3),
               aligns=['right', 'right'], colors=[SOFT, BLUE])


def scene5(t):
    q = lambda k, fr=0.0, off=0.0: Q(5, k, fr, off)
    header(5, "How the error shrinks")
    ax = fig.add_axes(AX_MAIN)
    ax.set_xscale('log'); ax.set_yscale('log')
    style_ax(ax, (6e-4, 1.6), (1e-5, 2.0), [1e-3, 1e-2, 1e-1, 1], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
             xlabels=[pw(-3, 0), pw(-2, 0), pw(-1, 0), pw(0, 0)],
             ylabels=[pw(-5, 0), pw(-4, 0), pw(-3, 0), pw(-2, 0), pw(-1, 0), pw(0, 0)], size=13)
    ax.set_xlabel('step size  $h$', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('error at  $t=1$', fontsize=14, color=MUTED, labelpad=4)
    a0 = A(t, q('setup', 0.4), 1.0)
    # blue: y' = y
    H, EL, EG = M.ORD_H, M.ORD_LIN, M.ORD_LOG
    p = lin(t, q('slope', 0.0), q('slope', 0.75))
    n = len(H) - 1; u = p * n; k = int(u)
    if p > 0:
        ax.plot(H[:k + 1], EL[:k + 1], 'o-', color=BLUE, lw=2.0, ms=5.5, zorder=5)
        if k < n:
            fr = u - k
            hx = math.exp(math.log(H[k]) + fr * (math.log(H[k + 1]) - math.log(H[k])))
            ey = math.exp(math.log(EL[k]) + fr * (math.log(EL[k + 1]) - math.log(EL[k])))
            ax.plot([H[k], hx], [EL[k], ey], color=BLUE, lw=2.0, zorder=5)
    a1 = A(t, q('slope', 0.55), 0.8)
    if a1 > 0.01:
        x0, x1 = 0.03, 0.3; y0, y1 = 1.5e-4, 1.5e-3
        ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color=FG, lw=1.1, alpha=0.8 * a1, zorder=3)
        TA(ax, math.sqrt(x0 * x1), y0 * 0.55, "1 decade across", size=13, color=SOFT, alpha=a1, ha='center', va='top')
        TA(ax, x1 * 1.12, math.sqrt(y0 * y1), "1 decade up", size=13, color=SOFT, alpha=a1, va='center')
        TA(ax, math.sqrt(x0 * x1), 2.2e-5, "slope 1:  halve $h$, halve the error", size=15, color=FG, alpha=a1, ha='center')
        TA(ax, 7.5e-4, 5.5e-3, "$y'=y$", size=17, color=BLUE, alpha=a1)
    # green: logistic
    pl = lin(t, q('logistic', 0.30), q('logistic', 0.95))
    if pl > 0:
        k2 = int(pl * n)
        ax.plot(H[:k2 + 1], EG[:k2 + 1], 'o--', color=GREEN, lw=1.8, ms=4.5, zorder=4)
        TA(ax, 7.5e-4, 2.2e-4, "logistic: $y'=y(1-y)$", size=15, color=GREEN, alpha=A(t, q('logistic', 0.55), 0.8))
    margin(t, [
        dict(t=q('slope'), kind='eq', size=19, h=0.07, text=r"$E(h)\approx\frac{e}{2}\,h\approx %.2f\,h$" % F['err_over_h']),
        dict(t=q('logistic'), text="Nonlinear problem, exact solution known: also slope %.2f." % F['sl_log']),
        dict(t=q('local'), kind='eq', size=18, h=0.07, text=r"local error $\approx\frac{1}{2}\,y''\,h^2$"),
        dict(t=q('local', 0.5), text="$h=0.1$, one step from $(0,1)$: error %.4f, prediction %.4f. "
                                     "Then $1/h$ steps add up to order $h$." % (F['loc_err'], F['loc_pred'])),
        dict(t=q('bill'), kind='eq', size=18, h=0.07, text=r"error $<10^{-3}$: %d steps" % F['n_1e3']),
    ], top=0.835)


def scene6(t):
    q = lambda k, fr=0.0, off=0.0: Q(6, k, fr, off)
    header(6, "A second worry: stability")
    panels = [(0.1, 'small', [-1, 0, 1], (-1.4, 1.4), 0.665, BLUE), (0.3, 'mid', [-1, 0, 1], (-1.4, 1.4), 0.405, YELLOW),
              (0.5, 'big', [-10, 0, 10], (-14, 14), 0.145, CORAL)]
    tt = np.linspace(0, 3, 200)
    for i, (h, key, yt, yr, py, col) in enumerate(panels):
        ax = fig.add_axes([0.075, py, 0.49, 0.16])
        style_ax(ax, (0, 3), yr, [0, 1, 2, 3], yt, size=12)
        if i < 2: ax.tick_params(labelbottom=False)
        a = A(t, q('setup', 0.55), 0.9)
        ax.plot(tt, np.exp(-5 * tt), color=FG, lw=1.3, ls=(0, (4, 3)), alpha=0.6 * a)
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5 * a)
        tk, yk = M.STAB[h]
        poly_prog(ax, tk, yk, lin(t, q(key, 0.12), q(key, 0.12) + 3.2), color=col, lw=2.2, ms=5.0)
        fac = 1 - M.LAM * h
        TA(ax, 3.12, yr[1] * 0.45, f"$h={h:g}$", size=18, color=col, alpha=A(t, q(key, 0.0), 0.7))
        TA(ax, 3.12, yr[1] * -0.25, f"factor  {fmt(fac, 1)}  per step",
           size=14, color=SOFT, alpha=A(t, q(key, 0.35), 0.7))
        if i == 2: TA(ax, 0.1, 9.5, "note the different scale", size=13, color=CORAL, style='italic',
                      alpha=A(t, q(key, 0.2), 0.7))
        if i == 2:
            ax.set_xlabel('time  $t$', fontsize=14, color=MUTED, labelpad=4)
            e = M.STAB[0.5][1][-1]
            TA(ax, 3.12, -11.0, f"at $t=3$:  {e:.1f}", size=14, color=CORAL, alpha=A(t, q('big', 0.7), 0.7))
    margin(t, [
        dict(t=q('setup'), kind='eq', size=20, h=0.07, text=r"$y'=-5\,y$"),
        dict(t=q('setup', 0.5), text="Exact solution: $e^{-5t}$, dashed. Euler step: $y_{n+1}=(1-\\lambda h)\\,y_n$ with $\\lambda=5$."),
        dict(t=q('limit'), kind='eq', size=19, h=0.07, text=r"stable $\Leftrightarrow |1-\lambda h|<1$"),
        dict(t=q('limit', 0.4), kind='eq', size=19, h=0.07, text=r"$h<2/\lambda=%.1f$" % F['stab_limit']),
        dict(t=q('limit', 0.7), kind='aside', text="Accuracy told us how fast the error shrinks. "
                                                   "Stability says whether it shrinks at all."),
    ], top=0.835, x=0.765, width=22)


def scene7(t):
    q = lambda k, fr=0.0, off=0.0: Q(7, k, fr, off)
    header(7, "The amplification factor")
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, (0, 3), (0, 2), [0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 0.5, 1, 1.5, 2],
             xlabels=['0', '0.5', '1', '1.5', '2', '2.5', '3'], size=13)
    ax.set_xlabel(r'$\lambda h$', fontsize=16, color=MUTED, labelpad=5)
    ax.set_ylabel('size of the factor', fontsize=14, color=MUTED, labelpad=4)
    a0 = A(t, q('factor', 0.1), 0.9)
    ax.axhline(1, color=MUTED, lw=0.9, ls=(0, (2, 2)), alpha=0.8 * a0)
    TA(ax, 0.03, 1.04, "1  (no growth)", size=14, color=SOFT, alpha=a0)
    ax.axvspan(2, 3, ymin=0, ymax=1, color=CORAL, alpha=0.07 * a0, lw=0)
    TA(ax, 2.2, 1.85, "unstable", size=16, color=CORAL, ha='center', alpha=A(t, q('factor', 0.65), 0.8))
    z = np.linspace(0, 3, 301)
    p = lin(t, q('factor', 0.25), q('factor', 0.75))
    k = int(p * 300)
    if k > 1:
        ax.plot(z[:k + 1], np.abs(1 - z[:k + 1]), color=BLUE, lw=2.6, zorder=5)
        TA(ax, 1.12, 1.55, "forward Euler\n$|1-\\lambda h|$", size=15, color=BLUE, alpha=A(t, q('factor', 0.5), 0.8), linespacing=1.2)
    for zz, hh, tf in ((0.5, 0.1, 0.62), (1.5, 0.3, 0.72), (2.5, 0.5, 0.82)):
        am = A(t, q('factor', tf), 0.6)
        if am > 0.01:
            ax.plot([zz], [abs(1 - zz)], 'o', color=YELLOW, ms=9, zorder=7, alpha=am)
            if zz < 1: TA(ax, zz - 0.1, abs(1 - zz) - 0.07, f"$h={hh}$", size=14, color=YELLOW, ha='right', alpha=am)
            else: TA(ax, zz, abs(1 - zz) + (0.13 if zz < 2 else -0.22), f"$h={hh}$", size=14, color=YELLOW, ha='center', alpha=am)
    pb = lin(t, q('back', 0.22), q('back', 0.62))
    kb = int(pb * 300)
    if kb > 1:
        ax.plot(z[:kb + 1], 1 / (1 + z[:kb + 1]), color=GREEN, lw=2.6, zorder=5)
        TA(ax, 2.95, 0.92, "backward Euler\n$1/(1+\\lambda h)$", size=15, color=GREEN, alpha=A(t, q('back', 0.45), 0.8),
           linespacing=1.2, va='top', ha='right')
        ab = A(t, q('back', 0.78), 0.7)
        ax.plot([2.5], [1 / 3.5], 'o', color=GREEN, ms=8, alpha=ab, zorder=7)
        TA(ax, 2.5, 0.12, f"{F['be_fac_05']:.2f}", size=14, color=GREEN, ha='center', alpha=ab)
    margin(t, [
        dict(t=q('factor'), text="Forward Euler, applied to $y'=-\\lambda y$, multiplies each step by $1-\\lambda h$."),
        dict(t=q('factor', 0.5), text="Yellow dots: the three runs on the previous page, $\\lambda=5$."),
        dict(t=q('back'), text="Backward Euler takes the slope at the new point:"),
        dict(t=q('back', 0.1), kind='eq', size=18, h=0.07, text=r"$y_{n+1}=y_n+h\,f(t_{n+1},y_{n+1})$"),
        dict(t=q('back', 0.5), kind='eq', size=18, h=0.07, text=r"$y_{n+1}=\dfrac{y_n}{1+\lambda h}$"),
        dict(t=q('cost'), kind='aside', text="Same accuracy order, different stability. The new value sits on "
                                             "both sides, so each step needs a solve."),
    ], top=0.835, x=0.715, width=30)


def scene8(t):
    q = lambda k, fr=0.0, off=0.0: Q(8, k, fr, off)
    header(8, "Euler on a spring")
    L = 3.6; tk = [-3, -2, -1, 0, 1, 2, 3]
    ax = phase_ax([0.07, 0.10, 0.45, 0.76], L, tk)
    th = np.linspace(0, 2 * np.pi, 300)
    a0 = A(t, q('back', 0.45), 1.0)
    ax.plot(np.cos(th), np.sin(th), color=FG, lw=1.6, ls=(0, (4, 3)), alpha=0.9 * a0, zorder=2)
    TA(ax, 0, 0.0, "exact\ncircle", size=15, color=SOFT, style='italic', ha='center', va='center', alpha=a0,
       linespacing=1.15)
    ax.set_xlabel('position  $x$', fontsize=14, color=MUTED, labelpad=6, alpha=a0)
    ax.set_ylabel('velocity  $v$', fontsize=14, color=MUTED, labelpad=4, alpha=a0)
    xe, ve = M.spring_euler()
    Ee = 0.5 * (xe ** 2 + ve ** 2)
    ks = [q('spiral', 0.02), q('spiral', 0.93), q('why', 0.95)]
    step = float(np.interp(t, ks, [0, 31, 60]))
    idx = min(60, int(step)); fr = step - idx
    if t > ks[0]:
        xs = list(xe[:idx + 1]); ys = list(ve[:idx + 1])
        if idx < 60:
            xs.append(xe[idx] + fr * (xe[idx + 1] - xe[idx])); ys.append(ve[idx] + fr * (ve[idx + 1] - ve[idx]))
        trail(ax, np.array(xs), np.array(ys), 1.0, color=CORAL, lw=1.6, ms=3.2, dots=1)
        TA(ax, -3.45, 3.05, f"energy  ×{Ee[idx] / Ee[0]:.1f}", size=21, color=CORAL, alpha=A(t, ks[0], 0.6))
    if idx >= 31:
        a = A(t, ks[1], 0.6)
        ax.plot([xe[31]], [ve[31]], 'o', mfc='none', mec=FG, ms=11, mew=1.3, alpha=a, zorder=7)
        TA(ax, xe[31] + 0.25, ve[31] - 0.1, f"one period: ×{F['spr_E1p']:.1f}", size=15, alpha=a)
    # inset: energy ratio against the step number (log scale): a straight line = constant factor per step
    ai = fig.add_axes([0.605, 0.13, 0.33, 0.22])
    ai.set_yscale('log')
    style_ax(ai, (0, 60), (1, 14), [0, 20, 40, 60], [1, 2, 5, 10], ylabels=['1', '2', '5', '10'], size=12)
    ai.set_xlabel('step  $n$', fontsize=13, color=MUTED, labelpad=3)
    ai.set_ylabel('$E_n/E_0$', fontsize=13, color=MUTED, labelpad=3)
    ai.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ak = A(t, q('spiral', 0.5), 0.8)
    if t > ks[0]:
        m = min(60, idx)
        ai.plot(np.arange(m + 1), Ee[:m + 1] / Ee[0], '-', color=CORAL, lw=2.0, alpha=ak)
        ai.plot([m], [Ee[m] / Ee[0]], 'o', color=CORAL, ms=6, alpha=ak)
    TA(ai, 2, 8.0, "a straight line on a log scale:\nthe same factor every step", size=12.5, color=SOFT,
       style='italic', alpha=ak, linespacing=1.15)
    margin(t, [
        dict(t=q('back'), text="Mass on a spring, $\\ddot x=-x$, as a pair $(x,v)$ with $\\dot x=v$, $\\dot v=-x$."),
        dict(t=q('spiral', 0.1), kind='eq', size=17, h=0.065, text=r"$x_{n+1}=x_n+h\,v_n,\quad v_{n+1}=v_n-h\,x_n$"),
        dict(t=q('spiral', 0.4), kind='eq', size=18, h=0.065, text=r"$E_{n+1}=(1+h^2)\,E_n$"),
        dict(t=q('why'), kind='aside', text="Same one-sided error, now on a system that should conserve energy."),
    ], x=0.585, width=40, top=0.835)
    T(0.585, 0.40, f"$h=0.2$:  factor $1+h^2={F['spr_fac']:.2f}$ per step", size=14, color=MUTED, style='italic',
      alpha=A(t, q('spiral', 0.3), 0.8))


def scene9(t):
    q = lambda k, fr=0.0, off=0.0: Q(9, k, fr, off)
    header(9, "What does accuracy cost?")
    ax = fig.add_axes(AX_MAIN)
    ax.set_xscale('log'); ax.set_yscale('log')
    style_ax(ax, (0.8, 6e6), (1e-12, 3.0), [1, 1e2, 1e4, 1e6], [1e-12, 1e-9, 1e-6, 1e-3, 1],
             xlabels=[pw(0, 0), pw(2, 0), pw(4, 0), pw(6, 0)],
             ylabels=[pw(-12, 0), pw(-9, 0), pw(-6, 0), pw(-3, 0), pw(0, 0)], size=13)
    ax.set_xlabel('slope evaluations  (cost)', fontsize=14, color=MUTED, labelpad=6)
    ax.set_ylabel('error at  $t=1$', fontsize=14, color=MUTED, labelpad=4)
    curves = [('euler', BLUE, 'idea', 0.35, "Euler: 1 slope / step", 'left'),
              ('midpoint', YELLOW, 'mid', 0.45, "midpoint: 2 / step", 'left'),
              ('rk4', GREEN, 'rk4', 0.45, "RK4: 4 / step", 'left')]
    tol_a = A(t, q('count', 0.0), 0.8)
    ax.axhline(1e-6, color=FG, lw=1.0, ls=(0, (2, 3)), alpha=0.6 * tol_a)
    TA(ax, 1.0, 1.6e-6, "error $=10^{-6}$", size=14, color=SOFT, alpha=tol_a)
    marks = {'euler': (q('count', 0.30), 'ev_euler', "%s" % F['w_ev_euler'].replace('one point four million', '1.4 million')),
             'midpoint': (q('count', 0.55), 'ev_midpoint', f"{F['ev_midpoint']:,}"),
             'rk4': (q('count', 0.78), 'ev_rk4', f"{F['ev_rk4']}")}
    labpos = {'euler': (3.0e2, 3.5e-2, 'left'), 'midpoint': (4.0e3, 3.0e-9, 'left'), 'rk4': (3.0, 1.0e-10, 'left')}
    for name, col, key, f0, lab, _ in curves:
        ev = M.WORK_N[name] * M.EVALS[name]; er = M.WORK_ERR[name]
        ok = er > 3e-13
        ev, er = ev[ok], er[ok]
        p = lin(t, q(key, 0.1), q(key, 0.1) + 3.0)
        n = len(ev) - 1; u = p * n; k = int(u)
        if p > 0:
            ax.plot(ev[:k + 1], er[:k + 1], '-', color=col, lw=2.4, zorder=5)
            if k < n:
                fr = u - k
                ex = math.exp(math.log(ev[k]) + fr * (math.log(ev[k + 1]) - math.log(ev[k])))
                ey = math.exp(math.log(er[k]) + fr * (math.log(er[k + 1]) - math.log(er[k])))
                ax.plot([ev[k], ex], [er[k], ey], color=col, lw=2.4, zorder=5)
            lx, ly, ha = labpos[name]
            TA(ax, lx, ly, lab, size=15, color=col, alpha=A(t, q(key, 0.3), 0.8), ha=ha)
        tm, ek, txt = marks[name]
        am = A(t, tm, 0.7)
        if am > 0.01:
            xx = F[ek]
            ax.plot([xx], [1e-6], 'o', color=col, ms=10, zorder=8, alpha=am, mec=BG, mew=1.2)
            TA(ax, xx, 3.3e-7 if name != 'rk4' else 1.6e-7, txt, size=17, color=col, ha='center', va='top', alpha=am)
    margin(t, [
        dict(t=q('idea'), text="Euler spends one evaluation of $f$ per step."),
        dict(t=q('mid'), kind='eq', size=16, h=0.062, text=r"$k=f(t_n,y_n)$"),
        dict(t=q('mid', 0.1), kind='eq', size=16, h=0.075, text=r"$y_{n+1}=y_n+h\,f\!\left(t_n+\frac{h}{2},\,y_n+\frac{h}{2}k\right)$"),
        dict(t=q('rk4'), text="RK4 blends four slopes per step."),
        dict(t=q('caveat'), kind='aside', text="Smooth problem, error measured at one time, formulas evaluated "
                                               "exactly. Tests check them against real loops."),
    ], top=0.835, x=0.705, width=31)


def scene10(t):
    q = lambda k, fr=0.0, off=0.0: Q(10, k, fr, off)
    header(10, "Where Euler sits")
    xs = [0.07, 0.31, 0.405, 0.60]
    al = ['left', 'center', 'center', 'center']
    xnote = 0.67
    hdr = ["method", "order", "slopes per step", "stable while  $\\lambda h<$"]
    a = A(t, q('row1', 0.0), 0.9)
    for x, s, h in zip(xs, hdr, al):
        T(x, 0.80, s, size=15, color=MUTED, alpha=a, ha=h)
    T(xnote, 0.80, "in a phrase", size=15, color=MUTED, alpha=a)
    hline(0.783, 0.07, 0.94, alpha=0.6 * a)
    rows = [
        (q('row1', 0.05), "Euler (forward)", "1", "1", "2", "slope at the start", BLUE),
        (q('row2', 0.15), "Euler (backward)", "1", "1 + a solve", "any size", "slope at the end", GREEN),
        (q('row3', 0.10), "Midpoint", "2", "2", "2", "slope in the middle", YELLOW),
        (q('row3', 0.55), "Runge–Kutta 4", "4", "4", f"{F['lim_rk4']:.2f}", "blend of four slopes", YELLOW),
        (q('row4', 0.20), "Leapfrog", "2", "1", "—", "bounded energy on springs", CORAL),
    ]
    pitch = 0.108
    for i, (ta, name, order, sl, lim, note, col) in enumerate(rows):
        y = 0.715 - i * pitch
        ar = A(t, ta, 0.8)
        T(xs[0], y, name, size=21, color=col, alpha=ar)
        T(xs[1], y, order, size=21, alpha=ar, ha='center')
        T(xs[2], y, sl, size=19, alpha=ar, ha='center')
        T(xs[3], y, lim, size=19, alpha=ar, ha='center')
        T(xnote, y, note, size=17, color=SOFT, style='italic', alpha=ar)
        hline(y - 0.035, 0.07, 0.94, alpha=0.28 * ar, lw=0.5)
    T(0.07, 0.145, "Stability column: the decay test $y'=-\\lambda y$. Leapfrog is built for oscillators, "
                   "so the decay test does not apply.", size=14, color=MUTED, style='italic', alpha=A(t, q('row4', 0.5), 1.0))
    ac = A(t, q('close', 0.20), 1.2)
    T(0.5, 0.075, "Every improvement is a better answer to:  which slope should I follow?", size=22, ha='center',
      alpha=ac, color=FG)


SCENES = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10]
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
