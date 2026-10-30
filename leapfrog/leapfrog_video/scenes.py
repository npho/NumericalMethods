"""Scene drawing for the leapfrog video: a 3Blue1Brown x Tufte hybrid.

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




# ------------------------------------------------------------------ extra helpers for this video
AX_MAIN = [0.075, 0.15, 0.60, 0.68]
KD = M.KD


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


# ================================================================== SCENES
def scene0(t):
    q = lambda k, fr=0.0, off=0.0: Q(0, k, fr, off)
    t0 = q('hook', 0, 0.1)
    T(0.07, 0.585, "Leapfrog Integration", size=42, alpha=A(t, t0, 1.2))
    T(0.07, 0.505, "Simulating orbits that stay orbits.", size=21, color=SOFT, style='italic',
      alpha=A(t, t0 + 1.0, 1.2), va='top')
    hline(0.685, 0.07, 0.55, alpha=0.7 * A(t, t0 + 0.6, 1.0))
    T(0.07, 0.24, "Numerical integration · part 2", size=14, color=MUTED, alpha=A(t, t0 + 2.2, 1.0))
    ax = fig.add_axes([0.62, 0.18, 0.32, 0.56])
    ax.set_facecolor('none'); ax.axis('off')
    ax.set_xlim(-0.8, 1.8); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal', adjustable='box')
    rows = KD['orb_lf']
    p = lin(t, t0 + 0.8, t0 + 14.0)
    trail(ax, rows[:, 1], rows[:, 2], p, color=BLUE, lw=2.2, ms=3.0, dots=8)
    ax.plot([0], [0], 'o', color=YELLOW, ms=11, zorder=6, alpha=A(t, t0 + 0.5, 1.0))


def scene1(t):
    q = lambda k, fr=0.0, off=0.0: Q(1, k, fr, off)
    header(1, "Why not just step forward?")
    L = 3.6; tk = [-3, -2, -1, 0, 1, 2, 3]
    ax = fig.add_axes([0.07, 0.10, 0.45, 0.76])
    style_ax(ax, (-L, L), (-L, L), tk, tk, size=12)
    ax.set_aspect('equal', adjustable='box')
    th = np.linspace(0, 2 * np.pi, 300)
    a0 = A(t, q('setup', 0.45), 1.0)
    ax.plot(np.cos(th), np.sin(th), color=FG, lw=1.6, ls=(0, (4, 3)), alpha=0.9 * a0, zorder=2)
    TA(ax, 0, 0.0, "exact\ncircle", size=15, color=SOFT, style='italic', ha='center', va='center',
       alpha=a0, linespacing=1.15)
    ax.set_xlabel('position  $x$', fontsize=14, color=MUTED, labelpad=6, alpha=a0)
    ax.set_ylabel('velocity  $v$', fontsize=14, color=MUTED, labelpad=4, alpha=a0)
    # Euler path, scheduled so that step 31 (one period) and step 60 (t = 12) land on their narration
    xe, ve = M.ho_euler(0.2, 60)
    Ee = M.ho_energy(xe, ve)
    ks = [q('euler', 0.55), q('grow', 0.42), q('grow', 0.92)]
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
        TA(ax, xe[31] + 0.25, ve[31] - 0.1, f"one period: ×{F['eul_E1p']:.1f}", size=15, alpha=a)
    if idx >= 60:
        a = A(t, ks[2], 0.6)
        ax.plot([xe[60]], [ve[60]], 'o', mfc='none', mec=FG, mew=1.3, ms=11, alpha=a, zorder=7)
        right = xe[60] > 1.2
        TA(ax, xe[60] + (-0.25 if right else 0.25), ve[60] - 0.55, f"t = 12: ×{F['eul_E12']:.1f}", size=15,
           alpha=a, ha='right' if right else 'left')
    margin(t, [
        dict(t=q('setup'), text="A mass on a spring, $\\ddot x=-x$. The energy $E=\\frac{1}{2}v^2+\\frac{1}{2}x^2$ "
                                "never changes, so the motion is a circle."),
        dict(t=q('euler'), kind='eq', size=17, h=0.06, text=r'$x_{n+1}=x_n+v_n\,\Delta t$'),
        dict(t=q('euler', 0.1), kind='eq', size=17, h=0.07, text=r'$v_{n+1}=v_n-x_n\,\Delta t$'),
        dict(t=q('grow'), text=f"Each step multiplies the energy by exactly $1+\\Delta t^2={F['eul_fac']:.2f}$."),
        dict(t=q('why'), kind='aside', text="Errors that all push the same way pile up. Errors that cancel stay small."),
    ], x=0.585, width=40)
    T(0.585, 0.065, f"Euler's method, $\\Delta t={F['eul_dt']}$.  Position across, velocity up.", size=14, color=MUTED,
      style='italic', alpha=A(t, q('euler'), 0.8))


def scene2(t):
    q = lambda k, fr=0.0, off=0.0: Q(2, k, fr, off)
    header(2, "The idea: stagger the grid")
    ax = fig.add_axes([0.05, 0.12, 0.62, 0.74])
    ax.set_facecolor('none'); ax.axis('off')
    ax.set_xlim(-1.15, 5.0); ax.set_ylim(-0.12, 1.0)
    YX, YV = 0.74, 0.30
    ta = A(t, q('stagger', 0.0), 1.0)
    ax.plot([-0.1, 4.6], [0.02, 0.02], color=MUTED, lw=0.9, alpha=ta)
    for n in range(5):
        ax.plot([n, n], [0.02, 0.055], color=MUTED, lw=0.9, alpha=ta)
        TA(ax, n, -0.07, rf'$t_{n}$', size=14, color=MUTED, ha='center', va='center', alpha=ta)
    TA(ax, -1.1, YX, "positions", size=16, color=BLUE, va='center', alpha=ta)
    TA(ax, -1.1, YV, "velocities", size=16, color=YELLOW, va='center', alpha=ta)
    order = [('x', 0), ('v', 0), ('x', 1), ('v', 1), ('x', 2), ('v', 2), ('x', 3), ('v', 3), ('x', 4)]
    for j, (kind, n) in enumerate(order):
        a = A(t, q('stagger', 0.18 + 0.075 * j), 0.5)
        if a < 0.01: continue
        if kind == 'x':
            ax.plot([n], [YX], 'o', color=BLUE, ms=11, alpha=a, zorder=5)
            TA(ax, n, YX + 0.075, rf'$x_{n}$', size=17, color=BLUE, ha='center', alpha=a)
        else:
            ax.plot([n + .5], [YV], 'o', color=YELLOW, ms=11, alpha=a, zorder=5)
            lab = rf'$v_{{{2 * n + 1}/2}}$'
            TA(ax, n + .5, YV - 0.115, lab, size=17, color=YELLOW, ha='center', va='top', alpha=a)

    def kick(n, prog, alpha=1.0, label=False):          # v_{n-1/2} -> v_{n+1/2}, uses x_n
        if prog <= 0.01: return
        p0, p1, c = (n - .5, YV), (n + .5, YV), (n, YV + 0.34)
        arc(ax, p0, c, p1, prog, YELLOW, alpha=alpha)
        ax.plot([n, n], [YX - 0.03, YV + 0.17], color=YELLOW, lw=1.2, ls=(0, (2, 2)),
                alpha=0.9 * alpha * clamp(prog * 3))
        if label: TA(ax, n + 0.07, 0.535, rf'$a(x_{n})$', size=15, color=YELLOW, alpha=alpha * clamp(prog * 2))

    def drift(n, prog, alpha=1.0, label=False):         # x_n -> x_{n+1}, uses v_{n+1/2}
        if prog <= 0.01: return
        p0, p1, c = (n, YX), (n + 1, YX), (n + .5, YX - 0.34)
        arc(ax, p0, c, p1, prog, BLUE, alpha=alpha)
        ax.plot([n + .5, n + .5], [YV + 0.03, YX - 0.17], color=BLUE, lw=1.2, ls=(0, (2, 2)),
                alpha=0.9 * alpha * clamp(prog * 3))
        if label: TA(ax, n + 0.57, 0.485, rf'$v_{{{2 * n + 1}/2}}$', size=15, color=BLUE, alpha=alpha * clamp(prog * 2))

    kick(1, A(t, q('kick', 0.35), 1.2), label=True)
    drift(1, A(t, q('drift', 0.35), 1.2), label=True)
    seq = [('d', 0), ('k', 2), ('d', 2), ('k', 3), ('d', 3)]          # the rest hop along in order
    for j, (kind, n) in enumerate(seq):
        pr = A(t, q('leap', 0.18 + 0.15 * j), 0.9)
        (kick if kind == 'k' else drift)(n, pr)
    ts = q('start', 0.25)                                              # the half kick that starts it all
    a = A(t, ts, 0.8)
    if a > 0.01:
        ax.plot([0], [YV], 'o', mfc=BG, mec=YELLOW, mew=1.8, ms=11, alpha=a, zorder=5)
        TA(ax, 0, YV - 0.115, r'$v_0$', size=17, color=YELLOW, ha='center', va='top', alpha=a)
        arc(ax, (0, YV), (0.25, YV + 0.19), (0.5, YV), A(t, ts + 0.5, 1.0), YELLOW, ls=(0, (3, 2)))
        ax.plot([0, 0], [YX - 0.03, YV + 0.05], color=YELLOW, lw=1.2, ls=(0, (2, 2)), alpha=0.9 * a)
        TA(ax, 0.08, 0.52, r'$a(x_0)$', size=15, color=YELLOW, alpha=a)
    margin(t, [
        dict(t=q('stagger'), text="Positions live at whole steps $t_n$. Velocities live halfway, at $t_{n+1/2}$."),
        dict(t=q('kick'), kind='eq', size=17, h=0.06, color=YELLOW,
             text=r'$v_{n+\frac{1}{2}}=v_{n-\frac{1}{2}}+a(x_n)\,\Delta t$'),
        dict(t=q('kick', 0.2), kind='aside', text="kick: use the force at the middle."),
        dict(t=q('drift'), kind='eq', size=17, h=0.06, color=BLUE, text=r'$x_{n+1}=x_n+v_{n+\frac{1}{2}}\,\Delta t$'),
        dict(t=q('drift', 0.2), kind='aside', text="drift: use the velocity at the middle."),
        dict(t=q('leap'), text="Each variable leaps over the other."),
        dict(t=q('start'), kind='eq', size=16, h=0.06, text=r'$v_{\frac{1}{2}}=v_0+\frac{1}{2}\,a(x_0)\,\Delta t$'),
    ], x=0.70, width=31)


def scene3(t):
    q = lambda k, fr=0.0, off=0.0: Q(3, k, fr, off)
    header(3, "Every update is a midpoint rule")
    xl = [0.07, 0.54]; pw = 0.38
    tt = np.linspace(-0.15, 1.15, 300)
    ax = fig.add_axes([xl[0], 0.30, pw, 0.46])                         # ---- left: position update
    style_ax(ax, (-0.15, 1.15), (-1.0, 0.08), [0, 0.5, 1], [-1, -0.5, 0],
             ['$t_n$', r'$t_{n+\frac{1}{2}}$', '$t_{n+1}$'], ['−1', '−0.5', '0'], size=13)
    ca = A(t, q('callback', 0.2), 1.0)
    ax.plot(tt, -np.sin(tt), color=FG, lw=2.3, zorder=3, alpha=ca)
    TA(ax, 0.10, -0.94, r'$v(t)=-\sin t$', size=16, alpha=ca, va='center')
    T(xl[0], 0.84, "Position update", size=19, alpha=ca)
    T(xl[0], 0.805, r'$\Delta x=\int v\,dt$', size=16, color=SOFT, alpha=ca)
    r0 = q('rect', 0.12)
    sx = np.linspace(0, 1, 200)
    ax.fill_between(sx, -np.sin(sx), 0, color=BLUE, alpha=0.30 * A(t, r0, 1.0), lw=0, zorder=1)
    ar = A(t, r0 + 1.2, 1.0)                                           # ghost right rectangle (left has height 0)
    ax.add_patch(Rectangle((0, F['mid_right']), 1, -F['mid_right'], fill=False, ec=to_rgba(SOFT, 0.8 * ar),
                           lw=1.3, ls=(0, (3, 2)), zorder=2))
    am = A(t, r0 + 2.4, 1.0)
    ax.add_patch(Rectangle((0, F['mid_mid']), 1, -F['mid_mid'], fc=to_rgba(YELLOW, 0.07 * am),
                           ec=to_rgba(YELLOW, am), lw=2.0, zorder=4))
    ax.plot([0.5, 0.5], [0, -np.sin(0.5)], color=YELLOW, lw=1.2, ls=(0, (2, 2)), alpha=am, zorder=5)
    ax.plot([0.5], [-np.sin(0.5)], 'o', color=YELLOW, ms=8, alpha=am, zorder=6)
    TA(ax, 0.02, 0.025, "left: height 0", size=13, color=SOFT, alpha=ar)
    TA(ax, 0.03, -0.81, "right: height at $t_{n+1}$", size=13, color=SOFT, alpha=ar, va='bottom')
    y0 = 0.205
    T(xl[0], y0, rf"exact  $\cos 1-\cos 0={fmt(F['mid_exact'])}$", size=16, color=BLUE, alpha=A(t, r0 + 0.6, 0.8))
    T(xl[0], y0 - 0.045, rf"middle  $-\sin\frac{{1}}{{2}}={fmt(F['mid_mid'])}$   ({F['mid_pct']:.1f}% off)", size=16,
      color=YELLOW, alpha=A(t, r0 + 2.8, 0.8))
    T(xl[0], y0 - 0.09, f"left 0  ({F['left_pct']:.0f}% off),  right {fmt(F['mid_right'])}  ({F['right_pct']:.0f}% off)",
      size=14, color=SOFT, alpha=A(t, r0 + 1.6, 0.8))
    b0 = q('both', 0.05)                                               # ---- right: velocity update
    ab = A(t, b0, 1.0)
    ax2 = fig.add_axes([xl[1], 0.30, pw, 0.46])
    style_ax(ax2, (0.35, 1.65), (-1.0, 0.08), [0.5, 1, 1.5], [-1, -0.5, 0],
             [r'$t_{n-\frac{1}{2}}$', '$t_n$', r'$t_{n+\frac{1}{2}}$'], ['−1', '−0.5', '0'], size=13)
    for s in ('left', 'bottom'): ax2.spines[s].set_alpha(ab)
    ax2.tick_params(labelcolor=to_rgba(MUTED, ab), color=to_rgba(MUTED, ab))
    t2 = np.linspace(0.35, 1.65, 300)
    ax2.plot(t2, -np.cos(t2), color=FG, lw=2.3, zorder=3, alpha=ab)
    TA(ax2, 1.62, -0.88, r'$a(t)=-\cos t$', size=16, alpha=ab, ha='right', va='center')
    T(xl[1], 0.84, "Velocity update", size=19, alpha=ab)
    T(xl[1], 0.805, r'$\Delta v=\int a\,dt$', size=16, color=SOFT, alpha=ab)
    s2 = np.linspace(0.5, 1.5, 200)
    ax2.fill_between(s2, -np.cos(s2), 0, color=BLUE, alpha=0.30 * A(t, b0 + 1.2, 1.0), lw=0, zorder=1)
    am2 = A(t, b0 + 3.0, 1.0)
    ax2.add_patch(Rectangle((0.5, F['acc_mid']), 1, -F['acc_mid'], fc=to_rgba(YELLOW, 0.07 * am2),
                            ec=to_rgba(YELLOW, am2), lw=2.0, zorder=4))
    ax2.plot([1, 1], [0, F['acc_mid']], color=YELLOW, lw=1.2, ls=(0, (2, 2)), alpha=am2, zorder=5)
    ax2.plot([1], [F['acc_mid']], 'o', color=YELLOW, ms=8, alpha=am2, zorder=6)
    T(xl[1], y0, rf"exact  {fmt(F['acc_exact'])}", size=16, color=BLUE, alpha=A(t, b0 + 1.5, 0.8))
    T(xl[1], y0 - 0.045, rf"middle  $-\cos 1={fmt(F['acc_mid'])}$   ({F['acc_pct']:.1f}% off)", size=16, color=YELLOW,
      alpha=A(t, b0 + 3.4, 0.8))
    T(0.07, 0.062, "A step of size 1 is exaggerated on purpose, so the rectangles are visible.", size=14, color=SOFT,
      alpha=A(t, q('both', 0.7), 1.0))


def scene4(t):
    q = lambda k, fr=0.0, off=0.0: Q(4, k, fr, off)
    header(4, "A worked example")
    dt, X, VH = M.TAB_DT, M.TAB_X, M.TAB_VH
    ax = fig.add_axes(AX_MAIN)
    style_ax(ax, (-0.05, 2.5), (-0.62, 1.12), [0, 0.5, 1, 1.5, 2], [-0.5, 0, 0.5, 1],
             ['0', '0.5', '1', '1.5', '2'], ['−0.5', '0', '0.5', '1'])
    tt = np.linspace(0, 2, 300)
    a0 = A(t, q('setup', 0.2), 1.0)
    ax.plot(tt, np.cos(tt), color=FG, lw=1.8, alpha=0.8 * a0, zorder=2)
    TA(ax, 0.9, 0.82, r'exact: $\cos t$', size=16, ha='left', alpha=a0, color=FG)
    TA(ax, 2.46, -0.58, r'$t$', size=16, color=MUTED, ha='right', alpha=a0)
    rt = [q('setup', 0.6), q('step', 0.30), q('step', 0.55), q('step', 0.80), q('table', 0.30)]
    for n in range(5):
        a = A(t, rt[n], 0.7)
        if a < 0.01: continue
        if n > 0:
            ax.plot([(n - 1) * dt, n * dt], [X[n - 1], X[n]], color=YELLOW, lw=1.6, alpha=a * 0.9, zorder=4)
        ax.plot([n * dt], [X[n]], 'o', color=YELLOW, ms=9, alpha=a, zorder=5)
    ac = A(t, q('compare', 0.12), 1.0)
    if ac > 0.01:
        ex = math.cos(2.0)
        ax.plot([2.0, 2.0], [X[4], ex], color=CORAL, lw=2.0, alpha=ac, zorder=6)
        TA(ax, 2.06, (X[4] + ex) / 2, f"off by {F['tab_err']:.3f}", size=15, color=CORAL, ha='left', va='center', alpha=ac)
    X0 = 0.715; xc = [X0, X0 + 0.055, X0 + 0.145]; ty = 0.80             # table
    th = A(t, q('setup', 0.4), 0.8)
    T(xc[0], ty, '$n$', size=16, color=MUTED, alpha=th)
    T(xc[1], ty, '$x_n$', size=16, color=MUTED, alpha=th)
    T(xc[2], ty, r'$v_{n+\frac{1}{2}}$', size=16, color=MUTED, alpha=th)
    hline(ty - 0.018, X0, 0.95, alpha=0.5 * th)
    vt = [q('half', 0.55), q('step', 0.40), q('step', 0.65), q('step', 0.90), None]
    for n in range(5):
        yy = ty - 0.065 - 0.062 * n
        T(xc[0], yy, f'{n}', size=17, alpha=A(t, rt[n], 0.7))
        T(xc[1], yy, fmt(X[n]), size=17, color=YELLOW if n == 4 else FG, alpha=A(t, rt[n], 0.7))
        if vt[n] is not None:
            T(xc[2], yy, fmt(VH[n]), size=17, alpha=A(t, vt[n], 0.7), color=SOFT)
    ya = ty - 0.065 - 0.062 * 5 - 0.045                                  # the arithmetic of the current step
    ah = A(t, q('half', 0.5), 0.8) * (1 - A(t, q('step', 0.0), 0.4))
    T(X0, ya, r'$v_{\frac{1}{2}}=0+\frac{1}{2}(-1)(0.5)=-0.25$', size=16, color=YELLOW, alpha=ah)
    ad = A(t, q('step', 0.0, 0.5), 0.6) * (1 - A(t, q('table', 0.2), 0.6))
    T(X0, ya, r'$x_1=1+(-0.25)(0.5)=0.875$', size=16, color=BLUE, alpha=ad)
    T(X0, ya - 0.055, r'$v_{\frac{3}{2}}=-0.25+(-0.875)(0.5)$', size=16, color=YELLOW, alpha=ad)
    T(X0, ya - 0.105, r'$\;\;=-0.6875$', size=16, color=YELLOW, alpha=ad)
    T(X0, ya, rf"cos 2 $={fmt(F['tab_cos2'])}$", size=17, color=SOFT, alpha=A(t, q('compare', 0.1), 0.8))
    T(0.06, 0.062, "Δt = 0.5, starting from $x_0=1$, $v_0=0$. Each row uses only the row before it: "
                   "one force evaluation per step.", size=14, color=SOFT, alpha=A(t, q('setup', 0.9), 1.0))


def scene5(t):
    q = lambda k, fr=0.0, off=0.0: Q(5, k, fr, off)
    header(5, "How accurate is it?")
    ax = fig.add_axes(AX_MAIN)
    ax.set_xscale('log'); ax.set_yscale('log')
    style_ax(ax, (0.0011, 0.32), (2e-11, 6), [0.01, 0.1], [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1],
             ['0.01', '0.1'], [r'$10^{-10}$', r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', '1'])
    ax.spines['bottom'].set_bounds(0.01, 0.1)
    a0 = A(t, q('setup', 0.3), 1.0)
    TA(ax, 0.31, 3.5e-11, r'step size $\Delta t$ (larger $\rightarrow$)', size=14, color=MUTED, ha='right', alpha=a0)
    TA(ax, 0.0012, 3.3, r'error in $x$ at $t=10$', size=14, color=MUTED, alpha=a0, va='bottom')
    dts = M.ORD_DT
    specs = [('euler', 'Euler', CORAL, 'euler', 'slope %.2f' % F['sl_euler']),
             ('leapfrog', 'Leapfrog', YELLOW, 'lf', 'slope %.2f' % F['sl_lf']),
             ('rk4', 'RK4', GREEN, 'rk4', 'slope %.2f' % F['sl_rk4'])]
    for name, label, col, key, sl in specs:
        p = lin(t, q(key, 0.05), q(key, 0.65))
        if p <= 0: continue
        k = int(round(p * (len(dts) - 1))) + 1
        ax.plot(dts[:k], M.ORD_ERR[name][:k], '-o', color=col, lw=2.2, ms=6, zorder=4)
        if k == len(dts):
            la = A(t, q(key, 0.7), 0.8)
            TA(ax, dts[-1] * 0.82, M.ORD_ERR[name][-1], f"{label}\n{sl}", size=16, color=col, ha='right', va='center',
               alpha=la, linespacing=1.15)
    a2 = A(t, q('lf', 0.8), 1.0)
    if a2 > 0.01:                                                        # halve the step, error / 4
        e1, e2 = M.ORD_ERR['leapfrog'][2], M.ORD_ERR['leapfrog'][3]
        d1, d2 = dts[2], dts[3]
        ax.plot([d1, d2, d2], [e1, e1, e2], color=FG, lw=1.0, alpha=0.9 * a2, zorder=6)
        TA(ax, d2 * 0.95, e1 * 2.6, f"halve $\\Delta t$:\nerror ÷{e1 / e2:.2f}", size=15, ha='right', alpha=a2,
           linespacing=1.15)
    T(0.075, 0.055, "Slopes are measured over the two smallest steps; they approach the nominal orders 1, 2 and 4.",
      size=13, color=MUTED, style='italic', alpha=A(t, q('rk4', 0.8), 1.0))
    margin(t, [
        dict(t=q('setup'), text="Error in $x$ at $t=10$, for the spring, as the step shrinks."),
        dict(t=q('euler'), kind='eq', size=17, h=0.06, color=CORAL, text=r'Euler: $E\propto\Delta t$'),
        dict(t=q('lf'), kind='eq', size=17, h=0.06, color=YELLOW, text=r'Leapfrog: $E\propto\Delta t^{2}$'),
        dict(t=q('rk4'), kind='eq', size=17, h=0.06, color=GREEN, text=r'RK4: $E\propto\Delta t^{4}$'),
        dict(t=q('rk4', 0.3), text="…but RK4 needs 4 force evaluations per step. Leapfrog needs 1."),
        dict(t=q('honest'), kind='aside', text="On a short run, higher order wins. Hold that thought."),
    ], x=0.715, width=29)


def scene6(t):
    q = lambda k, fr=0.0, off=0.0: Q(6, k, fr, off)
    header(6, "Run it backwards")
    L = 3.6; tk = [-3, -2, -1, 0, 1, 2, 3]
    panels = [('euler', 'Euler', CORAL, 0.07), ('leapfrog', 'Leapfrog', YELLOW, 0.54)]
    a_set = A(t, q('setup', 0.3), 1.0)
    for method, name, col, x0 in panels:
        key = 'euler' if method == 'euler' else 'lf'
        ax = phase_ax([x0, 0.225, 0.40, 0.55], L, tk, size=11)
        (xf, vf), (xb, vb), dist = M.reversal(method)
        n = len(xf)
        fs = lin(t, q(key, 0.05), q(key, 0.38))                          # forward leg
        bs = lin(t, q(key, 0.50), q(key, 0.88))                          # backward leg
        T(x0, 0.84, name, size=20, color=col, alpha=a_set)
        T(x0, 0.803, f"{M.REV_N} steps, flip $v$, {M.REV_N} steps", size=14, color=MUTED, style='italic', alpha=a_set)
        if fs > 0: trail(ax, xf, vf, fs, color=BLUE, lw=1.7, ms=3.2, dots=1, tip=(bs <= 0))
        if bs > 0:
            kk = int(round(bs * (n - 1))) + 1
            ax.plot([xf[-1]], [vf[-1]], 'o', mfc=BG, mec=FG, mew=1.4, ms=10, zorder=7)
            ax.plot(xb[:kk], vb[:kk], color=col, lw=2.0, ls=(0, (4, 2)), zorder=6)
            ax.plot([xb[kk - 1]], [vb[kk - 1]], 'o', color=col, ms=7, zorder=8)
        ax.plot([xf[0]], [vf[0]], 's', color=FG, ms=8, zorder=9, alpha=a_set)
        TA(ax, xf[0] + 0.12, vf[0] + 0.22, 'start', size=14, color=FG, alpha=a_set)
        ae = A(t, q(key, 0.9), 0.8)
        if method == 'euler':
            ax.plot([xf[0], xb[-1]], [vf[0], vb[-1]], color=CORAL, lw=1.2, alpha=ae)
            T(x0, 0.105, f"ends {dist:.1f} away from the start", size=18, color=CORAL, alpha=ae)
        else:
            T(x0, 0.105, f"ends {sci(dist)} from the start", size=18, color=YELLOW, alpha=ae)
    T(0.07, 0.040, "Same axes in both panels: position across, velocity up.  Euler step Δt = %.1f." % M.REV_DT,
      size=13, color=MUTED, style='italic', alpha=a_set)
    T(0.54, 0.040, "Leapfrog's step is symmetric in time, so flipping $v$ undoes it exactly.", size=13, color=SOFT,
      alpha=A(t, q('why', 0.2), 1.0))


def scene7(t):
    q = lambda k, fr=0.0, off=0.0: Q(7, k, fr, off)
    header(7, "Phase-space area")
    L = 2.5
    xs0 = [0.145, 0.355, 0.565, 0.775]; pw = 0.185; ph = 0.30
    rows = [('euler', 'Euler', CORAL, 0.46, 'euler'), ('leapfrog', 'Leapfrog', YELLOW, 0.11, 'lf')]
    ns = M.AREA_STEPS
    a0 = M.shoelace(*M.map_cloud('euler', 0))
    c0 = M.square_boundary()
    a_set = A(t, q('setup', 0.3), 1.0)
    for j, nn in enumerate(ns):
        T(xs0[j] + pw / 2, 0.835, f"n = {nn}" if nn else "start", size=16, color=MUTED, ha='center', alpha=a_set)
    for method, name, col, y0, key in rows:
        T(0.062, y0 + ph / 2, name, size=19, color=col, va='center', alpha=a_set)
        for j, nn in enumerate(ns):
            ax = fig.add_axes([xs0[j], y0, pw, ph])
            ax.set_facecolor('none'); ax.axis('off')
            ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal', adjustable='box')
            ax.axhline(0, color=MUTED, lw=0.6, alpha=0.5 * a_set); ax.axvline(0, color=MUTED, lw=0.6, alpha=0.5 * a_set)
            ax.plot(np.r_[c0[0], c0[0][0]], np.r_[c0[1], c0[1][0]], color=FG, lw=0.9, ls=(0, (3, 2)), alpha=0.55 * a_set)
            tcol = q('setup', 0.7) if nn == 0 else q(key, 0.12 + 0.22 * j)
            a = A(t, tcol, 0.9)
            if a < 0.01: continue
            x, v = M.map_cloud(method, nn)
            ax.fill(x, v, color=col, alpha=0.30 * a, lw=0)
            ax.plot(np.r_[x, x[0]], np.r_[v, v[0]], color=col, lw=1.8, alpha=a)
            area = M.shoelace(x, v) / a0
            TA(ax, 0, -L - 0.05, f"area ×{area:.2f}", size=15, color=col, ha='center', va='top', alpha=a)
    T(0.145, 0.045, f"Each panel: x across, v up, both from −{L} to {L}.  The dashed square is the starting cloud "
                    f"(side {2 * M.AREA_H:g}, centred at (1, 0)).  Step Δt = {M.AREA_DT}.", size=13, color=MUTED,
      style='italic', alpha=a_set)
    ay = A(t, q('lf', 0.85), 1.0)
    T(0.94, 0.915, rf"$\det = {F['det_l']:.0f}$", size=20, color=YELLOW, ha='right', alpha=ay)
    T(0.94, 0.915, rf"$\det = {F['det_e']:.2f}$", size=20, color=CORAL, ha='right',
      alpha=A(t, q('euler', 0.6), 1.0) * (1 - ay))


def scene8(t):
    q = lambda k, fr=0.0, off=0.0: Q(8, k, fr, off)
    header(8, "What leapfrog conserves")
    dt = M.SH_DT; c = 1 - dt ** 2 / 4
    N = 40
    xs, vs = M.ho_kdk(dt, N)
    ax = fig.add_axes([0.05, 0.14, 0.38, 0.70])
    L = 1.35; tk = [-1, 0, 1]
    style_ax(ax, (-L, L), (-L, L), tk, tk, size=12)
    ax.set_aspect('equal', adjustable='box')
    th = np.linspace(0, 2 * np.pi, 300)
    a0 = A(t, q('setup', 0.3), 1.0)
    ax.plot(np.cos(th), np.sin(th), color=FG, lw=1.4, ls=(0, (4, 3)), alpha=0.8 * a0, zorder=2)
    TA(ax, -0.1, 1.08, "true energy: a circle", size=14, color=SOFT, style='italic', ha='right', alpha=a0)
    kk = int(round(lin(t, q('setup', 0.55), q('claim', 0.45)) * N))      # one point per step
    if kk > 0:
        ax.plot(xs[:kk + 1], vs[:kk + 1], 'o', color=BLUE, ms=6, zorder=5)
    ae = A(t, q('claim', 0.40), 1.2)
    ax.plot(np.cos(th), np.sqrt(c) * np.sin(th), color=YELLOW, lw=1.6, alpha=ae, zorder=3)
    TA(ax, 0.1, -1.18, "leapfrog: a nearby ellipse", size=14, color=YELLOW, style='italic', ha='left', alpha=ae)
    Hs = M.shadow(dt, xs, vs); Et = M.ho_energy(xs, vs)
    n = np.arange(N + 1)
    ax1 = fig.add_axes([0.52, 0.575, 0.19, 0.22])
    style_ax(ax1, (0, N), (-2.2e-15, 2.2e-15), [0, 20, 40], [-2e-15, 0, 2e-15], size=11)
    ax1.set_yticklabels([r'$-2\!\times\!10^{-15}$', '0', r'$2\!\times\!10^{-15}$'])
    ax2 = fig.add_axes([0.52, 0.14, 0.19, 0.22])
    style_ax(ax2, (0, N), (-0.095, 0.01), [0, 20, 40], [-0.08, 0], size=11)
    ax2.set_yticklabels([r'$-\frac{\Delta t^2}{8}$', '0'])
    a1 = A(t, q('claim', 0.55), 1.0)
    a2 = A(t, q('bound', 0.1), 1.0)
    for a_, al in ((ax1, a1), (ax2, a2)):
        for s in ('left', 'bottom'): a_.spines[s].set_alpha(al)
        a_.tick_params(labelcolor=to_rgba(MUTED, al), color=to_rgba(MUTED, al))
    ax1.plot(n, Hs - Hs[0], 'o-', color=YELLOW, ms=3.5, lw=1.0, alpha=a1)
    T(0.52, 0.835, r"nearby energy $\tilde H-\tilde H_0$", size=15, color=YELLOW, alpha=a1)
    T(0.52, 0.805, rf"max ${F['sh_drift'] * 1e15:.1f}\times10^{{-15}}$ (rounding)", size=13, color=SOFT, alpha=a1)
    ax2.axhline(-dt ** 2 / 8, color=CORAL, lw=1.0, ls=(0, (3, 2)), alpha=a2)
    ax2.plot(n, Et - Et[0], 'o-', color=BLUE, ms=3.5, lw=1.0, alpha=a2)
    T(0.52, 0.445, r"true energy error $E-E_0$", size=15, color=BLUE, alpha=a2)
    T(0.52, 0.413, f"range {fmt(F['sh_Emin'], 4)} to 0", size=13, color=SOFT, alpha=a2)
    margin(t, [
        dict(t=q('claim', 0.1), kind='eq', size=16, h=0.075, color=YELLOW,
             text=r'$\tilde H=\frac{1}{2}v^2+\frac{1}{2}\left(1-\frac{\Delta t^2}{4}\right)x^2$'),
        dict(t=q('claim', 0.3), kind='aside',
             text=f"Here $\\Delta t={dt}$, so the coefficient is $1-\\Delta t^2/4={c:.2f}$."),
        dict(t=q('bound'), kind='eq', size=16, h=0.065, color=BLUE, text=r'$\left|E-E_0\right|\leq\frac{\Delta t^2}{8}$'),
        dict(t=q('bound', 0.3), kind='aside', text="for this orbit, however long we run."),
        dict(t=q('general'), text="For general forces, $\\tilde H$ is only an approximate series. Backward error "
                                  "analysis says it stays nearly conserved for an extremely long time."),
    ], x=0.745, width=26, top=0.835)


def scene9(t):
    q = lambda k, fr=0.0, off=0.0: Q(9, k, fr, off)
    header(9, "A planet, three methods")
    L = 2.4; tk = [-2, 0, 2]
    a_set = A(t, q('setup', 0.45), 1.0)
    e = M.ECC; b = math.sqrt(1 - e * e)
    th = np.linspace(0, 2 * np.pi, 300)
    ell = (np.cos(th) - e, b * np.sin(th))                               # true ellipse, focus at the origin
    specs = [('orb_euler', 'Euler', CORAL, 'euler', f"h = {M.H_LF}", 'env_euler'),
             ('orb_rk4', 'RK4', GREEN, 'rk4', f"h = {M.H_RK}", 'env_rk4'),
             ('orb_lf', 'Leapfrog', YELLOW, 'lf', f"h = {M.H_LF}", 'env_lf')]
    for j, (dk, name, col, key, hs, ek) in enumerate(specs):
        x0 = 0.05 + 0.315 * j
        ax = phase_ax([x0, 0.20, 0.29, 0.53], L, tk, size=11)
        ax.plot(ell[0], ell[1], color=FG, lw=1.0, ls=(0, (3, 3)), alpha=0.55 * a_set, zorder=2)
        ax.plot([0], [0], 'o', color=FG, ms=8, zorder=6, alpha=a_set)
        rows = KD[dk]
        p = lin(t, q(key, 0.08), q(key, 0.92))
        trail(ax, rows[:, 1], rows[:, 2], p, color=col, lw=1.5, ms=3.5, dots=0)
        T(x0, 0.835, name, size=20, color=col, alpha=a_set)
        T(x0, 0.800, f"{hs}   ·   ≈{F['evals']} force evals per orbit", size=13, color=MUTED, style='italic', alpha=a_set)
        ar = A(t, q(key, 0.9), 0.9)
        apo = float(np.max(np.hypot(rows[:, 1], rows[:, 2])))
        T(x0, 0.115, f"farthest point: {apo:.2f}  (exact 1.50)", size=15, color=SOFT, alpha=ar)
        T(x0, 0.072, f"largest energy error: {sci(float(KD[ek][1]))}", size=15, color=col, alpha=ar)
    T(0.06, 0.020, f"Eccentricity {e}, two orbits, same axes.  Dashed: the true ellipse.  Equal cost: RK4 takes "
                   f"{F['n_rk']} steps per orbit at 4 evaluations each; the others take {F['n_lf']} at 1.",
      size=12.5, color=MUTED, style='italic', alpha=a_set)


def scene10(t):
    q = lambda k, fr=0.0, off=0.0: Q(10, k, fr, off)
    header(10, "A thousand orbits")
    ax = fig.add_axes(AX_MAIN)
    ax.set_xscale('log'); ax.set_yscale('log')
    style_ax(ax, (0.8, 1500), (1e-7, 3), [1, 10, 100, 1000], [1e-6, 1e-4, 1e-2, 1],
             ['1', '10', '100', '1000'], [r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', '1'])
    a0 = A(t, q('setup', 0.3), 1.0)
    ax.set_xlabel('orbits', fontsize=14, color=MUTED, loc='right', labelpad=2, alpha=a0)
    TA(ax, 0.82, 4.5, 'largest energy error so far', size=14, color=MUTED, alpha=a0, va='bottom')
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), alpha=0.7 * a0)
    TA(ax, 1400, 0.62, 'total energy |E| = 0.5', size=13, color=MUTED, ha='right', alpha=a0, va='bottom')

    def curve(data, N, col, cue, span):
        p = lin(t, q(cue, span[0]), q(cue, span[1]))
        if p <= 0: return
        nshow = max(2, min(N, int(round(10 ** (p * math.log10(N))))))
        xs = np.arange(1, nshow + 1)
        ax.plot(xs, data[:nshow], color=col, lw=2.4, zorder=4, solid_capstyle='round')
        ax.plot([xs[-1]], [data[nshow - 1]], 'o', color=col, ms=8, zorder=6)
    curve(KD['env_euler'], 30, CORAL, 'euler', (0.10, 0.90))
    curve(KD['env_rk4'], 1000, GREEN, 'rk4', (0.20, 0.95))
    curve(KD['env_lf'], 1000, YELLOW, 'lf', (0.20, 0.95))
    TA(ax, 36, KD['env_euler'][-1] * 0.8, 'Euler', size=17, color=CORAL, alpha=A(t, q('euler', 0.9), 0.8), va='center')
    TA(ax, 1.15, 3.4e-5, 'RK4', size=17, color=GREEN, alpha=A(t, q('rk4', 0.5), 0.8))
    TA(ax, 1.15, 4.2e-4, 'Leapfrog', size=17, color=YELLOW, alpha=A(t, q('lf', 0.5), 0.8))
    ac = A(t, q('cross', 0.12), 1.0)
    if ac > 0.01:
        xc = F['cross']
        ax.plot([xc], [KD['env_lf'][-1]], 'o', mfc='none', mec=FG, ms=15, mew=1.5, alpha=ac, zorder=8)
        TA(ax, xc * 1.35, KD['env_lf'][-1] * 0.42, f"cross at ≈{xc} orbits", size=15, alpha=ac, va='top')
        a3 = A(t, q('cross', 0.55), 1.0)
        lo, hi = KD['env_lf'][-1], KD['env_rk4'][-1]
        ax.annotate('', xy=(1150, hi), xytext=(1150, lo), arrowprops=dict(arrowstyle='<->', color=FG, lw=1.2, alpha=a3))
        TA(ax, 1000, math.sqrt(lo * hi), f"×{F['rk_over_lf']:.0f}", size=17, ha='right', va='center', alpha=a3)
    margin(t, [
        dict(t=q('setup'), text="Same three runs as before. At each orbit, record the largest energy error seen so far."),
        dict(t=q('euler'), kind='aside', text="Euler: error is 87% of the total energy by orbit 30."),
        dict(t=q('rk4'), text="RK4: error grows in proportion to the number of orbits."),
        dict(t=q('lf'), text="Leapfrog: bounded, at about $2\\times10^{-4}$."),
        dict(t=q('honest'), kind='aside', text="The crossover depends on the setup. The shapes, growth versus a bound, do not."),
    ], x=0.715, width=29)


def scene11(t):
    q = lambda k, fr=0.0, off=0.0: Q(11, k, fr, off)
    header(11, "Where leapfrog falls short")
    a_set = A(t, q('coarse', 0.2), 1.0)
    L = 1.75; tk = [-1, 0, 1]                                              # ---- left: precession at a coarse step
    ax = phase_ax([0.05, 0.25, 0.40, 0.54], L, tk, size=11)
    e = M.ECC; b = math.sqrt(1 - e * e); th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th) - e, b * np.sin(th), color=FG, lw=1.0, ls=(0, (3, 3)), alpha=0.7 * a_set, zorder=2)
    ax.plot([0], [0], 'o', color=YELLOW, ms=8, zorder=6, alpha=a_set)
    rows = KD['coarse']
    p = lin(t, q('coarse', 0.45), q('precess', 0.35))
    trail(ax, rows[:, 1], rows[:, 2], p, color=BLUE, lw=0.9, tip=True, ms=3, alpha=0.9)
    T(0.05, 0.835, "Coarse step  h = 0.1", size=19, alpha=a_set)
    T(0.05, 0.802, "30 orbits.  Dashed: the true ellipse.", size=13, color=MUTED, style='italic', alpha=a_set)
    ac = A(t, q('coarse', 0.85), 0.9)
    T(0.05, 0.155, f"orbit turns {F['prec_01_abs']:.1f}° per orbit", size=17, color=BLUE, alpha=ac)
    T(0.05, 0.115, f"energy error stays below {sci(F['coarse_env_max'])}", size=15, color=SOFT, alpha=ac)
    ap = A(t, q('precess', 0.3), 0.9)
    T(0.05, 0.075, f"at h = 0.05: {F['prec_005_abs']:.2f}° per orbit (÷{F['prec_ratio']:.1f})", size=15, color=SOFT, alpha=ap)
    ax2 = fig.add_axes([0.58, 0.25, 0.36, 0.50])                          # ---- right: variable-step drift
    style_ax(ax2, (0, 1000), (0, 5.0), [0, 250, 500, 750, 1000], [0, 1, 2, 3, 4], size=11)
    av = A(t, q('var', 0.1), 1.0)
    for s in ('left', 'bottom'): ax2.spines[s].set_alpha(av)
    ax2.tick_params(labelcolor=to_rgba(MUTED, av), color=to_rgba(MUTED, av))
    pv = lin(t, q('var', 0.35), q('var', 0.95))
    nk = max(2, int(pv * 1000))
    orb = np.arange(1, 1001)
    if pv > 0:
        ax2.plot(orb[:nk], KD['env_lf'][:nk] * 1e4, color=YELLOW, lw=2.0, alpha=0.95)
        ax2.plot(orb[:nk], KD['var_env'][:nk] * 1e4, color=CORAL, lw=2.4)
        ax2.plot([nk], [KD['var_env'][nk - 1] * 1e4], 'o', color=CORAL, ms=7, zorder=6)
    T(0.58, 0.835, "Variable step", size=19, alpha=av)
    T(0.58, 0.802, r"largest energy error so far ($\times10^{-4}$)", size=13, color=MUTED, style='italic', alpha=av)
    TA(ax2, 20, 4.45, "naive adaptive step: keeps growing", size=14, color=CORAL, alpha=A(t, q('var', 0.8), 0.9))
    TA(ax2, 60, 2.4, "fixed step: bounded", size=14, color=YELLOW, alpha=A(t, q('var', 0.8), 0.9))
    ax2.set_xlabel('orbits', fontsize=13, color=MUTED, loc='right', labelpad=2, alpha=av)
    T(0.58, 0.128, rf"energy at orbit ends drifts ${F['var_e7']:.0f}\times10^{{-7}}$ per orbit", size=16, color=CORAL,
      alpha=A(t, q('var', 0.85), 0.9))
    T(0.58, 0.092, "The two runs use different step sizes: compare the shapes, not the levels.", size=13, color=MUTED,
      style='italic', alpha=A(t, q('var', 0.9), 0.9))
    hline(0.052, 0.05, 0.94, alpha=0.4 * A(t, q('forces', 0.0), 0.8))      # ---- bottom: two more caveats
    T(0.05, 0.020, r"Forces must depend on position only: $a=a(x)$, not on $v$.", size=14, color=SOFT,
      alpha=A(t, q('forces', 0.2), 1.0))
    T(0.50, 0.020, "Bounded energy ≠ correct position: the phase can still slip.", size=14, color=SOFT,
      alpha=A(t, q('phase', 0.2), 1.0))


def scene12(t):
    q = lambda k, fr=0.0, off=0.0: Q(12, k, fr, off)
    header(12, "Where this leaves us")
    cols = [0.06, 0.255, 0.395, 0.525, 0.66, 0.795]
    heads = ["method", "order", "force evals\nper step", "reversible", "symplectic", "energy error,\nlong runs"]
    hy = 0.765
    th = q('row1', 0, -0.6)
    for x, s in zip(cols, heads):
        T(x, hy, s, size=14, color=MUTED, style='italic', alpha=A(t, th + 0.2, 0.8), va='bottom', linespacing=1.15)
    hline(0.84, alpha=0.7 * A(t, th, 0.8)); hline(hy - 0.012, alpha=0.5 * A(t, th + 0.2, 0.8))
    rows = [
        ("Euler", "1", "1", "no", "no", "grows fast", CORAL),
        ("Leapfrog", "2", "1", "yes", "yes", "bounded", YELLOW),
        ("RK4", "4", "4", "no", "no", "grows slowly", GREEN),
        ("Yoshida", "4", "3", "yes", "yes", "bounded", FG),
    ]
    for i, r in enumerate(rows):
        yy = hy - 0.095 - 0.098 * i
        al = A(t, q(f'row{i + 1}', 0.05), 0.9)
        T(cols[0], yy, r[0], size=21, color=r[6], alpha=al)
        for j in range(1, 6):
            col = SOFT
            if r[j] == 'yes': col = GREEN
            if r[j] == 'no': col = CORAL
            if j == 5: col = YELLOW if r[j] == 'bounded' else (CORAL if 'fast' in r[j] else SOFT)
            T(cols[j], yy, r[j], size=19, color=col, alpha=al)
    hline(hy - 0.095 - 0.098 * 3 - 0.045, alpha=0.7 * A(t, q('row4', 0.9), 0.8))
    T(0.06, 0.27, "Yoshida: three leapfrog steps in a row, with weights", size=15, color=SOFT,
      alpha=A(t, q('row4', 0.35), 1.0))
    T(0.06, 0.215, rf"$w_1={F['yosh_w1']:.4f},\quad w_0={fmt(F['yosh_w0'])},\quad w_1={F['yosh_w1']:.4f}$"
                   rf"$\qquad$(measured order {F['yosh_slope']:.2f})", size=18, color=FG,
      alpha=A(t, q('row4', 0.5), 1.0))
    T(0.06, 0.105, "Order tells you how small the error is.  Structure tells you how it behaves.", size=22,
      color=YELLOW, alpha=A(t, q('close', 0.55), 1.4))
    T(0.06, 0.055, "Energy-error entries are for the planet test (e = 0.5); Yoshida's was checked in tests/.", size=13,
      color=MUTED, style='italic', alpha=A(t, q('close', 0.7), 1.0))


SCENES = [scene0, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10, scene11, scene12]
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
