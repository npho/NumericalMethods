#!/usr/bin/env python3
"""IEEE 754 in Motion: narrated explainer video, generated from code.

    python build.py                 # every step, 1080p
    python build.py --scale 2       # every step, 4K (3840x2160)
    python build.py video           # one step: models | narration | render | video | verify

Pipeline
  models     download the Kokoro text-to-speech model once (about 350 MB) into ./models
  narration  speak each line of narration.json, trim, limit and match loudness -> build/narration/cNN.wav
  render     screenshot every slide and the title-intro frames from src/scenes.html with headless Chromium
  video      lay everything on a timeline, crossfade slides, mix audio, encode H.264 + AAC -> output/*.mp4
  verify     decode the finished file and check sync, loudness, holds, resolution

Timeline (seconds)
  title centered and still (intro-hold) -> glides up to its place (intro-move) -> slide 1 ... slide 25.
  Each slide is fully visible for  max(min-hold, lead + narration + tail)  and slides are separated
  by crossfades of `fade` seconds. Narration starts `lead` seconds after a slide is fully visible,
  so speech never overlaps a crossfade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC, BUILD, MODELS, OUTPUT = ROOT / 'src', ROOT / 'build', ROOT / 'models', ROOT / 'output'
NARR = BUILD / 'narration'
MODEL_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'
MODEL_FILES = {'kokoro-v1.0.onnx': 300_000_000, 'voices-v1.0.bin': 20_000_000}  # name -> minimum plausible size (bytes)
VIEW_W, VIEW_H = 1920, 1080  # CSS pixels; --scale multiplies the real pixel count


# --------------------------------------------------------------------------- helpers
def load_cfg() -> dict:
    cfg = json.loads((ROOT / 'narration.json').read_text(encoding='utf-8'))
    assert cfg['lines'], 'narration.json has no lines'
    return cfg


def need(*tools: str) -> None:
    for t in tools:
        if not shutil.which(t):
            sys.exit(f'`{t}` was not found on PATH. Install ffmpeg (with libx264) and try again.')


def frames_dir(scale: int) -> Path:
    return BUILD / f'frames_{VIEW_W * scale}x{VIEW_H * scale}'


def run_ffmpeg(cmd: list[str], total: float | None = None, label: str = 'encoding') -> None:
    """Run ffmpeg, printing a percentage from its -progress stream; fail loudly with its error text."""
    full = [cmd[0], '-hide_banner', '-loglevel', 'error', '-nostats', '-progress', 'pipe:1'] + cmd[1:]
    p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last, errs = 0.0, []
    for line in p.stdout:
        line = line.strip()
        if line.startswith('out_time_us=') and total:
            try:
                t = int(line.split('=', 1)[1]) / 1e6
            except ValueError:
                continue
            if t - last >= 10:
                print(f'  {label}: {min(100.0, 100 * t / total):5.1f}%', flush=True)
                last = t
        elif not re.match(r'^[a-z_0-9]+=', line):
            errs.append(line)
    if p.wait():
        sys.exit('ffmpeg failed:\n' + '\n'.join(errs[-25:]))


def ease(p: float) -> float:  # easeInOutCubic
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def smooth(x: float) -> float:  # smoothstep on [0, 1]
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x * x * (3 - 2 * x)


# --------------------------------------------------------------------------- page
def build_page(title: str) -> Path:
    """build/video.html = the scenes page's CSS + JS inside the fixed 1920x1080 video layout."""
    html = (SRC / 'scenes.html').read_text(encoding='utf-8')
    css = re.search(r'<style>(.*?)</style>', html, re.S).group(1)
    js = re.search(r'<script>(.*)</script>', html, re.S).group(1)
    layout = (SRC / 'video_layout.css').read_text(encoding='utf-8')
    title = escape(title)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}
{layout}</style></head><body>
<h1 class="vtitle">{title}</h1>
<div class="vstage"><svg id="stage" viewBox="0 0 960 520"></svg></div>
<p id="cap" class="vcap"></p>
<div class="hidden"><button id="back"></button><button id="play"></button><button id="next"></button><div id="dots"></div><nav id="tabs"></nav></div>
<script>window.__VIDEO__ = true;</script>
<script>{js}</script></body></html>"""
    BUILD.mkdir(exist_ok=True)
    out = BUILD / 'video.html'
    out.write_text(page, encoding='utf-8')
    return out


def open_page(pw, title: str, scale: int):
    page_path = build_page(title)
    browser = pw.chromium.launch(args=['--allow-file-access-from-files'])
    ctx = browser.new_context(viewport={'width': VIEW_W, 'height': VIEW_H}, device_scale_factor=scale)
    page = ctx.new_page()
    page.goto(page_path.as_uri())
    ok = page.evaluate("""async () => {
        await document.fonts.load('600 68px "STIX Two Text"'); await document.fonts.load('400 40px "STIX Two Text"');
        await document.fonts.load('500 20px "JetBrains Mono"'); await document.fonts.ready;
        return document.fonts.check('600 68px "STIX Two Text"') && document.fonts.check('500 20px "JetBrains Mono"'); }""")
    if not ok:
        sys.exit('The bundled fonts in ./fonts failed to load.')
    return browser, page


def page_captions(page) -> list[str]:
    counts = page.evaluate('window.__scenes')
    shown = []
    for si, c in enumerate(counts):
        for st in range(c):
            page.evaluate(f'__show({si},{st})')
            shown.append(page.evaluate("document.getElementById('cap').textContent"))
    return shown


# --------------------------------------------------------------------------- step: models
def step_models(a) -> None:
    MODELS.mkdir(exist_ok=True)
    for name, min_size in MODEL_FILES.items():
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size >= min_size:
            print(f'  have {name}')
            continue
        print(f'  downloading {name} ...', flush=True)
        tmp = dest.with_name(dest.name + '.part')
        with urllib.request.urlopen(MODEL_URL + name) as r, open(tmp, 'wb') as f:
            total, done, mark = int(r.headers.get('Content-Length') or 0), 0, 0.25
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= mark:
                    print(f'    {100 * done / total:.0f}%', flush=True)
                    mark += 0.25
        if tmp.stat().st_size < min_size:
            sys.exit(f'{name} looks truncated ({tmp.stat().st_size} bytes). Delete {tmp} and retry.')
        tmp.replace(dest)


# --------------------------------------------------------------------------- step: narration
def _limit(y, sr, ceil):
    """Lookahead peak limiter (ffmpeg alimiter, auto-level off)."""
    import numpy as np
    import soundfile as sf
    tin, tout = NARR / '_in.wav', NARR / '_out.wav'
    sf.write(tin, y.astype(np.float32), sr, subtype='FLOAT')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tin), '-af',
                    f'alimiter=limit={ceil}:attack=5:release=60:level=0', '-c:a', 'pcm_f32le', str(tout)], check=True)
    out, _ = sf.read(tout)
    return out


def _speak(k, text, voice, speed, target, ceil):
    """Synthesize one line -> trimmed, faded, limited clip at the target integrated loudness."""
    import numpy as np
    import pyloudnorm as pyln
    y, sr = k.create(text, voice=voice, speed=speed, lang='en-us')
    y = y.astype(np.float64)
    idx = np.where(np.abs(y) > 10 ** (-45 / 20))[0]                    # trim silence below -45 dB, keep a little air
    y = y[max(0, idx[0] - int(0.06 * sr)): min(len(y), idx[-1] + int(0.12 * sr))]
    n = int(0.008 * sr)
    ramp = np.linspace(0, 1, n)
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    meter = pyln.Meter(sr)
    g = target - meter.integrated_loudness(y)
    for _ in range(6):                                                  # gain -> limit -> re-measure -> correct
        out = _limit(y * 10 ** (g / 20), sr, ceil)
        m = meter.integrated_loudness(out)
        if abs(m - target) < 0.15:
            break
        g += target - m
    return out, sr, m


def step_narration(a) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro
    from playwright.sync_api import sync_playwright

    need('ffmpeg')
    cfg = load_cfg()
    lines = cfg['lines']
    for name in MODEL_FILES:
        if not (MODELS / name).exists():
            sys.exit('Model files are missing. Run: python build.py models')

    # The spoken text must belong to the captions that are actually on the slides.
    with sync_playwright() as pw:
        browser, page = open_page(pw, cfg['title'], 1)
        shown = page_captions(page)
        browser.close()
    expected = [l['caption'] for l in lines]
    if shown != expected:
        for i, (s, e) in enumerate(zip(shown, expected)):
            if s != e:
                sys.exit(f'Caption {i + 1} in narration.json differs from the slide:\n  slide: {s!r}\n  json : {e!r}')
        sys.exit(f'The page has {len(shown)} captions but narration.json has {len(expected)}.')
    print(f'  all {len(shown)} captions match the slides')

    NARR.mkdir(parents=True, exist_ok=True)
    mf = NARR / 'manifest.json'
    old = json.loads(mf.read_text()) if mf.exists() else {}
    k = Kokoro(str(MODELS / 'kokoro-v1.0.onnx'), str(MODELS / 'voices-v1.0.bin'))
    if a.voice not in k.get_voices():
        sys.exit(f'Unknown voice {a.voice!r}. Available: {", ".join(sorted(k.get_voices()))}')
    new, done = {}, 0
    for i, line in enumerate(lines):
        key = hashlib.sha1(f'{a.voice}|{a.speed}|{a.lufs}|{a.peak}|{line["spoken"]}'.encode()).hexdigest()
        wav = NARR / f'c{i:02d}.wav'
        new[f'{i:02d}'] = key
        if old.get(f'{i:02d}') == key and wav.exists():
            continue                                                    # unchanged line: reuse the cached clip
        y, sr, m = _speak(k, line['spoken'], a.voice, a.speed, a.lufs, a.peak)
        sf.write(wav, y.astype(np.float32), sr, subtype='PCM_16')
        done += 1
        print(f'  line {i + 1:2d}: {len(y) / sr:5.2f}s  {m:6.2f} LUFS', flush=True)
    for tmp in ('_in.wav', '_out.wav'):
        (NARR / tmp).unlink(missing_ok=True)
    durs = [sf.info(NARR / f'c{i:02d}.wav').duration for i in range(len(lines))]
    sr = sf.info(NARR / 'c00.wav').samplerate
    mf.write_text(json.dumps(new))
    (NARR / 'durations.json').write_text(json.dumps({'sr': sr, 'durations': durs, 'voice': a.voice}))
    print(f'  {done} clip(s) synthesized, {len(lines) - done} reused; total narration {sum(durs):.1f}s')


# --------------------------------------------------------------------------- step: render
def step_render(a) -> None:
    from PIL import Image
    from playwright.sync_api import sync_playwright
    import numpy as np

    cfg = load_cfg()
    out = frames_dir(a.scale)
    (out / 'slides').mkdir(parents=True, exist_ok=True)
    (out / 'intro').mkdir(parents=True, exist_ok=True)
    for f in (out / 'intro').iterdir():
        f.unlink()
    n_move = round(a.intro_move * a.fps)
    t0 = time.time()
    with sync_playwright() as pw:
        browser, page = open_page(pw, cfg['title'], a.scale)
        counts = page.evaluate('window.__scenes')
        n = 0
        for si, c in enumerate(counts):
            for st in range(c):
                page.evaluate(f'__show({si},{st})')
                page.evaluate('document.fonts.ready')
                page.wait_for_timeout(150)
                page.screenshot(path=str(out / 'slides' / f's{n:02d}.png'))
                n += 1
        if n != len(cfg['lines']):
            sys.exit(f'{n} slides on the page but {len(cfg["lines"])} lines in narration.json')

        # Title intro: the layout's own <h1> starts large and centered, then eases up into place.
        page.evaluate('__show(0,0)')
        base_px = page.evaluate("parseFloat(getComputedStyle(document.querySelector('.vtitle')).fontSize)")
        cy = page.evaluate("(()=>{const r=document.querySelector('.vtitle').getBoundingClientRect();return r.top+r.height/2})()")

        def shoot(p: float, path: Path) -> None:
            e = ease(p)
            dy = (VIEW_H / 2 - cy) * (1 - e)
            s = 1 + (a.intro_title_px / base_px - 1) * (1 - e)
            op = smooth((p - 0.66) / 0.34)                               # panel + caption wait until the title has cleared them
            tf = '' if p >= 1 else f'translateY({dy:.3f}px) scale({s:.5f})'
            ops = '' if p >= 1 else f'{op:.4f}'
            page.evaluate("""([tf,op])=>{document.querySelector('.vtitle').style.transform=tf;
                document.querySelector('.vstage').style.opacity=op;document.getElementById('cap').style.opacity=op;}""", [tf, ops])
            page.wait_for_timeout(40)
            page.screenshot(path=str(path))

        shoot(0.0, out / 'intro' / 'hold.png')
        for i in range(1, n_move + 1):
            shoot(i / n_move, out / 'intro' / f'm{i:03d}.png')
        browser.close()
    last = np.asarray(Image.open(out / 'intro' / f'm{n_move:03d}.png').convert('RGB'), dtype=np.int16)
    first = np.asarray(Image.open(out / 'slides' / 's00.png').convert('RGB'), dtype=np.int16)
    size = Image.open(out / 'slides' / 's00.png').size
    print(f'  {n} slides + {n_move + 1} intro frames at {size[0]}x{size[1]} in {time.time() - t0:.0f}s; '
          f'intro end vs slide 1 max pixel diff = {int(np.abs(last - first).max())}')


# --------------------------------------------------------------------------- step: video
def make_timeline(durs: list[float], a) -> dict:
    n = len(durs)
    hold = [max(a.min_hold, (a.lead_first if k == 0 else a.lead) + durs[k] + a.tail) for k in range(n)]
    S = [0.0]
    for k in range(1, n):
        S.append(S[-1] + hold[k - 1] + a.fade)
    t0 = a.intro_hold + a.intro_move
    return {'n': n, 'T0': t0, 'S': S, 'hold': hold, 'dur': durs, 'fade': a.fade, 'fps': a.fps, 'scale': a.scale,
            'intro_hold': a.intro_hold, 'intro_move': a.intro_move, 'lufs': a.lufs,
            'speech_start': [t0 + S[k] + (a.lead_first if k == 0 else a.lead) for k in range(n)],
            'total': t0 + S[-1] + hold[-1]}


def step_video(a) -> None:
    import numpy as np
    import soundfile as sf

    need('ffmpeg')
    nar_file = NARR / 'durations.json'
    if not nar_file.exists():
        sys.exit('No narration yet. Run: python build.py narration')
    nar = json.loads(nar_file.read_text())
    fdir = frames_dir(a.scale)
    if not (fdir / 'intro' / 'hold.png').exists():
        sys.exit(f'No frames for scale {a.scale}. Run: python build.py render --scale {a.scale}')
    T = make_timeline(nar['durations'], a)
    n, fps, fade = T['n'], a.fps, a.fade
    W, H = VIEW_W * a.scale, VIEW_H * a.scale
    out = Path(a.out) if a.out else OUTPUT / f'ieee754_in_motion_{W}x{H}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    T['output'] = str(out)
    (BUILD / 'timeline.json').write_text(json.dumps(T, indent=1))
    print(f'  {n} slides, total {T["total"]:.1f}s ({sum(h > a.min_hold + 1e-6 for h in T["hold"])} held longer than {a.min_hold}s)')

    # audio: one mono track the length of the video, each clip dropped at its planned start
    sr = nar['sr']
    mix = np.zeros(int(np.ceil((T['total'] + 0.5) * sr)), dtype=np.float32)
    for k in range(n):
        y, s = sf.read(NARR / f'c{k:02d}.wav', dtype='float32')
        assert s == sr
        i0 = int(round(T['speech_start'][k] * sr))
        mix[i0:i0 + len(y)] += y
    assert np.abs(mix).max() <= 1.0, 'narration clips overlap'
    sf.write(NARR / 'mix.wav', mix, sr, subtype='FLOAT')

    # video: still intro, eased-title intro, then the slides with crossfades; one RGB->BT.709 conversion at the end
    cmd = ['ffmpeg', '-y']
    cmd += ['-loop', '1', '-framerate', str(fps), '-t', f'{a.intro_hold:.3f}', '-i', str(fdir / 'intro' / 'hold.png')]
    cmd += ['-framerate', str(fps), '-i', str(fdir / 'intro' / 'm%03d.png')]
    for k in range(n):
        length = (fade if k > 0 else 0) + T['hold'][k] + (fade if k < n - 1 else 0)
        cmd += ['-loop', '1', '-framerate', str(fps), '-t', f'{length:.3f}', '-i', str(fdir / 'slides' / f's{k:02d}.png')]
    norm = f'fps={fps},settb=1/{fps},setsar=1,format=gbrp'
    parts = [f'[0:v]{norm}[h]', f'[1:v]{norm}[m]'] + [f'[{k + 2}:v]{norm}[s{k}]' for k in range(n)]
    prev = 's0'
    for k in range(1, n):
        off = T['S'][k - 1] + T['hold'][k - 1]
        parts.append(f'[{prev}][s{k}]xfade=transition=fade:duration={fade}:offset={off:.3f}[x{k}]')
        prev = f'x{k}'
    parts.append(f'[h][m][{prev}]concat=n=3:v=1:a=0[cat]')
    parts.append('[cat]scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p[vout]')
    preset = a.preset or ('medium' if a.scale == 1 else 'fast')
    crf = a.crf if a.crf is not None else (16 if a.scale == 1 else 18)
    silent = BUILD / f'_silent_{W}.mp4'
    cmd += ['-filter_complex', ';'.join(parts), '-map', '[vout]', '-r', str(fps), '-c:v', 'libx264', '-preset', preset,
            '-crf', str(crf), '-g', str(fps * 3), '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'bt709', '-color_range', 'tv', str(silent)]
    t0 = time.time()
    print(f'  encoding {W}x{H} (x264 {preset}, crf {crf}) ...', flush=True)
    run_ffmpeg(cmd, total=T['total'])
    print(f'  video encoded in {time.time() - t0:.0f}s')
    run_ffmpeg(['ffmpeg', '-y', '-i', str(silent), '-i', str(NARR / 'mix.wav'), '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                '-af', 'aresample=48000', '-c:a', 'aac', '-b:a', '160k', '-t', f'{T["total"]:.3f}', '-movflags', '+faststart',
                str(out)], label='muxing')
    silent.unlink(missing_ok=True)
    print(f'  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)')


# --------------------------------------------------------------------------- step: verify
def step_verify(a) -> int:
    import numpy as np
    from PIL import Image

    need('ffmpeg', 'ffprobe')
    tf = BUILD / 'timeline.json'
    if not tf.exists():
        sys.exit('No timeline. Run: python build.py video')
    T = json.loads(tf.read_text())
    mp4, n, D, T0 = T['output'], T['n'], T['fade'], T['T0']
    fdir = frames_dir(T['scale'])
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')

    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height:format=duration',
                                      '-of', 'json', mp4], capture_output=True, text=True, check=True).stdout)
    v = next(s for s in info['streams'] if s['codec_type'] == 'video')
    dur = float(info['format']['duration'])
    check('resolution', (v['width'], v['height']) == (VIEW_W * T['scale'], VIEW_H * T['scale']), f'{v["width"]}x{v["height"]}')
    check('duration', abs(dur - T['total']) < 0.15, f'{dur:.2f}s (planned {T["total"]:.2f}s)')

    # audio: silent through intro and crossfades, speech inside each slide's window, consistent loudness
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', mp4, '-vn', '-f', 'f32le', '-ac', '1', '-ar', '48000', 'pipe:1'],
                         capture_output=True, check=True).stdout
    au, sr = np.frombuffer(raw, dtype=np.float32).astype(np.float64), 48000
    h = sr // 100
    env = 20 * np.log10(np.sqrt(np.convolve(au ** 2, np.ones(h) / h, 'valid')[::h]) + 1e-9)
    win = lambda t0, t1: env[int(max(t0, 0) * 100): int(t1 * 100)]
    fi = max(win(T0 + T['S'][k] - D, T0 + T['S'][k]).max() for k in range(1, n))
    fo = max(win(T0 + T['S'][k] + T['hold'][k], T0 + T['S'][k] + T['hold'][k] + D).max() for k in range(n - 1))
    check('quiet during intro', win(0, T['speech_start'][0] - 0.05).max() < -50, f'{win(0, T["speech_start"][0] - 0.05).max():.0f} dBFS')
    check('quiet during crossfades', max(fi, fo) < -50, f'loudest {max(fi, fo):.0f} dBFS')
    onsets, margins = [], []
    for k in range(n):
        s0 = T['speech_start'][k]
        s1 = s0 + T['dur'][k]
        w = win(s0 - 0.1, s1 + 0.3)
        idx = np.where(w > -40)[0]
        onsets.append(idx[0] / 100 - 0.1)
        margins.append(T0 + T['S'][k] + T['hold'][k] - (s0 + idx[-1] / 100 - 0.1))
    check('speech starts on cue', -0.05 <= min(onsets) and max(onsets) <= 0.25, f'{min(onsets):.2f}..{max(onsets):.2f}s after planned start')
    check('speech ends before each fade', min(margins) >= 0.3, f'at least {min(margins):.2f}s before the next crossfade')
    txt = subprocess.run(['ffmpeg', '-nostats', '-i', mp4, '-vn', '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    summ = txt[txt.rfind('Summary:'):]
    lufs = float(re.search(r'I:\s+(-?[\d.]+) LUFS', summ).group(1))
    tp = float(re.search(r'Peak:\s+(-?[\d.]+) dBFS', summ).group(1))
    check('loudness', abs(lufs - T['lufs']) < 1.0, f'{lufs:.1f} LUFS (target {T["lufs"]})')
    check('true peak', tp < -0.3, f'{tp:.1f} dBFS')

    # video: every slide is still and matches its source frame at the start, middle and end of its hold
    def grab(t: float):
        p = BUILD / '_f.png'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t:.3f}', '-i', mp4, '-frames:v', '1', str(p)], check=True)
        return np.asarray(Image.open(p).convert('RGB'), dtype=np.int16)

    worst, still = 0.0, 0.0
    for k in range(n):
        v0 = T0 + T['S'][k]
        v1 = v0 + T['hold'][k]
        src = np.asarray(Image.open(fdir / 'slides' / f's{k:02d}.png').convert('RGB'), dtype=np.int16)
        fr = [grab(v0 + 0.05), grab((v0 + v1) / 2), grab(v1 - 0.05)]
        worst = max(worst, max(np.abs(f - src).mean() for f in fr))
        still = max(still, np.abs(fr[0] - fr[2]).mean())
    (BUILD / '_f.png').unlink(missing_ok=True)
    check('slides match their source frames', worst < 3.0, f'worst mean difference {worst:.2f} of 255 (BT.709 conversion accounts for about 1.7)')
    check('slides hold still', still < 0.5, f'worst start-to-end difference {still:.3f}')
    check('minimum hold', min(T['hold']) >= 3.0 - 1e-9, f'shortest hold {min(T["hold"]):.2f}s')
    bad = [r for r in results if not r[1]]
    print('\nALL CHECKS PASSED' if not bad else f'\n{len(bad)} CHECK(S) FAILED')
    return 1 if bad else 0


# --------------------------------------------------------------------------- cli
def main() -> None:
    steps = ['models', 'narration', 'render', 'video', 'verify']
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', nargs='?', default='all', choices=['all'] + steps, help='which step to run (default: all)')
    ap.add_argument('--scale', type=int, choices=[1, 2], default=1, help='1 = 1920x1080, 2 = 3840x2160 (default 1)')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--out', help='output .mp4 path (default output/ieee754_in_motion_<W>x<H>.mp4)')
    g = ap.add_argument_group('narration')
    g.add_argument('--voice', default='af_heart', help='Kokoro voice id (default af_heart)')
    g.add_argument('--speed', type=float, default=1.0, help='speaking speed multiplier (default 1.0)')
    g.add_argument('--lufs', type=float, default=-17.0, help='integrated loudness per clip (default -17)')
    g.add_argument('--peak', type=float, default=0.84, help='peak ceiling, linear (default 0.84 = about -1.5 dBFS)')
    g = ap.add_argument_group('timing (seconds)')
    g.add_argument('--min-hold', type=float, default=3.0, help='shortest time a slide stays fully visible')
    g.add_argument('--fade', type=float, default=0.6, help='crossfade length between slides')
    g.add_argument('--lead', type=float, default=0.4, help='silence between a slide appearing and its narration')
    g.add_argument('--lead-first', type=float, default=0.6, help='same, for the first slide (after the intro)')
    g.add_argument('--tail', type=float, default=0.6, help='silence between narration ending and the next crossfade')
    g.add_argument('--intro-hold', type=float, default=2.5, help='how long the big centered title stays still')
    g.add_argument('--intro-move', type=float, default=1.4, help='how long the title takes to glide into place')
    g.add_argument('--intro-title-px', type=float, default=156.0, help='font size of the big intro title (CSS px)')
    g = ap.add_argument_group('encoder')
    g.add_argument('--preset', help='x264 preset (default: medium at 1080p, fast at 4K)')
    g.add_argument('--crf', type=int, help='x264 quality, lower is better (default: 16 at 1080p, 18 at 4K)')
    a = ap.parse_args()

    todo = steps if a.step == 'all' else [a.step]
    fns = {'models': step_models, 'narration': step_narration, 'render': step_render, 'video': step_video, 'verify': step_verify}
    code = 0
    for s in todo:
        print(f'== {s}', flush=True)
        code = fns[s](a) or 0
    sys.exit(code)


if __name__ == '__main__':
    main()
