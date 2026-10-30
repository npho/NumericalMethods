# Runge–Kutta: a 3Blue1Brown × Tufte explainer video

A fully scripted, regenerable video on the classic fourth-order Runge–Kutta method (RK4): why one slope
is not enough, the midpoint method, the four slopes and their weights, the link to Simpson's rule, order
of accuracy (and where it fails), cost, stability, and a long run on the spring. Animation, on-screen
numbers, narration text, voice and timing are all generated from source. This is part 4, after the
midpoint-rule, leapfrog and Euler videos; it uses the same pipeline.

* **Visuals:** matplotlib, frame by frame. Dark canvas and accent colours (3Blue1Brown); serif type,
  range-frame axes, direct labels, margin sidenotes, small multiples, hairline tables (Tufte).
* **Voice:** [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural TTS, local, CPU.
  "Euler" is written "Oiler" and "Runge" "Runga" in `narration.json` so the voice says them correctly.
* **Sync:** narration drives timing. Each spoken line has a key; the build measures the clip and
  schedules on-screen events from it (`Q(scene, key, frac, off)` in `scenes.py`).
* **Truthfulness:** every number on screen and in the voiceover comes from `mathcore.py` (narration uses
  `{placeholders}`; decimals are spelled out in words). `tests/` checks each mathematical claim
  numerically (23 tests), including the closed forms used in the cost plot against real loops.

## Quick start
```bash
brew install ffmpeg              # Debian/Ubuntu: sudo apt install ffmpeg
uv sync
uv run build.py setup            # download Kokoro model files (~350 MB) into ./models
uv run build.py test             # verify the maths
uv run build.py all --res 4k     # tts -> timeline -> audio -> render -> build/rk4_4k.mp4
```
`--res` accepts `720p`, `1080p`, `1440p`, `4k`; use `720p` while iterating.
Stages: `setup tts timeline audio render mux all still test`. `still N SECONDS --res 720p --out f.png`
previews a frame without audio. Flags: `--scenes 4,5`, `--procs N`, `--crf`, `--preset`, `--voice`,
`--speed`, `--silent`, `--force-tts`.

## What is in the video (12 scenes)
Title · what Euler's step misses · the midpoint method · the four slopes k1..k4 and weights 1,2,2,1 over 6 ·
a worked step on y' = y (65/24 against e) · Simpson's rule as the special case · four steps of ½ ·
order of accuracy (4 on smooth problems, 1.5 on y' = √t) · error versus cost · stability limits
(2.79 for RK4, but 0.70 per evaluation) · 200 periods on the spring against leapfrog · summary table.

## Editing
* **Words:** `narration.json`. Spell maths for the voice; use `{w_...}` placeholders (see
  `spoken_facts` in `mathcore.py`). Only changed lines are re-synthesised.
* **Visuals:** `rk4_video/scenes.py`; each scene is a pure function of local time.
* **Maths / parameters:** `rk4_video/mathcore.py` (step sizes, budget, tolerance, spring run).
* **Tools:** `uv run --extra asr tools_asr_check.py` transcribes every clip with Whisper and diffs it against the script.
  Expect ~3% mismatch from number formatting; look for real misreadings.

## Layout
```
build.py  narration.json  pyproject.toml  uv.lock  tools_asr_check.py
rk4_video/{mathcore,scenes,narration,render}.py
tests/test_math.py
fonts/                TeX Gyre Pagella (bundled, GUST Font License)
models/ build/        generated (git-ignored; models/ is a symlink or a download)
```

## Caveats
* Test problems are y' = y, a logistic equation, y' = √t and the spring. Orders and costs are shown for
  those, and the video says "this problem".
* The cost comparison counts slope evaluations on a smooth problem at one end time, the best case for
  high order. Euler's 13.6 billion evaluations come from a closed form (expm1/log1p, free of round-off
  cancellation); the tests check it against the first-order law, and RK4's 492 and the midpoint method's
  134,618 against the real integrators.
* The stability numbers are for the decay test y' = −λy. "Per evaluation" divides by slopes per step.
* The spring energy figures use h = 0.5 over 200 periods. RK4's loss shrinks quickly with h (h⁶).
* The statement that RK45 is scipy's default `solve_ivp` method was checked against the installed scipy.
* TTS is intelligible (checked with Whisper) but not a human voice.
* Licences: Kokoro-82M Apache-2.0, `kokoro-onnx` MIT, TeX Gyre GUST Font License. Verify before publishing.
