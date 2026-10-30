# Euler's Method: a 3Blue1Brown × Tufte explainer video

A fully scripted, regenerable video on Euler's method for ODEs: following the slope, its first-order
error, its stability limit, backward Euler, the spring that gains energy, and what the midpoint method
and RK4 buy for their extra slope evaluations. Animation, on-screen numbers, narration text, voice and
timing are all generated from source. This is part 3, after the midpoint-rule and leapfrog videos;
it uses the same pipeline.

* **Visuals:** matplotlib, frame by frame. Dark canvas and accent colours (3Blue1Brown); serif type,
  range-frame axes, direct labels, margin sidenotes, small multiples, hairline tables (Tufte).
* **Voice:** [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural TTS, local, CPU.
  "Euler" is written "Oiler" in `narration.json` so the voice says it correctly.
* **Sync:** narration drives timing. Each spoken line has a key; the build measures the clip and
  schedules on-screen events from it (`Q(scene, key, frac, off)` in `scenes.py`).
* **Truthfulness:** every number on screen and in the voiceover comes from `mathcore.py` (narration uses
  `{placeholders}`; decimals are spelled out in words). `tests/` checks each mathematical claim
  numerically (22 tests), including the closed forms used for the cost plot against real loops.

## Quick start
```bash
brew install ffmpeg              # Debian/Ubuntu: sudo apt install ffmpeg
uv sync                          # once, at the repo root or here (one shared environment)
uv run build.py setup            # download Kokoro model files (~350 MB) into ../models
uv run build.py test             # verify the maths
uv run build.py all --res 4k     # tts -> timeline -> audio -> render -> build/euler_4k.mp4
```
`--res` accepts `720p`, `1080p`, `1440p`, `4k`; use `720p` while iterating.
Stages: `setup tts timeline audio render mux all still test`. `still N SECONDS --res 720p --out f.png`
previews a frame without audio. Flags: `--scenes 4,5`, `--procs N`, `--crf`, `--preset`, `--voice`,
`--speed`, `--silent`, `--force-tts`.

## What is in the video (11 scenes, about 7.5 minutes)
Title · the slope field of y' = y · one step of size 1 (misses e by 0.718) · four steps of ½ ·
shrinking the step, (1 + 1/n)^n → e · order of accuracy (slope 1 on y' = y and on a logistic equation;
local error ½y''h²) · stability on y' = −5y (limit h < 2/λ = 0.4) · amplification factors, backward
Euler · Euler on a spring (energy × (1 + h²) per step) · error against slope evaluations for Euler,
midpoint and RK4 · summary table.

## Editing
* **Words:** `narration.json`. Spell maths for the voice; use `{w_...}` placeholders (see
  `spoken_facts` in `mathcore.py`). Only changed lines are re-synthesised.
* **Visuals:** `euler_video/scenes.py`; each scene is a pure function of local time.
* **Maths / parameters:** `euler_video/mathcore.py` (step sizes, λ, tolerance).
* **Tools:** `uv run --extra asr tools_asr_check.py` transcribes every clip with Whisper and diffs it against the script.
  Expect ~3% mismatch from number formatting ("two point five" vs "2.5"); look for real misreadings.

## Layout
```
build.py  narration.json  tools_asr_check.py
euler_video/{mathcore,scenes,narration,render}.py
tests/test_math.py
build/                generated (git-ignored)
../fonts/ ../models/  shared with the other videos (TeX Gyre Pagella is bundled; models are downloaded)
```

## Caveats
* The tests are single problems (y' = y, y' = −5y, a logistic equation, a spring). Orders and stability
  limits are shown for those; the video says "this problem", not "all problems".
* Cost comparison (Euler 1.4 million evaluations vs midpoint 1,346 vs RK4 52 for error < 10⁻⁶) is for a
  smooth problem at one end time, the best case for high-order methods, and counts slope evaluations
  only. The plot uses closed forms; tests confirm the minimum step counts with the real integrators.
* The stability column in the summary table is for the decay test y' = −λy. Leapfrog is built for
  oscillators, so it is shown as "—".
* TTS is intelligible (checked with Whisper) but not a human voice.
* Licences: Kokoro-82M Apache-2.0, `kokoro-onnx` MIT, TeX Gyre GUST Font License. Verify before publishing.
