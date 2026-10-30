# Leapfrog Integration: a 3Blue1Brown × Tufte explainer video

A fully scripted, regenerable video on the leapfrog (kick-drift) method for integrating ODEs such as
orbits. Animation, on-screen numbers, narration text, voice and timing are all generated from source.
This is part 2, after the midpoint-rule video; it uses the same pipeline.

* **Visuals:** matplotlib, frame by frame. Dark canvas and accent colours (3Blue1Brown); serif type,
  range-frame axes, direct labels, margin sidenotes, small multiples, hairline table (Tufte).
* **Voice:** [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural TTS, local, CPU.
* **Sync:** narration drives timing. Each spoken line has a key; the build measures the clip and
  schedules on-screen events from it (`Q(scene, key, frac, off)` in `scenes.py`).
* **Truthfulness:** every number on screen and in the voiceover comes from `mathcore.py` (narration uses
  `{placeholders}`; decimals are spelled out in words so the TTS cannot misread them). `tests/` checks
  each mathematical claim numerically (23 tests).

## Quick start
```bash
brew install ffmpeg              # Debian/Ubuntu: sudo apt install ffmpeg
uv sync                          # once, at the repo root or here (one shared environment)
uv run build.py setup            # download Kokoro model files (~350 MB) into ../models
uv run build.py test             # verify the maths
uv run build.py all --res 4k     # tts -> timeline -> audio -> render -> build/leapfrog_4k.mp4
```
`--res` accepts `720p`, `1080p`, `1440p`, `4k`; use `720p` while iterating.
Stages: `setup tts timeline audio render mux all still test`. `still N SECONDS --res 720p --out f.png`
previews a frame without audio. Flags: `--scenes 4,5`, `--procs N`, `--crf`, `--preset`, `--voice`,
`--speed`, `--silent`, `--force-tts`.

## What is in the video (13 scenes, about 10 minutes)
Euler spirals out on a spring · the staggered grid (kick, drift, half-kick start) · each update is a
midpoint rule · a worked example (Δt = 0.5) · order of accuracy (1, 2, 4) · time reversibility ·
phase-space area · the conserved "shadow" energy · a Kepler orbit, three methods at equal cost ·
a thousand orbits (RK4 error grows, leapfrog's is bounded; crossover ≈ 45 orbits) · limits (phase error,
naive adaptive steps, velocity-dependent forces) · summary with Yoshida's 4th-order composition.

## Editing
* **Words:** `narration.json`. Spell maths for the voice; use `{w_...}` placeholders (see
  `spoken_facts` in `mathcore.py`). Only changed lines are re-synthesised.
* **Visuals:** `leapfrog_video/scenes.py`; each scene is a pure function of local time.
* **Maths / parameters:** `leapfrog_video/mathcore.py` (eccentricity, steps, run lengths). Heavy Kepler
  runs are cached in `build/cache_kepler.pkl` (about 3 s to recompute; bump `_VER` if you change them).
* **Tools:** `uv run --extra asr tools_asr_check.py` transcribes every clip with Whisper and diffs it against the script.

## Layout
```
build.py  narration.json  tools_asr_check.py
leapfrog_video/{mathcore,scenes,narration,render}.py
tests/test_math.py
build/                generated (git-ignored)
../fonts/ ../models/  shared with the other videos (TeX Gyre Pagella is bundled; models are downloaded)
```

## Caveats
* Experiments use one Kepler orbit (e = 0.5). The RK4-versus-leapfrog crossover (≈ 45 orbits) depends on
  that setup and on RK4's step (0.05); the video says so. At 4× the leapfrog step RK4 is *not* used
  for the comparison because it becomes unstable there, which would be a strawman.
* The shadow-energy result is exact only for the linear spring; for general forces it is an asymptotic
  statement (backward error analysis), as the video says.
* The variable-step comparison uses different step sizes, so only its shape (growth) is meaningful.
* TTS is intelligible (checked with Whisper) but not a human voice.
* Licences: Kokoro-82M Apache-2.0, `kokoro-onnx` MIT, TeX Gyre GUST Font License. Verify before publishing.
