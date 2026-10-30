# The Midpoint Rule: a 3Blue1Brown × Tufte explainer video

A fully scripted, regenerable video on numerical integration with the midpoint rule. Animation, on-screen
numbers, narration text, voice and timing are all generated from source. This is part 1 of the series.

* **Visuals:** matplotlib, frame by frame. Dark canvas and accent colours (3Blue1Brown); serif type,
  range-frame axes, direct labels, margin sidenotes, hairline table (Tufte).
* **Voice:** [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural TTS, local, CPU.
* **Sync:** narration drives timing. Each spoken line has a key; the build measures the clip and
  schedules on-screen events from it (`Q(scene, key, frac, off)` in `scenes.py`).
* **Truthfulness:** every number on screen and in the voiceover comes from `mathcore.py` (narration uses
  `{placeholders}`). `tests/` checks each mathematical claim numerically (11 tests).

## Quick start
```bash
brew install ffmpeg              # Debian/Ubuntu: sudo apt install ffmpeg
uv sync
uv run build.py setup            # download Kokoro model files (~350 MB) into ./models
uv run build.py test             # verify the maths
uv run build.py all --res 4k     # tts -> timeline -> audio -> render -> build/midpoint_rule_4k.mp4
```
`--res` accepts `720p`, `1080p`, `1440p`, `4k`; use `720p` while iterating.
Stages: `setup tts timeline audio render mux all still test`. `still N SECONDS --res 720p --out f.png`
previews a frame without audio. Flags: `--scenes 4,5`, `--procs N`, `--crf`, `--preset`, `--voice`,
`--speed`, `--silent`, `--force-tts`.

## What is in the video (8 scenes, about 6.5 minutes)
Title · e^(−x²) on [0, 2] has no elementary antiderivative · left, right and midpoint sums on four strips ·
the recipe, worked by hand · why it is accurate (tilt the top to the tangent; error ≈ f''h³/24 per strip) ·
adding up the strip errors · convergence (slope −1 for endpoints, −2 for midpoint and trapezoid) ·
summary table, with Simpson's rule and the smoothness caveat.

## Editing
* **Words:** `narration.json`. Spell maths for the voice ("f double prime", "h cubed over twenty-four");
  use `{name:.3f}`-style placeholders (see `facts()` in `mathcore.py`). Only changed lines are re-synthesised.
* **Voice and pacing:** `voice` / `speed` in `narration.json` (or `--voice` / `--speed`); per-scene
  `lead` and `tail`, `defaults.gap`, or a per-line `gap`. Visuals re-time to the new clip lengths.
* **Visuals:** `midpoint_video/scenes.py`; each scene is a pure function of local time.
* **Maths / parameters:** `midpoint_video/mathcore.py` (integrand, derivatives, interval). Static text in
  `scenes.py` that is specific to e^(−x²) (labels, the `[1.0, 1.5]` zoom strip) needs updating too.
* **Tools:** `uv run --extra asr tools_asr_check.py` transcribes every clip with Whisper and diffs it
  against the script. Expect ~2% mismatch from number formatting; look for real misreadings.

## Layout
```
build.py  narration.json  pyproject.toml  uv.lock  tools_asr_check.py
midpoint_video/{mathcore,scenes,narration,render}.py
tests/test_math.py
fonts/                TeX Gyre Pagella (bundled, GUST Font License)
models/ build/        generated (git-ignored; models/ is a symlink or a download)
```

## Caveats
* Resolution independence: the figure is always 12.8 × 7.2 in and DPI scales, so fonts and lines scale
  together. A full 4K render is about 11.5k frames and roughly 30 minutes on 2 cores.
* Visuals are deterministic. Kokoro output is deterministic for a given model, library version and text;
  an upgrade may shift clip lengths slightly, which the timeline absorbs.
* TTS is intelligible (checked with Whisper, ≈98% word agreement) but not a human voice. Prosody on long
  equations is the weakest spot.
* Licences: Kokoro-82M Apache-2.0, `kokoro-onnx` MIT, TeX Gyre GUST Font License. Verify before publishing.
