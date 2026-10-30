# IEEE 754 in Motion: a narrated explainer video

A fully scripted, regenerable video on 32-bit floating point. An HTML/SVG page of five scenes is
screenshotted slide by slide with headless Chromium, narrated with a local neural voice, and timed from
the narration. Unlike the matplotlib videos in this repo, slides are stills joined by crossfades.

* **Visuals:** `src/scenes.html` draws each scene as SVG in JavaScript. It is also an interactive page:
  open it in a browser and step with Space or the arrow keys.
* **Voice:** [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural TTS, local, CPU. Each clip is
  trimmed, peak-limited and matched to the same loudness.
* **Sync:** each slide is fully visible for `max(min-hold, lead + narration + tail)`, so speech never
  overlaps a crossfade. `verify` checks this on the finished file.
* **Truthfulness:** `narration` stops if a caption in `narration.json` differs from the slide it belongs to.

## Quick start
```bash
brew install ffmpeg              # Debian/Ubuntu: sudo apt install ffmpeg
uv sync
uv run playwright install chromium
uv run build.py                  # models -> narration -> render -> video -> verify, 1080p
uv run build.py --scale 2        # same at 4K -> output/ieee754_in_motion_3840x2160.mp4
```
Run one step with `uv run build.py <step>`. After editing narration only, run `narration` then `video`;
after editing slides, run `render` then `video`. Behind a TLS-intercepting proxy, set `SSL_CERT_FILE`.

| Step | Does | Output |
|---|---|---|
| `models` | downloads the Kokoro model and voices once (~350 MB) | `models/` |
| `narration` | checks captions, speaks each line, trims, limits, normalises loudness; caches unchanged lines | `build/narration/` |
| `render` | screenshots every slide and the title-intro frames; `--scale 2` redraws at 3840×2160 | `build/frames_WxH/` |
| `video` | builds the timeline, crossfades, mixes audio, encodes H.264 + AAC | `output/*.mp4` |
| `verify` | checks resolution, duration, silence in fades, speech timing, loudness, slide stillness | PASS/FAIL, exit 1 on failure |

Options: `--min-hold 3.0`, `--fade 0.6`, `--lead 0.4`, `--lead-first 0.6`, `--tail 0.6`,
`--intro-hold 2.5`, `--intro-move 1.4`, `--intro-title-px 156`, `--voice af_heart`, `--speed 1.0`,
`--lufs -17`, `--peak 0.84`, `--preset`, `--crf`, `--fps 30`, `--out`. Timing options affect `video`
only; `--voice`, `--speed`, `--lufs` and `--peak` are part of the narration cache key.

## What is in the video (5 scenes, 25 slides)
The 32 bits (sign, exponent, fraction) · decoding 6.25 field by field · gaps grow with magnitude
(a toy 3-bit format, then binary32) · why 0.1 + 0.2 returns 0.30000000000000004 · special values
(subnormals, infinity, NaN, ±0).

## Editing
* **Words:** `spoken` in `narration.json`. Spell out numbers and symbols for the voice
  ("zero point one", "double equals").
* **Slides:** the `scenes` array in `src/scenes.html`. Each scene has an `svg()` function that reveals
  parts as the step counter `st` advances (`data-from`), and a `caps` list with one caption per step.
  Keep each `caption` in `narration.json` identical to its slide; the slide count is the sum of all `caps`.
* **Voice:** list voices with `uv run python -c "from kokoro_onnx import Kokoro; print(sorted(Kokoro('models/kokoro-v1.0.onnx','models/voices-v1.0.bin').get_voices()))"`,
  then `uv run build.py narration --voice am_michael` and `uv run build.py video`.
* **Look:** CSS variables at the top of `src/scenes.html`; the video frame is `src/video_layout.css`.

## Layout
```
build.py  narration.json  pyproject.toml  uv.lock
src/scenes.html  src/video_layout.css
fonts/                STIX Two Text, JetBrains Mono, STIXGeneral, DejaVu Sans Mono (bundled, see fonts/README.txt)
models/ build/ output/    generated (git-ignored)
```

## Caveats
* 4K is rendered, not upscaled: the layout is 1920×1080 CSS px and Playwright's `device_scale_factor` is 2.
* Renders are not bit-identical between runs (Chromium antialiasing varies by a few levels); this is invisible.
* Slide contents appear all at once per step; only the crossfades and the title glide move.
* TTS is intelligible but not a human voice. Listen to "NaN" (spoken "nan") and the long numbers.
* 4K H.264 (level 5.1) plays on most modern devices; some older TVs and editors reject it.
* Licences: fonts per `fonts/README.txt`; Kokoro-82M Apache-2.0, `kokoro-onnx` MIT. Verify before publishing.
