# NumericalMethods

Narrated explainer videos on numerical methods, each generated entirely from code: visuals, on-screen
numbers, narration (local Kokoro TTS) and timing.

| Project | Topic | Pipeline |
|---|---|---|
| [`midpoint-rule/`](midpoint-rule/) | Part 1: the midpoint rule for integrals | matplotlib |
| [`leapfrog/`](leapfrog/) | Part 2: leapfrog (kick-drift) integration | matplotlib |
| [`euler/`](euler/) | Part 3: Euler's method for ODEs | matplotlib |
| [`rk4/`](rk4/) | Part 4: fourth-order Runge–Kutta | matplotlib |
| [`ieee754/`](ieee754/) | IEEE 754 floating point | HTML/SVG + Playwright |

## Requirements
[uv](https://docs.astral.sh/uv/) and `ffmpeg` with libx264 (`brew install ffmpeg` or `apt install ffmpeg`).
Fonts are bundled; models (~350 MB) download on first setup.

## Usage
Each folder is a self-contained uv project:
```bash
cd rk4
uv sync
uv run build.py setup     # ieee754: `uv run playwright install chromium`, then `uv run build.py`
uv run build.py all --res 720p
```
See each project's README for stages, options and editing notes.
