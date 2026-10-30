# AGENTS.md

Guidance for contributors and coding assistants working in this repository.

## Repository shape
One folder per video; each is a self-contained uv project with no shared code between folders.

| Folder | Package | Output |
|---|---|---|
| `midpoint-rule/` | `midpoint_video/` | `build/midpoint_rule_<res>.mp4` |
| `leapfrog/` | `leapfrog_video/` | `build/leapfrog_<res>.mp4` |
| `euler/` | `euler_video/` | `build/euler_<res>.mp4` |
| `rk4/` | `rk4_video/` | `build/rk4_<res>.mp4` |
| `ieee754/` | none (`build.py` + `src/`) | `output/ieee754_in_motion_<W>x<H>.mp4` |

## Every project has
- `build.py`: the only entry point. Run it as `uv run build.py <stage>`.
- `pyproject.toml`, `uv.lock` and `.python-version` (3.13). `[tool.uv] package = false`.
- `narration.json`: the script. Spoken text is the source of truth for timing.
- `fonts/`: bundled fonts with their licence files. Rendering never depends on system fonts or network.
- `.gitignore` ignoring `build/`, `models` (no trailing slash, so a symlink is ignored too), `.venv/` and caches.
- `README.md` with the sections, in order: summary bullets, Quick start, What is in the video, Editing,
  Layout, Caveats.

## matplotlib projects (midpoint-rule, leapfrog, euler, rk4)
- Package modules: `mathcore.py` (all maths and `FACTS`), `scenes.py` (drawing), `narration.py` (TTS,
  timeline, mix), `render.py` (parallel x264 encode, concat, mux).
- `narration.py`, `render.py` and `tools_asr_check.py` are **byte-identical** across the four projects.
  `build.py` differs only in the package name, docstring title and output filename. Keep it that way:
  apply a change to all four copies, then check with `md5 */*_video/narration.py`.
- The `configure()` font block in `scenes.py` is also identical across projects. It registers
  `fonts/*.otf` with `font_manager.fontManager.addfont` before checking for `TeX Gyre Pagella`.
- Every number on screen or in narration must come from `mathcore.py` (`{placeholder}` in
  `narration.json`), and each claim needs a test in `tests/test_math.py`. Run `uv run build.py test`
  (or `uv run pytest`); `pythonpath = ["."]` in `pyproject.toml` lets tests import the package.
- Scenes are pure functions of local time. Time events with `Q(scene, key, frac, off)`, never fixed seconds.
- Rendering uses `multiprocessing.Pool` with an initializer. macOS uses the `spawn` start method, so
  worker state must be set in `render._init`, not inherited from the parent.
- Dependencies: runtime in `[project.dependencies]`, pytest in the `dev` group, Whisper QA in the
  `asr` extra (`uv run --extra asr tools_asr_check.py`).

## ieee754
- Slides live in `src/scenes.html` (also an interactive page); `build.py` screenshots them with Playwright.
- Steps: `models narration render video verify` (default: all). Resolution is `--scale 1|2`, not `--res`.
- Each `caption` in `narration.json` must match its slide caption exactly; `narration` enforces this.
- Needs `uv run playwright install chromium` once.

## Platform
- Development target is macOS (arm64) and Linux. System requirement is only `ffmpeg` with libx264.
- Do not add apt-only or Linux-path assumptions. Prefer bundling assets into the project folder.
- Large generated files (`models/`, `build/`, `output/`, `*.mp4`, `*.wav`) are never committed.

## Verification
- After code changes, run the tests and render a still:
  `uv run build.py still <scene> <seconds> --res 720p --out /path/preview.png` (works without audio).
- Full video rebuilds are slow (minutes at 720p, ~30 min at 4K) and need the Kokoro model.

## Documentation and commits
- Keep READMEs concise; update the project README and, if needed, the top-level README with each change.
- Write in a neutral voice. No first-person asides.
- Commits: one project or concern per commit. Title is one descriptive sentence ending with a period,
  followed by a short dashed list of changes. No AI attribution or co-author trailers.
