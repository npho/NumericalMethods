# AGENTS.md

Guidance for contributors and coding assistants working in this repository.

## Repository shape
One folder per video, plus shared assets and one uv environment at the root. Video folders share no
Python code with each other.

| Folder | Package | Output |
|---|---|---|
| `midpoint-rule/` | `midpoint_video/` | `build/midpoint_rule_<res>.mp4` |
| `leapfrog/` | `leapfrog_video/` | `build/leapfrog_<res>.mp4` |
| `euler/` | `euler_video/` | `build/euler_<res>.mp4` |
| `rk4/` | `rk4_video/` | `build/rk4_<res>.mp4` |
| `ieee754/` | none (`build.py` + `src/`) | `output/ieee754_in_motion_<W>x<H>.mp4` |

## Shared at the root
- `pyproject.toml`, `uv.lock`, `.python-version` (3.13): one environment, the superset of every video's
  dependencies. `[tool.uv] package = false`. `uv run` inside a video folder finds it by walking up.
  Add a dependency here, with a comment naming the video that needs it.
- `fonts/`: every bundled font with its licence (`fonts/README.txt`). Rendering never depends on system
  fonts or network. matplotlib videos load `../../fonts` relative to their package; ieee754's CSS uses
  `../../fonts` relative to `src/` and `build/`.
- `models/`: the Kokoro model, downloaded once and used by every video (git-ignored; may be a symlink,
  so the ignore pattern has no trailing slash). Each `build.py` defines it as `ROOT/../models`.
- `.gitignore` for `.venv/`, `models` and Python caches; each video's `.gitignore` covers its own outputs.

## Every video folder has
- `build.py`: the only entry point. Run it as `uv run build.py <stage>`.
- `narration.json`: the script. Spoken text is the source of truth for timing.
- `README.md` with the sections, in order: summary bullets, Quick start, What is in the video, Editing,
  Layout, Caveats.

## matplotlib projects (midpoint-rule, leapfrog, euler, rk4)
- Package modules: `mathcore.py` (all maths and `FACTS`), `scenes.py` (drawing), `narration.py` (TTS,
  timeline, mix), `render.py` (parallel x264 encode, concat, mux).
- `render.py` splits every scene into `CHUNK`-frame work units (a multiple of the 2 s GOP, so chunks join
  by stream copy) and caps x264 at `THREADS` per worker, since the pool already fills every core.
- `narration.py`, `render.py` and `tools_asr_check.py` are **byte-identical** across the four projects.
  `build.py` differs only in the package name, docstring title and output filename. Keep it that way:
  apply a change to all four copies, then check with `md5 */*_video/narration.py`.
- The `configure()` font block in `scenes.py` is also identical across projects. It registers the
  root `fonts/*.otf` with `font_manager.fontManager.addfont` before checking for `TeX Gyre Pagella`.
- Every number on screen or in narration must come from `mathcore.py` (`{placeholder}` in
  `narration.json`), and each claim needs a test in `tests/test_math.py`. Run `uv run build.py test`
  in a video folder, or `uv run pytest` at the root for all four. The root pytest config lists each
  folder in `pythonpath` and uses `--import-mode=importlib` because every test file is `test_math.py`.
  Tests must locate files from `__file__` (a `ROOT` constant), never from the working directory.
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
- Do not add apt-only or Linux-path assumptions. Put bundled assets in the shared root folders.
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
