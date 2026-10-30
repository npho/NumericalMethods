#!/usr/bin/env python3
"""Build the 'Midpoint Rule' explainer video.

    python build.py setup                 # download the Kokoro TTS model (~350 MB, once)
    python build.py all --res 4k          # tts -> timeline -> audio -> render -> final mp4
    python build.py still 4 12.5 --res 720p --out s4.png   # preview one frame (no audio needed)

Individual stages: tts, timeline, audio, render, mux, test.
"""
import argparse, json, os, sys

from midpoint_video import narration as N

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def load_tl(script, estimate=False):
    return N.build_timeline(script, "build", estimate=estimate)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["setup", "tts", "timeline", "audio", "render", "mux", "all", "still", "test"])
    ap.add_argument("args", nargs="*", help="still: SCENE SECONDS")
    ap.add_argument("--res", default="4k", choices=["720p", "1080p", "1440p", "4k"])
    ap.add_argument("--procs", type=int, default=None, help="render worker processes (default: all cores)")
    ap.add_argument("--crf", type=int, default=17, help="x264 quality, lower = better (default 17)")
    ap.add_argument("--preset", default="fast", help="x264 preset (default fast)")
    ap.add_argument("--voice", default=None, help="Kokoro voice, e.g. af_heart, am_michael, bf_emma, bm_george")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--force-tts", action="store_true", help="re-synthesise even if cached")
    ap.add_argument("--scenes", default=None, help="render only these scene indices, e.g. 4,5")
    ap.add_argument("--silent", action="store_true", help="mux without narration")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    script = N.load_script("narration.json")
    if a.cmd == "setup":
        N.setup_models("models"); return
    if a.cmd == "test":
        sys.exit(os.system(f"{sys.executable} -m pytest -q tests"))
    if a.cmd in ("tts", "all"):
        N.synth_all(script, "build", "models", a.voice, a.speed, a.force_tts)
    if a.cmd == "tts": return
    if a.cmd == "still":
        from midpoint_video import scenes, render
        scenes.configure(render.RES[a.res])
        scenes.set_timeline(load_tl(script, estimate=True) if not os.path.exists("build/timeline.json")
                            else json.load(open("build/timeline.json")))
        idx, t = int(a.args[0]), float(a.args[1])
        scenes.new_frame(); scenes.SCENES[idx](t)
        scenes.fig.savefig(a.out or f"still_{idx}_{t:g}.png", dpi=scenes.DPI, facecolor=scenes.BG)
        return
    tl = N.build_timeline(script, "build", a.voice, a.speed)
    print(f"timeline: {tl['total']:.1f}s total; " + ", ".join(f"{s['name']} {s['duration']:.1f}s" for s in tl["scenes"]))
    if a.cmd == "timeline": return
    wav = N.mix_audio(tl, "build")
    print("narration track:", wav)
    if a.cmd == "audio": return
    from midpoint_video import render
    chunks = os.path.join("build", f"chunks_{a.res}")
    if a.cmd in ("render", "all"):
        only = {int(x) for x in a.scenes.split(",")} if a.scenes else None
        render.render_all(tl, chunks, a.res, a.procs, a.crf, a.preset, only)
    if a.cmd in ("mux", "all"):
        out = a.out or os.path.join("build", f"midpoint_rule_{a.res}.mp4")
        render.concat_and_mux(tl, chunks, None if a.silent else wav, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
