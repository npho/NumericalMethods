"""Parallel video rendering: one H.264 chunk per scene, then concat + audio mux."""
import json, os, subprocess
from multiprocessing import Pool

from . import scenes

RES = {"720p": 1280, "1080p": 1920, "1440p": 2560, "4k": 3840}


def _init(width, tl):
    scenes.configure(width)
    scenes.set_timeline(tl)


def _render_scene(args):
    idx, outdir, crf, preset = args
    sc = scenes.TL["scenes"][idx]
    fps = scenes.FPS
    n = int(round(sc["duration"] * fps))
    out = os.path.join(outdir, f"scene{idx}.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{scenes.W}x{scenes.H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-tune", "animation", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-color_primaries", "bt709", "-color_trc", "bt709",
         "-colorspace", "bt709", out], stdin=subprocess.PIPE)
    last_key, last = None, None
    for k in range(n):
        t = k / fps
        te = min(t, sc["anim_end"])                 # after the last animation the frame is static: reuse it
        key = (round(te, 5), round(scenes.fade_val(idx, t), 4))
        if key != last_key:
            last = scenes.render_frame(idx, te, t)
            last_key = key
        ff.stdin.write(last)
    ff.stdin.close(); ff.wait()
    return out


def render_all(tl, outdir, res="4k", procs=None, crf=17, preset="fast", only=None):
    width = RES[res]
    os.makedirs(outdir, exist_ok=True)
    procs = procs or os.cpu_count() or 2
    order = sorted(range(len(tl["scenes"])), key=lambda i: -tl["scenes"][i]["duration"])
    if only is not None: order = [i for i in order if i in only]
    with Pool(procs, initializer=_init, initargs=(width, tl)) as pool:
        for r in pool.imap_unordered(_render_scene, [(i, outdir, crf, preset) for i in order]):
            print("rendered", r, flush=True)


def concat_and_mux(tl, chunkdir, wav, out_mp4):
    lst = os.path.join(chunkdir, "list.txt")
    with open(lst, "w") as fh:
        for sc in tl["scenes"]: fh.write(f"file 'scene{sc['index']}.mp4'\n")
    silent = os.path.join(chunkdir, "silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent],
                   check=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", silent]
    if wav:
        cmd += ["-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-shortest"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)
    return out_mp4
