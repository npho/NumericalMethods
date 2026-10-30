"""Parallel video rendering: scenes split into equal frame chunks, encoded with H.264, then concat + audio mux."""
import json, os, subprocess
from multiprocessing import Pool

from . import scenes

RES = {"720p": 1280, "1080p": 1920, "1440p": 2560, "4k": 3840}
CHUNK = 300        # frames per work unit; a multiple of the GOP (2 s) so chunks join on keyframes
THREADS = 2        # x264 threads per worker; the pool already uses every core


def _init(width, tl):
    scenes.configure(width)
    scenes.set_timeline(tl)


def _chunk_path(outdir, idx, k0):
    return os.path.join(outdir, f"scene{idx}_{k0:06d}.mp4")


def _render_chunk(args):
    idx, k0, k1, outdir, crf, preset, threads = args
    sc = scenes.TL["scenes"][idx]
    fps = scenes.FPS
    out = _chunk_path(outdir, idx, k0)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{scenes.W}x{scenes.H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-tune", "animation", "-crf", str(crf),
         "-threads", str(threads), "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-color_primaries", "bt709",
         "-color_trc", "bt709", "-colorspace", "bt709", out], stdin=subprocess.PIPE)
    last_key, last = None, None
    for k in range(k0, k1):
        t = k / fps
        te = min(t, sc["anim_end"])                 # after the last animation the frame is static: reuse it
        key = (round(te, 5), round(scenes.fade_val(idx, t), 4))
        if key != last_key:
            last = scenes.render_frame(idx, te, t)
            last_key = key
        ff.stdin.write(last)
    ff.stdin.close(); ff.wait()
    return out


def _join(paths, out):
    lst = out + ".txt"
    with open(lst, "w") as fh:
        for p in paths: fh.write(f"file '{os.path.basename(p)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   check=True)
    os.remove(lst)


def render_all(tl, outdir, res="4k", procs=None, crf=17, preset="fast", only=None, threads=THREADS):
    width = RES[res]
    os.makedirs(outdir, exist_ok=True)
    procs = procs or os.cpu_count() or 2
    fps = scenes.FPS
    todo = [i for i in range(len(tl["scenes"])) if only is None or i in only]
    chunks = {i: list(range(0, int(round(tl["scenes"][i]["duration"] * fps)), CHUNK)) for i in todo}
    jobs = [(i, k0, min(k0 + CHUNK, int(round(tl["scenes"][i]["duration"] * fps))), outdir, crf, preset, threads)
            for i in todo for k0 in chunks[i]]
    jobs.sort(key=lambda j: j[1] - j[2])           # full chunks first, short tails fill in at the end
    with Pool(procs, initializer=_init, initargs=(width, tl)) as pool:
        for n, r in enumerate(pool.imap_unordered(_render_chunk, jobs), 1):
            print(f"rendered [{n}/{len(jobs)}] {os.path.basename(r)}", flush=True)
    for i in todo:
        parts = [_chunk_path(outdir, i, k0) for k0 in chunks[i]]
        _join(parts, os.path.join(outdir, f"scene{i}.mp4"))
        for p in parts: os.remove(p)


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
