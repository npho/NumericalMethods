"""Narration: script -> speech (Kokoro TTS) -> timeline -> mixed audio track.

The timeline is the single source of truth for timing. Each spoken line has a
key; scenes.py asks the timeline "when does line X start / how long is it?" and
schedules its animations from that, so picture follows voice.
"""
import hashlib, json, math, os, urllib.request
import numpy as np

from .mathcore import FACTS

SR = 24000                      # Kokoro output rate
FPS = 30
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]


def load_script(path="narration.json"):
    with open(path, encoding="utf-8") as fh:
        script = json.load(fh)
    for sc in script["scenes"]:
        for ln in sc["lines"]:
            ln["spoken"] = ln["text"].format(**FACTS)       # fill numbers from mathcore
    return script


def _hash(text, voice, speed, lang):
    return hashlib.sha1(f"{voice}|{speed}|{lang}|{text}".encode()).hexdigest()[:10]


def clip_path(build, si, ln, script, voice=None, speed=None):
    voice = voice or script["voice"]; speed = speed or script["speed"]
    return os.path.join(build, "audio", f"{si:02d}_{ln['key']}_{_hash(ln['spoken'], voice, speed, script['lang'])}.wav")


def setup_models(models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    for name in MODEL_FILES:
        dst = os.path.join(models_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            print("have", dst); continue
        print("downloading", name, "...")
        urllib.request.urlretrieve(MODEL_URL + name, dst)


def _trim(x, thresh=0.004, pad=0.06):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0: return x
    a = max(0, idx[0] - int(pad * SR)); b = min(len(x), idx[-1] + int(pad * SR))
    return x[a:b]


def synth_all(script, build="build", models_dir="models", voice=None, speed=None, force=False):
    import soundfile as sf
    from kokoro_onnx import Kokoro
    voice = voice or script["voice"]; speed = speed or script["speed"]
    kokoro = None
    os.makedirs(os.path.join(build, "audio"), exist_ok=True)
    for si, sc in enumerate(script["scenes"]):
        for ln in sc["lines"]:
            p = clip_path(build, si, ln, script, voice, speed)
            if os.path.exists(p) and not force: continue
            if kokoro is None:
                kokoro = Kokoro(os.path.join(models_dir, MODEL_FILES[0]), os.path.join(models_dir, MODEL_FILES[1]))
            x, sr = kokoro.create(ln["spoken"], voice=voice, speed=speed, lang=script["lang"])
            assert sr == SR, sr
            x = _trim(np.asarray(x, dtype=np.float32))
            sf.write(p, x, SR, subtype="PCM_16")
            print(f"  [{si}:{ln['key']}] {len(x) / SR:5.1f}s  {ln['spoken'][:70]}")


def _estimate(text):                       # used only when audio hasn't been synthesised yet
    return max(1.5, len(text.split()) / 2.6)


def build_timeline(script, build="build", voice=None, speed=None, estimate=False):
    """Returns the timeline dict and writes build/timeline.json (unless estimating)."""
    import soundfile as sf
    d = script["defaults"]
    scenes, cursor = [], 0.0
    for si, sc in enumerate(script["scenes"]):
        lead = sc.get("lead", d["lead"]); tail = sc.get("tail", d["tail"])
        t = lead; cues = {}
        for k, ln in enumerate(sc["lines"]):
            p = clip_path(build, si, ln, script, voice, speed)
            if os.path.exists(p):
                info = sf.info(p); dur = info.frames / info.samplerate
            elif estimate:
                dur = _estimate(ln["spoken"]); p = None
            else:
                raise FileNotFoundError(f"missing audio {p}; run `python build.py tts`")
            cues[ln["key"]] = dict(start=round(t, 3), dur=round(dur, 3), file=p, text=ln["spoken"])
            t += dur + ln.get("gap", d["gap"])
        end = t - d["gap"] + tail
        duration = math.ceil(end * FPS) / FPS                        # whole frames
        last = list(cues.values())[-1]
        anim_end = min(duration, last["start"] + last["dur"] + 3.0)  # frames after this are static
        scenes.append(dict(index=si, name=sc["name"], start=round(cursor, 4), duration=duration,
                           anim_end=round(anim_end, 3), cues=cues))
        cursor += duration
    tl = dict(fps=FPS, total=round(cursor, 4), scenes=scenes)
    if not estimate:
        os.makedirs(build, exist_ok=True)
        with open(os.path.join(build, "timeline.json"), "w") as fh: json.dump(tl, fh, indent=1)
    return tl


def mix_audio(tl, build="build"):
    """Place every clip at its absolute time on one mono track -> build/narration.wav"""
    import soundfile as sf
    total = int((tl["total"] + 1.0) * SR)
    buf = np.zeros(total, dtype=np.float32)
    fade = int(0.015 * SR)
    for sc in tl["scenes"]:
        for c in sc["cues"].values():
            x, sr = sf.read(c["file"], dtype="float32"); assert sr == SR
            x = x.copy(); x[:fade] *= np.linspace(0, 1, fade); x[-fade:] *= np.linspace(1, 0, fade)
            a = int((sc["start"] + c["start"]) * SR)
            buf[a:a + len(x)] += x
    peak = float(np.abs(buf).max())
    if peak > 0.98: buf *= 0.98 / peak
    out = os.path.join(build, "narration.wav")
    sf.write(out, buf, SR, subtype="PCM_16")
    return out
