#!/usr/bin/env python3
"""Optional QA: transcribe every synthesised clip with Whisper and report word-level mismatches
against narration.json.  Needs the `asr` extra (downloads a ~250 MB Whisper model on first use).
    uv run --extra asr tools_asr_check.py
Expect ~2% mismatch from recognizer formatting ("0.94" -> "zero ninety four"); look for real misreadings."""
import json, re, difflib
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

tl = json.load(open("build/timeline.json"))
model = WhisperModel("small.en", device="cpu", compute_type="int8")
NUM = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))


def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)} point {m.group(2)}", s)
    s = re.sub(r"\d", lambda m: " " + NUM[m.group(0)] + " ", s)
    return re.findall(r"[a-z']+", s)


tot = err = 0
for sc in tl["scenes"]:
    for key, c in sc["cues"].items():
        x, sr = sf.read(c["file"], dtype="float32")
        x = resample_poly(x, 2, 3).astype("float32")          # 24 kHz -> 16 kHz
        segs, _ = model.transcribe(x, language="en", beam_size=5)
        hyp, ref = norm(" ".join(s.text for s in segs)), norm(c["text"])
        e = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp).get_opcodes() if t != "equal")
        tot += len(ref); err += e
        if e >= 2: print(f"[{sc['index']}:{key}] diff={e}\n  REF {' '.join(ref)}\n  ASR {' '.join(hyp)}")
print(f"word-level mismatch {err}/{tot} = {100 * err / tot:.1f}%")
