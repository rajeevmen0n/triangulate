"""Frame selection + dedupe (plan 3.6 step 1). Local only, no network.

Public API:
    extract_frames(video, out_dir, *, max_frames=200, ...) -> list[Frame]
    dhash64(image) -> int ; hamming(a, b) -> int
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

SCENE_THRESHOLD = 0.3
MIN_INTERVAL_S = 2.0
TARGET_UNIFORM_FRAMES = 300
HASH_DIST = 6
THUMB_LONG_EDGE = 1280


class Frame(BaseModel):
    id: str  # "f0001"
    t_s: float  # position in the video (s)
    path: str  # thumbnail jpg (<= 1280 px long edge)
    dhash: str  # 16 hex chars
    source: str = "uniform"  # "scene" | "uniform" (best guess)


def dhash64(img: np.ndarray) -> int:
    """64-bit difference hash of a BGR/gray image (9x8 resize, horizontal gradient)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _duration_s(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if fps and n and fps > 0:
        return float(n / fps)
    raise RuntimeError(f"cannot determine duration of {video}")


def uniform_interval(duration_s: float) -> float:
    return max(MIN_INTERVAL_S, duration_s / TARGET_UNIFORM_FRAMES)


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    max_frames: int = 200,
    scene_threshold: float = SCENE_THRESHOLD,
    hash_dist: int = HASH_DIST,
    long_edge: int = THUMB_LONG_EDGE,
) -> list[Frame]:
    """Scene-change frames + uniform frames, dHash de-duplicated, capped at `max_frames`.

    Thumbnails are written to `out_dir/fNNNN.jpg`; an index `frames.json` is written alongside.
    """
    video, out = Path(video), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("f*.jpg"):
        old.unlink()
    dur = _duration_s(video)
    iv = uniform_interval(dur)
    sel = (
        f"gt(scene,{scene_threshold})+isnan(prev_selected_t)"
        f"+gte(t-prev_selected_t,{iv:.3f})"
    )
    scale = f"scale='if(gt(iw,ih),min({long_edge},iw),-2)':'if(gt(iw,ih),-2,min({long_edge},ih))'"
    vf = f"select='{sel}',{scale},showinfo"
    tmp = out / "raw_%05d.jpg"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", str(video), "-an", "-sn",
         "-vf", vf, "-fps_mode", "vfr", "-q:v", "3", str(tmp)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")
    times = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", proc.stderr)]
    raws = sorted(out.glob("raw_*.jpg"))
    times = times[: len(raws)] + [0.0] * max(0, len(raws) - len(times))

    kept: list[tuple[float, Path, int]] = []
    for t, p in zip(times, raws, strict=False):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            p.unlink(missing_ok=True)
            continue
        h = dhash64(img)
        if any(hamming(h, k[2]) <= hash_dist for k in kept):
            p.unlink(missing_ok=True)
            continue
        kept.append((t, p, h))

    if len(kept) > max_frames:  # evenly thin, keeping first and last
        idx = sorted({int(round(i)) for i in np.linspace(0, len(kept) - 1, max_frames)})
        drop = set(range(len(kept))) - set(idx)
        for i in drop:
            kept[i][1].unlink(missing_ok=True)
        kept = [kept[i] for i in idx]

    frames: list[Frame] = []
    for n, (t, p, h) in enumerate(kept, 1):
        dst = out / f"f{n:04d}.jpg"
        p.rename(dst)
        frames.append(Frame(id=f"f{n:04d}", t_s=round(t, 3), path=str(dst), dhash=f"{h:016x}"))
    (out / "frames.json").write_text(json.dumps([f.model_dump() for f in frames], indent=1))
    return frames
