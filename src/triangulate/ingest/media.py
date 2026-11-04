"""Audio decode (plan 3.2 preprocessing). Decodes PCM only; container metadata and tags are never
read or reported, and audio position 0 is the first decoded sample."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 4000


class DecodeError(RuntimeError):
    pass


def decode_audio(path: str | Path, sr: int = SAMPLE_RATE, *, max_seconds: float | None = None) -> np.ndarray:
    """Decode the first audio stream to mono float32 at `sr` Hz via ffmpeg (soxr resampler)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "a:0", "-map_metadata", "-1"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-ac", "1", "-ar", str(sr), "-af", "aresample=resampler=soxr", "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:  # pragma: no cover
        raise DecodeError("ffmpeg not found") from e
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no audio decoded"]
        raise DecodeError(f"ffmpeg failed: {msg[0]}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
