import shutil
import subprocess
import time

import numpy as np
import pytest

from triangulate.acoustic.spectrogram import (
    SR,
    _bg_subtract,
    compute_spectrogram,
    log_resample,
    power_db,
    whiten,
)
from triangulate.ingest.media import decode_audio


def _scene(dur=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = 0.01 * rng.standard_normal(len(t))
    for f in (60.0, 125.0, 250.0):
        x += 0.1 * np.sin(2 * np.pi * f * t)
    m = (t > 300) & (t < 340)
    x += 0.02 * np.sin(2 * np.pi * 150.0 * t) * m
    for c in rng.integers(0, len(t) - 50, 30):
        x[c : c + 20] += rng.standard_normal(20) * 0.8
    return x.astype(np.float32), t


def test_tone_and_hum():
    x, _ = _scene()
    s = compute_spectrogram(x)
    ti = np.argmin(abs(s.t - 320))
    assert abs(s.f[np.argmax(s.Z[ti].astype(float))] - 150.0) <= 1.0
    # hum is louder than the tone in raw dB but suppressed >= 20 dB relative to it after whitening
    raw = lambda f0: s.db[ti, np.argmin(abs(s.f - f0))]  # noqa: E731
    assert raw(125.0) > raw(150.0) + 5
    r = _bg_subtract(s.db, s.t[1] - s.t[0])
    res = lambda f0: r[ti, np.argmin(abs(s.f - f0))]  # noqa: E731
    for h in (60.0, 125.0, 250.0):
        assert res(150.0) - res(h) >= 20
    zr = whiten(s.db, s.t[1] - s.t[0], s.f, clip=False)
    assert zr[ti, np.argmin(abs(s.f - 125.0))] < 3


def test_log_resample():
    x, _ = _scene(120)
    ls = log_resample(compute_spectrogram(x))
    assert ls.Z.shape == (len(ls.t), len(ls.u)) and abs(ls.t[1] - ls.t[0] - 0.5) < 1e-6


@pytest.mark.slow
def test_one_hour_speed():
    x = np.random.default_rng(1).standard_normal(3600 * SR).astype(np.float32) * 0.01
    t0 = time.time()
    compute_spectrogram(x)
    assert time.time() - t0 <= 15


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg")
def test_mp3_roundtrip(tmp_path):
    x, _ = _scene(300, 3)
    x[: 0] = 0
    t = np.arange(len(x)) / SR
    x += (0.03 * np.sin(2 * np.pi * 150 * t) * ((t > 100) & (t < 140))).astype(np.float32)
    wav, mp3 = tmp_path / "a.f32", tmp_path / "a.mp3"
    wav.write_bytes(x.astype("<f4").tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(wav),
                    "-c:a", "libmp3lame", "-b:a", "64k", str(mp3)], check=True)
    y = decode_audio(mp3)
    assert abs(len(y) - len(x)) < SR  # codec delay/padding

    def snr(sig):
        t_, f, db = power_db(sig)
        r = _bg_subtract(db, 0.125)
        ti = (t_ > 105) & (t_ < 135)
        k = np.argmin(abs(f - 150))
        return r[ti, k].mean() - np.median(r[:, k])

    assert snr(x) - snr(y) <= 3.0


@pytest.mark.slow
def test_audio_mp3_length():
    from triangulate.config import project_root

    p = project_root() / "audio.mp3"
    if not p.exists():
        pytest.skip("audio.mp3 absent")
    assert abs(len(decode_audio(p)) / SR - 3600) <= 1
