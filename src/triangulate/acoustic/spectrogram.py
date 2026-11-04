"""STFT, whitening into the SNR map Z(t, f), and a log-frequency resampler (plan 3.2).

Public API:
    decode_and_whiten(path) / compute_spectrogram(x, sr) -> Spectrogram
    log_resample(spec, du=0.004, dt=0.5) -> LogSpectrogram   (for H(t,u) harmonic stacking)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import median_filter

SR = 4000
WIN = 2048
HOP = 500
NFFT = 4096
FMIN, FMAX = 20.0, 1000.0
Z_CLIP = 12.0


@dataclass
class Spectrogram:
    t: np.ndarray  # (T,) frame-centre times, s from the first sample
    f: np.ndarray  # (F,) Hz
    Z: np.ndarray  # (T, F) float16 whitened SNR map, clipped to [0, 12]
    db: np.ndarray  # (T, F) float32 raw power in dB (re full scale^2, not whitened)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else HOP / SR

    @property
    def df(self) -> float:
        return float(self.f[1] - self.f[0])


@dataclass
class LogSpectrogram:
    t: np.ndarray  # (T',) s
    u: np.ndarray  # (U,) ln f
    Z: np.ndarray  # (T', U) float32

    @property
    def f(self) -> np.ndarray:
        return np.exp(self.u)


def power_db(x: np.ndarray, sr: int = SR, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann STFT (2048, hop 500, zero-padded to 4096), 20-1000 Hz, power in dB. -> (t, f, db[T,F])."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n_frames = 1 + (len(x) - WIN) // HOP
    freqs = np.fft.rfftfreq(NFFT, 1.0 / sr)
    sel = np.where((freqs >= FMIN) & (freqs <= FMAX))[0]
    win = np.hanning(WIN + 1)[:-1].astype(np.float32)  # periodic Hann
    scale = 1.0 / (win.sum() ** 2)
    db = np.empty((n_frames, len(sel)), dtype=np.float32)
    view = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP]
    for i in range(0, n_frames, chunk):
        seg = view[i : i + chunk] * win
        sp = sfft.rfft(seg, n=NFFT, axis=1, workers=-1)[:, sel]
        p = (sp.real**2 + sp.imag**2) * scale
        db[i : i + chunk] = 10.0 * np.log10(p + 1e-14)
    t = (np.arange(n_frames) * HOP + WIN / 2) / sr
    return t, freqs[sel], db


def _bg_subtract(db: np.ndarray, dt: float, bg_s: float = 300.0, blk_s: float = 5.0) -> np.ndarray:
    """Step 1: per-bin running median over ~5 min (computed on 5-s block medians, then interpolated)."""
    T, F = db.shape
    nb = max(1, int(round(blk_s / dt)))
    nblk = max(1, T // nb)
    blocks = np.median(db[: nblk * nb].reshape(nblk, nb, F), axis=1)  # (nblk, F)
    if T > nblk * nb:  # ragged tail
        blocks = np.vstack([blocks, np.median(db[nblk * nb :], axis=0)])
    k = max(1, int(round(bg_s / blk_s)) | 1)
    k = min(k, len(blocks) | 1)
    bg = median_filter(blocks, size=(k, 1), mode="nearest")
    centres = (np.arange(len(blocks)) + 0.5) * nb
    centres[-1] = min(centres[-1], T - 1) if T > nblk * nb else centres[-1]
    idx = np.arange(T, dtype=np.float64)
    out = np.empty_like(db)
    # linear interpolation in time, per bin (vectorised via two-neighbour gather)
    j = np.clip(np.searchsorted(centres, idx) - 1, 0, max(len(blocks) - 2, 0))
    j2 = np.minimum(j + 1, len(blocks) - 1)
    span = np.maximum(centres[j2] - centres[j], 1e-9)
    w = np.clip((idx - centres[j]) / span, 0.0, 1.0)[:, None] if len(blocks) > 1 else np.zeros((T, 1))
    out[:] = db - ((1 - w) * bg[j] + w * bg[j2])
    return out


def whiten(db: np.ndarray, dt: float, f: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Plan 3.2 whitening. Returns Z (T,F) float32 (clipped to [0,12] if `clip`)."""
    r = _bg_subtract(db, dt)  # 1. stationary tones
    r -= np.median(r, axis=1, keepdims=True)  # 2. broadband clicks / wind
    T, F = r.shape
    bt = max(1, int(round(10.0 / dt)))
    bf = max(1, int(round(50.0 / (f[1] - f[0]))))
    Tp, Fp = -(-T // bt) * bt, -(-F // bf) * bf
    rp = np.pad(r, ((0, Tp - T), (0, Fp - F)), mode="edge")
    blk = rp.reshape(Tp // bt, bt, Fp // bf, bf).transpose(0, 2, 1, 3).reshape(Tp // bt, Fp // bf, -1)
    med = np.median(blk, axis=2, keepdims=True)
    sigma = 1.4826 * np.median(np.abs(blk - med), axis=2)
    sigma = np.maximum(sigma, 0.25)  # dB floor: avoids blow-up in digital silence
    s_full = np.repeat(np.repeat(sigma, bt, axis=0), bf, axis=1)[:T, :F]
    z = r / s_full  # 3. robust sigma
    return np.clip(z, 0.0, Z_CLIP) if clip else z  # 4.


def compute_spectrogram(x: np.ndarray, sr: int = SR) -> Spectrogram:
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio")
    t, f, db = power_db(x, sr)
    dt = HOP / sr
    Z = whiten(db, dt, f).astype(np.float16)
    return Spectrogram(t=t, f=f, Z=Z, db=db)


def decode_and_whiten(path: str | Path) -> Spectrogram:
    from ..ingest.media import decode_audio

    return compute_spectrogram(decode_audio(path, SR), SR)


def log_resample(
    spec: Spectrogram, du: float = 0.004, dt: float = 0.5,
    fmin: float | None = None, fmax: float | None = None,
) -> LogSpectrogram:
    """Resample Z onto u = ln f (step `du`, linear interpolation) and mean-pool time to `dt` seconds."""
    lo = max(fmin or spec.f[0], spec.f[0])
    hi = min(fmax or spec.f[-1], spec.f[-1])
    u = np.arange(np.log(lo), np.log(hi), du)
    pos = (np.exp(u) - spec.f[0]) / spec.df
    i0 = np.clip(np.floor(pos).astype(int), 0, len(spec.f) - 2)
    w = (pos - i0).astype(np.float32)
    n = max(1, int(round(dt / spec.dt)))
    T = (len(spec.t) // n) * n
    Z = spec.Z[:T].astype(np.float32).reshape(T // n, n, -1).mean(axis=1)
    out = Z[:, i0] * (1 - w) + Z[:, i0 + 1] * w
    t = spec.t[:T].reshape(T // n, n).mean(axis=1)
    return LogSpectrogram(t=t, u=u, Z=out.astype(np.float32))
