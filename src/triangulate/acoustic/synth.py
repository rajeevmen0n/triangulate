"""Synthetic audio from arbitrary source tracks (plan WP06), for tests and calibration (WP08/11/22).

Method (exact retarded Doppler, no frequency integration):

    s(t_r) = sum_h  a_h * (r_ref / r(t_e)) * sin(2 pi h f0 (t_e(t_r) - t_ref) + phi_h)

with t_e from ``physics.solve_emission``. The instantaneous frequency of each term is
h f0 dt_e/dt_r = h f0 c/(c + v_r) exactly. ``a_h`` is the amplitude at range ``r_ref`` (1 km).

Time base: audio sample n is received at t_r = t0 + n/fs, where ``t0`` (= T_s in the plan) is in
the same base as the track times (epoch s or any relative base). Audio position tau = t_r - t0.
Positions are local Cartesian metres (ENU, z up), observer ``obs`` in the same frame.

Background: pink (1/f power) noise, stationary hum lines, broadband clicks, optional MP3 round trip
(ffmpeg libmp3lame) to mimic re-compressed uploads. Every random draw comes from one seeded
``numpy.random.Generator`` so scenes are reproducible.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .physics import C_DEFAULT, Pass, Track, solve_emission, track_passes

__all__ = [
    "FS_DEFAULT", "DEMO_HUM_HZ", "SynthSource", "SourceTruth", "SynthResult",
    "straight_pass_track", "circle_track", "render_source", "source_truth",
    "pink_noise", "hum_lines", "transients", "mp3_roundtrip", "band_power", "synthesize", "write_wav",
]

FS_DEFAULT = 4000  # Hz, the analysis rate of the pipeline (plan 3.2)
DEMO_HUM_HZ = (40.0, 60.0, 125.0, 180.0, 250.0, 360.0)  # stationary lines seen in the test audio (plan 0)


@dataclass(frozen=True, eq=False)
class SynthSource:
    """A tonal source moving along ``track``. ``harmonics[h-1]`` = amplitude of harmonic h at r_ref."""

    track: Track
    f0: float
    harmonics: tuple[float, ...] = (1.0, 0.6, 0.4)
    r_ref: float = 1000.0
    phases: tuple[float, ...] | None = None  # rad per harmonic; None -> random from the scene rng
    taper_s: float = 2.0  # raised-cosine fade (emission time) at both ends of the track
    label: str = ""


@dataclass(frozen=True)
class SourceTruth:
    label: str
    f0: float
    passes: list[Pass]  # absolute times (track base)
    tau_h: list[float]  # heard CPA audio positions (s from sample 0), one per pass


@dataclass(eq=False)
class SynthResult:
    audio: np.ndarray  # float32 (n,), the final mix (after MP3 if requested)
    fs: int
    t0: float  # absolute time of sample 0 (= T_s), track time base
    c: float
    obs: np.ndarray  # (3,)
    truth: list[SourceTruth] = field(default_factory=list)
    signal: np.ndarray | None = None  # float32 sum of tonal sources only (pre-codec), if kept
    background: np.ndarray | None = None  # float32 noise + hum + clicks only (pre-codec), if kept

    @property
    def duration(self) -> float:
        return self.audio.size / self.fs

    def times(self) -> np.ndarray:
        """Audio positions tau (s) of each sample."""
        return np.arange(self.audio.size) / self.fs


# --- Track builders ---------------------------------------------------------------------------


def straight_pass_track(obs, t_h: float, v: float, d: float, *, height: float | None = None,
                        heading_deg: float = 90.0, side: int = 1, half_span_s: float = 300.0,
                        dt: float = 1.0, c: float = C_DEFAULT, label: str = "") -> Track:
    """Constant-velocity level track whose *heard* CPA at ``obs`` is ``t_h`` at slant distance ``d``.

    ``height`` = CPA height above the observer (default min(d, 1000 m)); the horizontal CPA offset
    is sqrt(d^2 - height^2), on the right (side=+1) or left (-1) of the flight direction.
    ``heading_deg`` is clockwise from north (90 = flying east). Samples every ``dt`` seconds over
    t_c +- half_span_s, where t_c = t_h - d/c.
    """
    ob = np.asarray(obs, dtype=float)
    h = min(d, 1000.0) if height is None else float(height)
    if not 0 <= h <= d:
        raise ValueError("need 0 <= height <= d")
    hd = math.radians(heading_deg)
    u = np.array([math.sin(hd), math.cos(hd), 0.0])  # unit velocity (E, N, U)
    right = np.array([u[1], -u[0], 0.0])
    p_c = ob + side * math.sqrt(d * d - h * h) * right + np.array([0.0, 0.0, h])
    t_c = t_h - d / c
    k = int(math.ceil(half_span_s / dt))
    t = t_c + dt * np.arange(-k, k + 1)
    pos = p_c + np.outer(t - t_c, v * u)
    vel = np.tile(v * u, (t.size, 1))
    return Track(t=t, pos=pos, vel=vel, label=label)


def circle_track(center, radius: float, speed: float, t_start: float, duration: float, *,
                 dt: float = 1.0, phase0_deg: float = 0.0, clockwise: bool = False,
                 climb_rate: float = 0.0, label: str = "") -> Track:
    """Constant-speed circle (orbit / turning pattern) with exact node velocities.

    ``center`` (3,) gives the circle centre and starting altitude; angle measured from east.
    """
    ce = np.asarray(center, dtype=float)
    t = t_start + np.arange(0.0, duration + 0.5 * dt, dt)
    w = (-1.0 if clockwise else 1.0) * speed / radius
    a = math.radians(phase0_deg) + w * (t - t_start)
    pos = np.stack([ce[0] + radius * np.cos(a), ce[1] + radius * np.sin(a),
                    ce[2] + climb_rate * (t - t_start)], axis=1)
    vel = np.stack([-radius * w * np.sin(a), radius * w * np.cos(a), np.full_like(t, climb_rate)], axis=1)
    return Track(t=t, pos=pos, vel=vel, label=label)


# --- Rendering ----------------------------------------------------------------------------------


def _audible_span(track: Track, obs: np.ndarray, c: float) -> tuple[float, float]:
    """Reception-time interval during which emissions from inside the track span arrive."""
    r0 = float(np.linalg.norm(track.pos[0] - obs))
    r1 = float(np.linalg.norm(track.pos[-1] - obs))
    return track.t_start + r0 / c, track.t_end + r1 / c


def render_source(src: SynthSource, obs, t_r: np.ndarray, c: float = C_DEFAULT,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Pressure waveform (float64, shape t_r.shape) of one source at the observer.

    Zero where the emission time falls outside the track (with a raised-cosine taper inside it).
    """
    ob = np.asarray(obs, dtype=float)
    t_r = np.asarray(t_r, dtype=np.float64)
    out = np.zeros(t_r.shape)
    lo, hi = _audible_span(src.track, ob, c)
    sel = np.nonzero((t_r >= lo) & (t_r <= hi))[0]
    if sel.size == 0:
        return out
    a, b = sel[0], sel[-1] + 1  # t_r is assumed sorted: one contiguous slice
    em = solve_emission(t_r[a:b], src.track, ob, c)
    t_rel = em.t_e - src.track.t_start
    env = src.r_ref / np.maximum(em.r, 1.0)
    if src.taper_s > 0:
        x0 = np.clip(t_rel / src.taper_s, 0.0, 1.0)
        x1 = np.clip((src.track.t_end - em.t_e) / src.taper_s, 0.0, 1.0)
        env = env * (0.5 - 0.5 * np.cos(np.pi * x0)) * (0.5 - 0.5 * np.cos(np.pi * x1))
    env = np.where(em.valid, env, 0.0)
    if src.phases is not None:
        phases = list(src.phases)
    else:
        g = rng if rng is not None else np.random.default_rng(0)
        phases = list(g.uniform(0, 2 * np.pi, len(src.harmonics)))
    seg = np.zeros(b - a)
    two_pi_f0 = 2.0 * np.pi * src.f0
    for h, amp in enumerate(src.harmonics, start=1):
        if amp != 0.0:
            seg += amp * np.sin(two_pi_f0 * h * t_rel + phases[h - 1])
    out[a:b] = seg * env
    return out


def source_truth(src: SynthSource, obs, t0: float, c: float = C_DEFAULT,
                 r_max: float = math.inf) -> SourceTruth:
    passes = track_passes(src.track, obs, c, r_max=r_max)
    return SourceTruth(label=src.label, f0=src.f0, passes=passes, tau_h=[p.t_h - t0 for p in passes])


# --- Background -------------------------------------------------------------------------------


def pink_noise(n: int, fs: int, rms: float, rng: np.random.Generator, f_lo: float = 5.0) -> np.ndarray:
    """Gaussian noise with 1/f power spectrum above ``f_lo`` (flat below), scaled to ``rms``."""
    if n == 0 or rms == 0:
        return np.zeros(n)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec *= 1.0 / np.sqrt(np.maximum(f, f_lo))
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return x * (rms / max(float(np.sqrt(np.mean(x * x))), 1e-300))


def hum_lines(n: int, fs: int, lines: Sequence[tuple[float, float]], rng: np.random.Generator,
              am_depth: float = 0.0) -> np.ndarray:
    """Stationary tones; ``lines`` = [(freq_Hz, amplitude), ...]. Optional slow (<0.1 Hz) AM."""
    t = np.arange(n) / fs
    x = np.zeros(n)
    for f, a in lines:
        mod: float | np.ndarray = 1.0
        if am_depth > 0:
            mod = 1.0 + am_depth * np.sin(2 * np.pi * rng.uniform(0.005, 0.1) * t + rng.uniform(0, 2 * np.pi))
        x += a * mod * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return x


def transients(n: int, fs: int, rate_per_min: float, amp: float, rng: np.random.Generator,
               dur_s: tuple[float, float] = (0.005, 0.05)) -> np.ndarray:
    """Poisson broadband clicks: exponentially decaying white-noise bursts with random peak <= amp."""
    x = np.zeros(n)
    if rate_per_min <= 0 or amp == 0 or n == 0:
        return x
    k = rng.poisson(rate_per_min * n / fs / 60.0)
    for start in rng.integers(0, n, size=k):
        tau = rng.uniform(*dur_s)
        m = min(n - start, int(5 * tau * fs) + 1)
        tt = np.arange(m) / fs
        x[start:start + m] += amp * rng.uniform(0.3, 1.0) * np.exp(-tt / tau) * rng.standard_normal(m)
    return x


def band_power(x: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    """Mean power of ``x`` restricted to [f_lo, f_hi] Hz (Parseval on the rfft)."""
    x = np.asarray(x, dtype=np.float64)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / fs)
    m = (f >= f_lo) & (f <= f_hi)
    w = np.where((f == 0) | (f == fs / 2), 1.0, 2.0)  # one-sided weights
    return float(np.sum(w[m] * np.abs(spec[m]) ** 2) / x.size**2)


# --- Codec ------------------------------------------------------------------------------------


def mp3_roundtrip(audio: np.ndarray, fs: int = FS_DEFAULT, kbps: int = 64, encode_fs: int = 22050,
                  passes: int = 1) -> np.ndarray:
    """Encode to MP3 (libmp3lame, CBR ``kbps``) at ``encode_fs`` and decode back to ``fs``, ``passes`` times.

    MP3 does not support 4 kHz, so the signal is upsampled (soxr) for encoding. Files (not pipes)
    are used so the LAME gapless header is written and the decoder trims encoder delay; output length
    equals input length. Audio is peak-normalised for the codec and scaled back afterwards.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (run inside `nix develop`)")
    x = np.asarray(audio, dtype=np.float32)
    n = x.size
    peak = float(np.max(np.abs(x))) if n else 0.0
    gain = 0.8 / peak if peak > 0 else 1.0
    y = x * gain
    with tempfile.TemporaryDirectory(prefix="tri_mp3_") as td:
        raw, mp3, dec = Path(td, "in.f32"), Path(td, "x.mp3"), Path(td, "out.f32")
        for _ in range(passes):
            y.astype("<f4").tofile(raw)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "f32le", "-ar", str(fs), "-ac", "1", "-i", str(raw),
                 "-af", f"aresample={encode_fs}:resampler=soxr", "-c:a", "libmp3lame",
                 "-b:a", f"{kbps}k", str(mp3)],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3),
                 "-map", "a:0", "-ac", "1", "-af", f"aresample={fs}:resampler=soxr", "-f", "f32le", str(dec)],
                check=True,
            )
            z = np.fromfile(dec, dtype="<f4")
            y = z[:n] if z.size >= n else np.pad(z, (0, n - z.size))
    return (y / gain).astype(np.float32)


# --- Scene ------------------------------------------------------------------------------------


def synthesize(duration_s: float, sources: Sequence[SynthSource], obs, *, fs: int = FS_DEFAULT,
               t0: float = 0.0, c: float = C_DEFAULT, noise_rms: float = 0.0,
               hum: Sequence[tuple[float, float]] = (), hum_am_depth: float = 0.0,
               click_rate_per_min: float = 0.0, click_amp: float = 0.0, mp3_kbps: int | None = None,
               mp3_passes: int = 1, seed: int = 0, keep_components: bool = False) -> SynthResult:
    """Render a scene: tonal sources (exact retarded Doppler) + pink noise + hum + clicks [+ MP3].

    Sample n is heard at absolute time t0 + n/fs. ``truth`` lists every pass (CPA) of every source
    with its heard audio position tau_h. Set ``keep_components`` to also return the pre-codec
    signal-only and background-only waveforms (for SNR bookkeeping).
    """
    ob = np.asarray(obs, dtype=float)
    n = int(round(duration_s * fs))
    rng = np.random.default_rng(seed)
    t_r = t0 + np.arange(n) / fs
    sig = np.zeros(n)
    truths = []
    for src in sources:
        sig += render_source(src, ob, t_r, c, rng)
        truths.append(source_truth(src, ob, t0, c))
    bg = pink_noise(n, fs, noise_rms, rng)
    if hum:
        bg += hum_lines(n, fs, hum, rng, hum_am_depth)
    bg += transients(n, fs, click_rate_per_min, click_amp, rng)
    mix = (sig + bg).astype(np.float32)
    if mp3_kbps is not None:
        mix = mp3_roundtrip(mix, fs, mp3_kbps, passes=mp3_passes)
    return SynthResult(
        audio=mix, fs=fs, t0=float(t0), c=float(c), obs=ob, truth=truths,
        signal=sig.astype(np.float32) if keep_components else None,
        background=bg.astype(np.float32) if keep_components else None,
    )


def write_wav(path: str | Path, audio: np.ndarray, fs: int = FS_DEFAULT) -> Path:
    """Write mono float32 WAV (soundfile), e.g. to feed the real ingest path in end-to-end tests."""
    import soundfile as sf

    p = Path(path)
    sf.write(p, np.asarray(audio, dtype=np.float32), fs, subtype="FLOAT")
    return p
