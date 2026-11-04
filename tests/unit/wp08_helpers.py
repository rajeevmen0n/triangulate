"""Synthetic scenes for the WP08 detection/fit tests (built on WP06 synth.py)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triangulate.acoustic.synth import (
    FS_DEFAULT,
    SynthSource,
    band_power,
    hum_lines,
    mp3_roundtrip,
    pink_noise,
    render_source,
    source_truth,
    straight_pass_track,
    transients,
)

OBS = np.zeros(3)


@dataclass
class Truth:
    label: str
    tau_h: float
    f0: float
    v: float
    d: float
    vehicle: bool = False


@dataclass
class Scene:
    audio: np.ndarray
    truth: list[Truth]
    snr_db: dict[str, float]


def make_scene(duration: float, passes: list[dict], *, seed: int = 0, noise_rms: float = 0.05,
               hum_hz=(40.0, 60.0, 125.0, 180.0, 250.0), hum_amp: float = 0.03,
               click_rate: float = 4.0, click_amp: float = 0.5, mp3_kbps: int | None = None) -> Scene:
    """Mix straight passes with a prescribed in-band SNR.

    Each pass dict: t_h, v, d, f0, snr_db, [harmonics, height, heading, vehicle, label].
    SNR = 10 log10(P_sig / P_bg) in 20-1000 Hz over t_h +- 60 s, with P_bg = pink noise + clicks
    (hum lines excluded; they are a separate nuisance).
    """
    fs = FS_DEFAULT
    n = int(round(duration * fs))
    rng = np.random.default_rng(seed)
    t_r = np.arange(n) / fs
    bg = pink_noise(n, fs, noise_rms, rng) + transients(n, fs, click_rate, click_amp, rng)
    sig = np.zeros(n)
    truth, snrs = [], {}
    for i, p in enumerate(passes):
        label = p.get("label", f"s{i}")
        tr = straight_pass_track(OBS, p["t_h"], p["v"], p["d"], height=p.get("height"),
                                 heading_deg=p.get("heading", 90.0), half_span_s=p.get("half_span", 400.0))
        src = SynthSource(tr, f0=p["f0"], harmonics=tuple(p.get("harmonics", (1.0, 0.6, 0.4))),
                          phases=tuple(rng.uniform(0, 6.28, 4)), label=label)
        x = render_source(src, OBS, t_r)
        a, b = int(max(p["t_h"] - 60, 0) * fs), int(min(p["t_h"] + 60, duration) * fs)
        ps = band_power(x[a:b], fs, 20, 1000)
        pb = band_power(bg[a:b], fs, 20, 1000)
        x *= np.sqrt(pb / ps * 10 ** (p["snr_db"] / 10))
        sig += x
        st = source_truth(src, OBS, 0.0)
        ps_ = st.passes[0]
        truth.append(Truth(label, ps_.t_h, p["f0"], ps_.v, ps_.d, p.get("vehicle", False)))
        snrs[label] = p["snr_db"]
    if hum_hz:
        sig += hum_lines(n, fs, [(f, hum_amp) for f in hum_hz], rng, am_depth=0.2)
    mix = (sig + bg).astype(np.float32)
    if mp3_kbps:
        mix = mp3_roundtrip(mix, fs, mp3_kbps)
    return Scene(mix, truth, snrs)


def random_pass(rng: np.random.Generator, t_h: float) -> dict:
    d = float(np.exp(rng.uniform(np.log(600), np.log(6000))))
    return dict(t_h=t_h, v=float(rng.uniform(45, 150)), d=d,
                f0=float(rng.uniform(70, 230)), snr_db=float(rng.uniform(-6, 6)),
                heading=float(rng.uniform(0, 360)),
                harmonics=(1.0, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.0, 0.5))))


def match(events, tr: Truth, tol_t: float = 10.0):
    """Highest-SNR event within tol_t of the truth's heard CPA time (and at the right frequency)."""
    c = [e for e in events if abs(e.t_h - tr.tau_h) <= tol_t and abs(np.log(e.f0 / tr.f0)) < 0.1]
    return max(c, key=lambda e: e.snr) if c else None


def errors(e, tr: Truth) -> dict:
    return dict(t=abs(e.t_h - tr.tau_h), v=abs(e.v / tr.v - 1), d=abs(e.d / tr.d - 1),
                f0=abs(e.f0 / tr.f0 - 1),
                inside=bool(e.v_range.contains(tr.v) and e.d_range.contains(tr.d)
                            and e.t_h_range.contains(tr.tau_h) and e.f0_range.contains(tr.f0)))
