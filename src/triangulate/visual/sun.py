"""Sun / shadow late re-rank (plan 1.3 step 6, WP17).

A shadow clue constrains the sun elevation (from height/shadow-length) and, if the camera heading
is known, the sun azimuth. Each top-K candidate (lat, lon, T_s) predicts the sun position via
pvlib; `sun_logliks` turns the mismatch into one ln-likelihood per candidate for
`fusion.candidates.rerank(cands, "sun", logliks)`.

Robustness (never a hard veto):
- each term is a mixture `conf * N(delta; 0, sigma) + (1 - conf) * Uniform`, with conf capped at
  `MAX_CONF`; conf = 0 gives the same value for every candidate (neutral);
- time: a candidate's own `t_s` is used; else the clue's `time_window` is marginalised
  (likelihood averaged over the window); with neither, the candidate gets None (skipped);
- azimuth is used only with a world azimuth, or a camera-relative one plus heading option(s);
  several heading options (e.g. a 180-degree ambiguity) are averaged; no heading = elevation only.

Expected mapping from WP15 clue JSON (type "sun_shadow"), done by the caller:
  evidence/user marks -> `shadow_length_over_height` (or `sun_elevation_deg`),
  `shadow_azimuth_deg` (compass direction the shadow points to) or `shadow_rel_azimuth_deg`
  (clockwise from the camera axis) + `camera_heading_deg`, and `confidence` -> `confidence`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from ..fusion.candidates import FusionCandidate, rerank

__all__ = ["SunClue", "sun_logliks", "sun_rerank", "predicted_sun", "MAX_CONF"]

MAX_CONF = 0.95
_U_ELEV = 1.0 / 90.0  # uniform density over elevation (per degree)
_U_AZ = 1.0 / 360.0


@dataclass
class SunClue:
    """Shadow evidence. Give `shadow_length_over_height` or `sun_elevation_deg` (or neither if only
    azimuth is known)."""

    shadow_length_over_height: float | None = None  # shadow length / object height
    sun_elevation_deg: float | None = None
    elevation_sigma_deg: float = 3.0
    # Direction the shadow points to (compass, 0 = N, clockwise), in the world frame...
    shadow_azimuth_deg: float | None = None
    # ...or relative to the camera axis (clockwise), needing heading option(s):
    shadow_rel_azimuth_deg: float | None = None
    camera_heading_deg: Sequence[float] | float | None = None  # one or several options
    azimuth_sigma_deg: float = 15.0
    confidence: float = 1.0  # 0..1
    time_window: tuple[datetime, datetime] | None = None  # used when a candidate has no t_s
    window_step_s: float = 300.0


def _elevation(clue: SunClue) -> float | None:
    if clue.sun_elevation_deg is not None:
        return float(clue.sun_elevation_deg)
    r = clue.shadow_length_over_height
    if r is not None and r > 0:
        return math.degrees(math.atan2(1.0, r))
    return None


def _headings(clue: SunClue) -> list[float]:
    h = clue.camera_heading_deg
    if h is None:
        return []
    return [float(h)] if isinstance(h, int | float) else [float(v) for v in h]


def predicted_sun(lat: float, lon: float, times: Sequence[datetime]) -> tuple[np.ndarray, np.ndarray]:
    """(elevation_deg, azimuth_deg) from pvlib (apparent elevation) at tz-aware UTC times."""
    import pandas as pd
    from pvlib import solarposition

    idx = pd.DatetimeIndex([t if t.tzinfo else t.replace(tzinfo=UTC) for t in times]).tz_convert("UTC")
    sp = solarposition.get_solarposition(idx, lat, lon)
    return sp["apparent_elevation"].to_numpy(), sp["azimuth"].to_numpy()


def _window_times(clue: SunClue) -> list[datetime]:
    assert clue.time_window is not None
    t0, t1 = clue.time_window
    n = max(1, int(math.ceil((t1 - t0).total_seconds() / clue.window_step_s)))
    n = min(n, 2000)
    return [t0 + (t1 - t0) * (i / n) for i in range(n + 1)] if t1 > t0 else [t0]


def _time_lik(lat: float, lon: float, times: Sequence[datetime], clue: SunClue,
              elev: float | None) -> float:
    """Mean over `times` of the clue likelihood (density scale), for one position."""
    el, az = predicted_sun(lat, lon, times)
    c = min(max(clue.confidence, 0.0), MAX_CONF)
    lik = np.ones(len(times))
    if elev is not None:
        s = clue.elevation_sigma_deg
        g = np.exp(-0.5 * ((el - elev) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        lik *= c * g + (1 - c) * _U_ELEV
    shadow_az = clue.shadow_azimuth_deg
    heads = _headings(clue)
    if shadow_az is not None:
        opts = [shadow_az]
    elif clue.shadow_rel_azimuth_deg is not None and heads:
        opts = [h + clue.shadow_rel_azimuth_deg for h in heads]
    else:
        opts = []
    if opts:
        s = clue.azimuth_sigma_deg
        acc = np.zeros(len(times))
        for o in opts:
            d = (az - (o + 180.0) + 180.0) % 360.0 - 180.0  # sun is opposite the shadow
            acc += np.exp(-0.5 * (d / s) ** 2) / (s * math.sqrt(2 * math.pi))
        lik *= c * acc / len(opts) + (1 - c) * _U_AZ
    return float(lik.mean())


def sun_logliks(cands: Sequence[FusionCandidate], clue: SunClue) -> list[float | None]:
    """One ln-likelihood per candidate (None = skipped: no usable clue or no time)."""
    elev = _elevation(clue)
    has_az = clue.shadow_azimuth_deg is not None or (
        clue.shadow_rel_azimuth_deg is not None and bool(_headings(clue)))
    if elev is None and not has_az:
        return [None] * len(cands)
    window = _window_times(clue) if clue.time_window is not None else None
    out: list[float | None] = []
    for c in cands:
        if c.t_s is not None:
            ts = c.t_s if c.t_s.tzinfo else c.t_s.replace(tzinfo=UTC)
            times = [ts]
        elif window is not None:
            times = window
        else:
            out.append(None)
            continue
        lik = _time_lik(c.lat, c.lon, times, clue, elev)
        out.append(math.log(max(lik, 1e-300)))
    return out


def sun_rerank(cands: Sequence[FusionCandidate], clue: SunClue, weight: float = 1.0,
               ) -> list[FusionCandidate]:
    """Apply `sun_logliks` through `fusion.candidates.rerank` (name "sun")."""
    return rerank(cands, "sun", sun_logliks(cands, clue), weight=weight)


