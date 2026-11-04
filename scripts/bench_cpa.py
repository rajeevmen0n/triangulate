"""WP09 benchmark: CPA stage in region mode (plan 3.3.8: 100 x 100 km, 3 h, <= 500 prop flights, <= 60 s).

`python scripts/bench_cpa.py [--flights 500] [--km 100] [--hours 3] [--threads 4]`

Synthetic scene (flat DEM, no network): GA-like flights with random entry points, headings, speeds
(40-160 m/s), altitudes (300-3000 m) and gentle turns, 10 % orbiting patterns, sampled at 1 s in the
grid's ENU frame. Gates are the video's two events [N 2 10:33] with a 3 h start-time window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from triangulate.acoustic.cpa import EventGate, cpa_search  # noqa: E402
from triangulate.core.geo import EARTH_R  # noqa: E402
from triangulate.fusion.grid import GridSpec  # noqa: E402

T0 = 1_788_890_400.0  # 2026-09-08 18:00 UTC (any epoch works)
CENTER = (40.876, -73.526)


def region_bbox(km: float, center=CENTER) -> tuple[float, float, float, float]:
    dlat = math.degrees(km / 2 * 1000 / EARTH_R)
    dlon = dlat / math.cos(math.radians(center[0]))
    return (center[0] - dlat, center[1] - dlon, center[0] + dlat, center[1] + dlon)


def make_scene(n_flights: int = 500, km: float = 100.0, hours: float = 3.0, seed: int = 0) -> pl.DataFrame:
    """Prepared-schema frame (icao, seg, t, e, n, u, gs_ms, ...) in the ENU frame of the region centre."""
    rng = np.random.default_rng(seed)
    half = km * 500.0
    rows = []
    for k in range(n_flights):
        dur = float(rng.uniform(900, 3600))
        t_start = T0 + float(rng.uniform(-600, hours * 3600 - 300))
        t = t_start + np.arange(0.0, dur, 1.0)
        spd = float(rng.choice([rng.uniform(40, 90), rng.uniform(85, 115), rng.uniform(125, 160)],
                               p=[0.6, 0.25, 0.15]))
        alt = float(rng.uniform(300, 3000))
        if rng.random() < 0.1:  # orbit / pattern work
            ce = rng.uniform(-half, half, 2)
            rad = float(rng.uniform(1500, 4000))
            a = rng.uniform(0, 2 * np.pi) + spd / rad * (t - t_start)
            e, n = ce[0] + rad * np.cos(a), ce[1] + rad * np.sin(a)
            u = np.full_like(t, alt)
        else:
            p = rng.uniform(-half, half, 2)
            hd = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.004, t.size))
            hd += 0.02 * np.sin((t - t_start) / rng.uniform(60, 300))
            e = p[0] + np.cumsum(spd * np.sin(hd))
            n = p[1] + np.cumsum(spd * np.cos(hd))
            u = alt + np.cumsum(rng.normal(0, 0.3, t.size))
        rows.append(pl.DataFrame({"icao": f"{k:06x}", "seg": 0, "t": t, "e": e, "n": n, "u": u,
                                  "gs_ms": np.full_like(t, spd)}))
    return pl.concat(rows)


def video_gates(t_s=(T0, T0 + 3 * 3600.0)) -> list[EventGate]:
    return [
        EventGate.from_ranges("ev1", (130.0, 154.0), (4300.0, 7800.0), (2149.0, 2155.0), t_s),
        EventGate.from_ranges("ev2", (88.0, 110.0), (1390.0, 4690.0), (3070.0, 3076.0), t_s),
    ]


def run(n_flights=500, km=100.0, hours=3.0, threads=None, seed=0) -> dict:
    bbox = region_bbox(km)
    spec = GridSpec.for_bbox(bbox)
    tg = time.perf_counter()
    tracks = make_scene(n_flights, km, hours, seed)
    gen = time.perf_counter() - tg
    origin = (spec.lat0, spec.lon0)
    flat = lambda la, lo: np.zeros_like(la)  # noqa: E731
    gates = video_gates((T0, T0 + hours * 3600.0))
    cpa_search(tracks.filter(pl.col("icao") == "000000"), origin, spec, gates, ground_hae=flat)  # warm numba
    t = time.perf_counter()
    res = cpa_search(tracks, origin, spec, gates, ground_hae=flat, n_threads=threads)
    wall = time.perf_counter() - t
    return dict(flights=n_flights, km=km, hours=hours, samples=tracks.height, scene_gen_s=gen,
                wall_s=wall, **{k: v for k, v in res.stats.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=500)
    ap.add_argument("--km", type=float, default=100.0)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    out = run(a.flights, a.km, a.hours, a.threads)
    print(json.dumps(out, indent=1, default=float))
    ok = out["wall_s"] <= 60.0
    print(f"CPA stage {out['wall_s']:.1f} s (target <= 60 s): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
