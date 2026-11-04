"""WP11 on real data: audio.mp3 (decoded PCM only, never metadata) + adsb.lol 2026-09-08, Oyster Bay
region (+-0.3 deg), T_s window 19:30-22:30 UTC, events from WP08 `find_events` (unaided).

The full acoustic pipeline before WP22/23: gates -> CPA -> offset match -> silence -> locked scoring ->
calibration -> refine 200 -> 100 -> 50 -> 25 m -> fusion layer. The ground truth is used only for the
report/assertions, never by the algorithm. Markers: network (adsb.lol members, DEM and geoid on first
run; cached afterwards), slow.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from triangulate.acoustic.search import SearchConfig, acoustic_search, band_maps
from triangulate.core.geo import haversine_m

pytestmark = [pytest.mark.network, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
GT = (40.876722, -73.526444)
BBOX = (GT[0] - 0.3, GT[1] - 0.3, GT[0] + 0.3, GT[1] + 0.3)
T0 = datetime(2026, 9, 8, 19, 30, tzinfo=UTC)
T1 = datetime(2026, 9, 8, 22, 30, tzinfo=UTC)
EXPECTED_TS = datetime(2026, 9, 8, 20, 17, 35, tzinfo=UTC).timestamp()  # WP07/WP10 anchor (report only)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%H:%M:%S.%f")[:-4] + "Z"


def test_oyster_bay_acoustic_search():
    from triangulate.acoustic.detect import find_events
    from triangulate.acoustic.spectrogram import decode_and_whiten
    from triangulate.adsb.adsblol import AdsbLolProvider
    from triangulate.adsb.aircraft_types import load_types
    from triangulate.adsb.tracks import prepare_trackset
    from triangulate.core.geo import bbox_expand_m
    from triangulate.core.types import TrackQuery
    from triangulate.fusion.grid import GridSpec

    path = ROOT / "audio.mp3"
    if not path.exists():
        pytest.skip("audio.mp3 not present")
    t = time.perf_counter()
    spec = decode_and_whiten(path)
    events = find_events(spec).events
    t_det = time.perf_counter() - t
    ts = AdsbLolProvider().fetch(TrackQuery(bbox=bbox_expand_m(BBOX, 15_000), t0=T0, t1=T1))
    grid = GridSpec.for_bbox(BBOX)
    origin = (grid.lat0, grid.lon0)
    prep, _ = prepare_trackset(ts, origin, load_types())
    t_s = (T0.timestamp(), T1.timestamp())
    t = time.perf_counter()
    res = acoustic_search(spec, events, prep, grid, origin, t_s)
    t_search = time.perf_counter() - t
    assert res.ok
    rep: dict = {
        "events": [(e.id, round(e.t_h, 1), round(e.v), round(e.d)) for e in res.events],
        "detect_s": round(t_det, 1),
        "search_s": round(t_search, 1),
        "stats": {k: v for k, v in res.stats.items() if k != "timing_s"},
        "timing_s": {k: round(v, 2) for k, v in res.stats["timing_s"].items()},
    }
    rep["tuples"] = [
        {
            "rank": r["rank"],
            "flights": r["flights"],
            "z_sum": round(r["z_sum"], 2),
            "z": [round(r["z_0"], 2), round(r["z_1"], 2)],
            "S": round(r["S"], 3),
            "level": r["level"],
            "lat": round(r["lat"], 6),
            "lon": round(r["lon"], 6),
            "t_s": _iso(r["t_s"]),
            "dt": round(r["dt"], 2),
            "dist_m": round(haversine_m(r["lat"], r["lon"], *GT)),
            "centroid_dist_m": round(haversine_m(r["lat_c"], r["lon_c"], *GT)),
        }
        for r in res.tuples.head(10).iter_rows(named=True)
    ]
    rep["top10_200m"] = [
        {
            "flights": r["flights"],
            "z_sum": round(r["z_sum"], 2),
            "t_s": _iso(r["t_s"]),
            "dist_m": round(haversine_m(r["lat"], r["lon"], *GT)),
        }
        for r in res.top(10, level=3).iter_rows(named=True)
    ]
    rep["margin_z"] = res.margin_z
    rep["margin_S"] = res.margin_S
    rep["calibration"] = [
        (c.id, round(c.mu, 4), round(c.sigma, 4), round(c.q99, 2)) for c in res.calibration.events
    ]
    bm = band_maps(res, radius_m=3000.0, level=4)
    rep["band_cells"] = bm.height
    full = acoustic_search(
        spec, events, prep, grid, origin, t_s, config=SearchConfig(silence_scope="full", exempt="aircraft")
    )
    rep["silence_full_aircraft"] = {
        "hyps": full.stats["hyps_after_silence"],
        "tuples": full.stats["tuples_after_silence"],
        "best": full.best()["flights"] if full.ok else None,
    }
    print(json.dumps(rep, indent=1, default=str))

    best = res.best()
    assert {f[:6] for f in best["flights"]} == {"a1dfeb", "ace391"}
    assert haversine_m(best["lat"], best["lon"], *GT) <= 500.0
    assert res.margin_z is not None and res.margin_z > 0
    assert abs(best["t_s"] - EXPECTED_TS) <= 120.0
    assert t_search <= 180.0
