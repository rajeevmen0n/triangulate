"""WP10 on real data: Oyster Bay, 2026-09-08 19:30-22:30 UTC, +-0.3 deg, the video's event ranges.

Reports candidate counts after the gap match and after the silence filter, and whether the ground-truth
cells survive. The ground truth is used only for the check, never by the algorithm.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from triangulate.acoustic.cpa import EventGate, cpa_search
from triangulate.acoustic.matching import match_offsets
from triangulate.acoustic.silence import silence_filter
from triangulate.core.types import Event, Range
from triangulate.fusion.grid import GridSpec, latlon_to_cells

pytestmark = [pytest.mark.network, pytest.mark.slow]

GT = (40.876722, -73.526444)
BBOX = (GT[0] - 0.3, GT[1] - 0.3, GT[0] + 0.3, GT[1] + 0.3)
T0 = datetime(2026, 9, 8, 19, 30, tzinfo=UTC)
T1 = datetime(2026, 9, 8, 22, 30, tzinfo=UTC)
TAU = (2152.0, 3073.0)
DURATION = 3600.3
EXPECTED_TS = datetime(2026, 9, 8, 20, 17, 35, tzinfo=UTC).timestamp()


def _ev(i, tau, v, d):
    return Event(id=f"ev{i}", t_h=tau, t_h_range=Range(lo=tau - 3, hi=tau + 3), f0=150.0,
                 f0_range=Range(lo=100, hi=200), v=sum(v) / 2, v_range=Range(lo=v[0], hi=v[1]),
                 d=sum(d) / 2, d_range=Range(lo=d[0], hi=d[1]))


def test_oyster_bay_match_and_silence():
    from triangulate.adsb.adsblol import AdsbLolProvider
    from triangulate.adsb.aircraft_types import load_types
    from triangulate.adsb.tracks import prepare_trackset
    from triangulate.core.geo import bbox_expand_m
    from triangulate.core.types import TrackQuery

    ts = AdsbLolProvider().fetch(TrackQuery(bbox=bbox_expand_m(BBOX, 15_000), t0=T0, t1=T1))
    spec = GridSpec.for_bbox(BBOX)
    origin = (spec.lat0, spec.lon0)
    prep, _ = prepare_trackset(ts, origin, load_types())
    events = [_ev(1, TAU[0], (130.0, 154.0), (4300.0, 7800.0)),
              _ev(2, TAU[1], (88.0, 110.0), (1390.0, 4690.0))]
    t_s = (T0.timestamp(), T1.timestamp())
    gates = [EventGate.from_event(e, t_s) for e in events]
    res = cpa_search(prep, origin, spec, gates)
    hyps = match_offsets(res.table, TAU, t_s=t_s)
    out = dict(cells_eligible_all=res.stats["cells_eligible_all"], hyps_after_gap=hyps.height,
               cells_after_gap=hyps["cell"].n_unique(),
               tuples_after_gap=hyps.select("icao0", "icao1").unique().height)
    gx, gy = (int(np.ravel(a)[0]) for a in latlon_to_cells(spec, GT[0], GT[1], 3))
    near = (((pl.col("ix") - gx).abs() <= 1) & ((pl.col("iy") - gy).abs() <= 1))
    for name, kw in (("between", {}), ("full", dict(scope="full", duration=DURATION))):
        sr = silence_filter(hyps, prep, res.observers, TAU, **kw)
        out[f"silence_{name}"] = dict(hyps=sr.hyps.height, cells=sr.hyps["cell"].n_unique(),
                                      tuples=sr.hyps.select("icao0", "icao1").unique().height)
        gt = sr.hyps.filter(near & (pl.col("icao0").str.starts_with("a1dfeb"))
                            & pl.col("icao1").str.starts_with("ace391"))
        out[f"silence_{name}"]["gt_cells"] = gt["cell"].n_unique()
        if gt.height:
            out[f"silence_{name}"]["gt_Ts_offset_s"] = float(gt["o"].mean() - EXPECTED_TS)
    g0 = hyps.filter(near & pl.col("icao0").str.starts_with("a1dfeb")
                     & pl.col("icao1").str.starts_with("ace391"))
    out["gt_cells_after_gap"] = g0["cell"].n_unique()
    print(json.dumps(out, indent=1, default=float))
    assert out["hyps_after_gap"] > 0
    assert out["silence_between"]["hyps"] <= out["hyps_after_gap"]
    assert out["gt_cells_after_gap"] >= 1
    assert out["silence_between"]["gt_cells"] >= 1
