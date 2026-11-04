"""WP09 on real data: Oyster Bay region (+-0.3 deg), 2026-09-08 19:30-22:30 UTC, the video's event ranges.

Reports candidate counts after each gate and checks that cells near the ground truth survive. The ground
truth is used only for the check, never by the algorithm. Markers: network (adsb.lol members, DEM and
geoid on first run; cached afterwards), slow.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from triangulate.acoustic.cpa import EventGate, cpa_search, eligible_expr
from triangulate.fusion.grid import GridSpec, latlon_to_cells

pytestmark = [pytest.mark.network, pytest.mark.slow]

GT = (40.876722, -73.526444)
BBOX = (GT[0] - 0.3, GT[1] - 0.3, GT[0] + 0.3, GT[1] + 0.3)
T0 = datetime(2026, 9, 8, 19, 30, tzinfo=UTC)
T1 = datetime(2026, 9, 8, 22, 30, tzinfo=UTC)
TAU = (2152.0, 3073.0)  # audio positions of the two events [N 7]
SLACK = 5.0


def test_oyster_bay_gate_counts():
    from triangulate.adsb.adsblol import AdsbLolProvider
    from triangulate.adsb.aircraft_types import load_types
    from triangulate.adsb.tracks import prepare_trackset
    from triangulate.core.geo import bbox_expand_m
    from triangulate.core.types import TrackQuery

    ts = AdsbLolProvider().fetch(TrackQuery(bbox=bbox_expand_m(BBOX, 15_000), t0=T0, t1=T1))
    spec = GridSpec.for_bbox(BBOX)
    origin = (spec.lat0, spec.lon0)
    prep, cls = prepare_trackset(ts, origin, load_types())  # jets removed
    t_s = (T0.timestamp(), T1.timestamp())  # wide start-time window (plan 8.2)
    gates = [
        EventGate.from_ranges("ev1", (130.0, 154.0), (4300.0, 7800.0), (TAU[0] - 3, TAU[0] + 3), t_s, SLACK),
        EventGate.from_ranges("ev2", (88.0, 110.0), (1390.0, 4690.0), (TAU[1] - 3, TAU[1] + 3), t_s, SLACK),
    ]
    res = cpa_search(prep, origin, spec, gates)
    tab = res.table
    # informative extra stage (WP10 does this properly): cells with an offset-consistent pair
    e1 = tab.filter(eligible_expr(0)).select("cell", pl.col("icao").alias("i1"),
                                             (pl.col("t_h") - TAU[0]).alias("o1"))
    e2 = tab.filter(eligible_expr(1)).select("cell", pl.col("icao").alias("i2"),
                                             (pl.col("t_h") - TAU[1]).alias("o2"))
    pairs = e1.join(e2, on="cell").filter((pl.col("o1") - pl.col("o2")).abs() <= SLACK)
    report = dict(aircraft=ts.aircraft.height, kept=int(cls["keep"].sum()), **res.stats,
                  cells_offset_pair=pairs["cell"].n_unique())
    print(json.dumps(report, indent=1, default=float))

    # cells around the ground truth (3x3 at 200 m)
    gx, gy = (int(np.ravel(a)[0]) for a in latlon_to_cells(spec, GT[0], GT[1], 3))
    near = tab.filter(((pl.col("ix") - gx).abs() <= 1) & ((pl.col("iy") - gy).abs() <= 1))
    agg = near.group_by("cell").agg(pl.col("eligible").bitwise_or())
    both = int((agg["eligible"] == 3).sum())
    near_pairs = pairs.join(near.select("cell").unique(), on="cell")
    print("GT 3x3: cells eligible for both events:", both, "| offset pairs:",
          near_pairs.select("i1", "i2").unique().rows())
    assert res.stats["cells_eligible_all"] < res.stats["cells_candidates"]
    assert both >= 5  # the ground-truth neighbourhood survives the speed + distance gates
    assert near_pairs.height >= 1  # ...and has a pair heard ~921 s apart (+-5 s)
