"""WP10: offset matching + silence filter on hand-built CPA tables and a small synthetic scene."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from triangulate.acoustic.cpa import CPA_SCHEMA, EventGate, compute_cpa
from triangulate.acoustic.grid import SEARCH_LEVEL, CellSet, observer_cells
from triangulate.acoustic.matching import match_offsets, taus_from_events
from triangulate.acoustic.silence import silence_filter, silent_spans
from triangulate.acoustic.synth import straight_pass_track
from triangulate.core.types import Event, Range
from triangulate.fusion.grid import GridSpec

# --- matching on hand-built tables ---------------------------------------------------------------


def table(rows):
    """rows: (cell, icao, t_h, eligible_bits[, pass])"""
    recs = []
    for i, r in enumerate(rows):
        cell, icao, th, el = r[:4]
        recs.append(dict(cell=cell, level=3, ix=cell, iy=0, icao=icao, seg=0,
                         **{"pass": r[4] if len(r) > 4 else i},
                         t_cpa=th - 5, t_h=th, d=2000.0 + i, v=70.0, gs=70.0, agl=500.0, eligible=el))
    return pl.DataFrame(recs, schema=CPA_SCHEMA)


T0 = 1.789e9
TAUS = [1000.0, 1920.0]


def test_pairs_within_slack_exact():
    # event-0 pass at o = T0; event-1 passes at offsets T0 + {0, 4.9, 5.0, 5.1, -5.0, -6}
    deltas = [0.0, 4.9, 5.0, 5.1, -5.0, -6.0]
    rows = [(1, "A", T0 + TAUS[0], 1)] + [(1, f"B{i}", T0 + TAUS[1] + dlt, 2) for i, dlt in enumerate(deltas)]
    h = match_offsets(table(rows), TAUS, slack=5.0)
    assert sorted(h["icao1"].to_list()) == ["B0", "B1", "B2", "B4"]  # |delta| <= 5, exactly
    assert h.filter(pl.col("icao1") == "B1")["o"][0] == pytest.approx(T0 + 2.45)
    assert h["o_spread"].max() <= 5.0 + 1e-9
    # a different cell never pairs
    rows2 = [(1, "A", T0 + TAUS[0], 1), (2, "B", T0 + TAUS[1], 2)]
    assert match_offsets(table(rows2), TAUS).height == 0
    # eligibility bits are respected
    rows3 = [(1, "A", T0 + TAUS[0], 2), (1, "B", T0 + TAUS[1], 2)]
    assert match_offsets(table(rows3), TAUS).height == 0


def test_start_window_and_single_event():
    rows = [(1, "A", T0 + TAUS[0], 1), (1, "B", T0 + TAUS[1], 2)]
    assert match_offsets(table(rows), TAUS, t_s=(T0 - 10, T0 + 10)).height == 1
    assert match_offsets(table(rows), TAUS, t_s=(T0 + 11, T0 + 50)).height == 0
    with pytest.raises(ValueError):
        match_offsets(table(rows), [1000.0])
    tb = table([(1, "A", T0 + 1000.0, 1), (2, "Z", T0 + 4000.0, 1)])
    one = match_offsets(tb, [1000.0], t_s=(T0 - 90, T0 + 90))
    assert one["icao0"].to_list() == ["A"] and one["o"][0] == pytest.approx(T0)


def test_chaining_n3_and_same_aircraft():
    taus = [500.0, 1500.0, 2600.0]
    rows = [(1, "A", T0 + 500, 1), (1, "B", T0 + 1500 + 3, 2), (1, "C", T0 + 2600 - 2, 4),
            (1, "D", T0 + 2600 + 9, 4),  # fits the 0-1 offset but not event 0
            (2, "A", T0 + 500, 1), (2, "B", T0 + 1500 + 4, 2), (2, "C", T0 + 2600 - 4.5, 4)]  # 8.5 s apart
    h = match_offsets(table(rows), taus)
    assert h["cell"].to_list() == [1] and h["icao2"].to_list() == ["C"]  # pairwise, not chained to the mean
    # the same aircraft, two different passes, explains events 0 and 1
    rows = [(1, "A", T0 + 1000, 1, 10), (1, "A", T0 + 1920, 2, 11)]
    h = match_offsets(table(rows), TAUS)
    assert h.height == 1 and h["icao0"][0] == h["icao1"][0] == "A"
    # but one pass cannot serve two events
    rows = [(1, "A", T0 + 1000, 3, 10)]
    assert match_offsets(table(rows), [1000.0, 1002.0]).height == 0


# --- silence filter on a synthetic scene ---------------------------------------------------------

SPEC = GridSpec.for_bbox((40.80, -73.62, 41.00, -73.42))
ORIGIN = (SPEC.lat0, SPEC.lon0)
NX, NY = SPEC.shape(SEARCH_LEVEL)
OBS = observer_cells(CellSet.window(SPEC, SEARCH_LEVEL, NX // 2, NY // 2, 2, 2), ORIGIN,
                     lambda lat, lon: np.full_like(np.asarray(lat, float), -30.0))
X = OBS.xyz[0]


def frame(tr, icao):
    return pl.DataFrame({"icao": icao, "seg": 0, "t": tr.t, "e": tr.pos[:, 0], "n": tr.pos[:, 1],
                         "u": tr.pos[:, 2], "gs_ms": np.hypot(tr.vel[:, 0], tr.vel[:, 1])})


def ev(i, tau, lo=None, hi=None):
    r = Range
    return Event(id=f"e{i}", t_h=tau, t_h_range=r(lo=lo or tau - 3, hi=hi or tau + 3), f0=150.0,
                 f0_range=r(lo=140, hi=160), v=60.0, v_range=r(lo=40, hi=80), d=3000.0,
                 d_range=r(lo=1000, hi=4000))


def scene(decoy_tau: float, decoy_d: float, extra=()):
    trs = [frame(straight_pass_track(X, T0 + TAUS[0], 60.0, 3000.0, heading_deg=90, label="A"), "AAA"),
           frame(straight_pass_track(X, T0 + TAUS[1], 60.0, 2000.0, heading_deg=0, label="B"), "BBB"),
           frame(straight_pass_track(X, T0 + decoy_tau, 55.0, decoy_d, heading_deg=45, label="C"), "CCC"),
           *extra]
    tracks = pl.concat(trs)
    events = [ev(0, TAUS[0]), ev(1, TAUS[1])]
    gates = [EventGate.from_event(e, v_margin=0) for e in events]
    cpa = compute_cpa(tracks, OBS, gates)
    h = match_offsets(cpa, TAUS)
    return tracks, h


def run(decoy_tau=1500.0, decoy_d=2500.0, **kw):
    tracks, h = scene(decoy_tau, decoy_d)
    assert h.height >= 1 and set(h["icao0"]) == {"AAA"} and set(h["icao1"]) == {"BBB"}
    return h, silence_filter(h, tracks, OBS, TAUS, **kw)


def test_silent_spans():
    assert silent_spans([1000, 1920]) == [(1060, 1860)]
    assert silent_spans([1000, 1100]) == []  # gap shorter than 2 guards
    assert silent_spans([1000, 1500, 2000]) == [(1060, 1440), (1560, 1940)]
    assert silent_spans([1000], scope="full", duration=3000) == [(0, 940), (1060, 3000)]
    with pytest.raises(ValueError):
        silent_spans([1000], scope="full")


def test_decoy_rejected_and_kappa():
    h, r = run()  # third flight at d=2500 < max(d_k)=3000 inside the silent span
    assert r.hyps.height == 0 and r.rejected["icao"].unique().to_list() == ["CCC"]
    h, r = run(kappa=0.5)  # threshold 1500 m < 2500 m: kept
    assert r.hyps.height == h.height
    h, r = run(decoy_d=3500.0)  # too far to have been heard
    assert r.hyps.height == h.height


def test_guard_exemption_and_soft():
    h, r = run(decoy_tau=1000 + 30)  # inside the guard around event 0: not counted
    assert r.hyps.height == h.height
    h, r = run(decoy_tau=100.0)  # outside the span (before event 0)
    assert r.hyps.height == h.height
    h, r = run(exempt_events=[ev(9, 1500.0)])  # detected, unselected event at the decoy's time
    assert r.hyps.height == h.height
    h, r = run(exempt_events=[ev(9, 1530.0)])  # 30 s away: not exempt
    assert r.hyps.height == 0
    h, r = run(soft_beta=2.0)
    assert r.hyps.height == h.height and (r.hyps["penalty"] > 0).all()
    assert r.hyps["penalty"][0] == pytest.approx(2.0 * (1 - 2500 / 3000), abs=0.05)


def test_own_flights_and_single_event_scope():
    tracks, h = scene(1500.0, 2500.0)
    # decoy flight is part of the tuple's aircraft set -> exempt; with own_exempt=False tuple's own
    # other pass would count, but here the tuple flights have a single pass, so nothing changes
    assert silence_filter(h, tracks, OBS, TAUS, own_exempt=False).hyps.height == 0
    # single-event recording: the 'between' span is empty (no-op); 'full' covers the recording
    h1 = match_offsets(compute_cpa(tracks, OBS, [EventGate.from_event(ev(0, TAUS[0]), v_margin=0)]),
                       [TAUS[0]], t_s=(T0 - 20, T0 + 20))
    assert h1.height >= 1
    r = silence_filter(h1, tracks, OBS, [TAUS[0]])
    assert r.stats["noop"] and r.hyps.height == h1.height
    r = silence_filter(h1, tracks, OBS, [TAUS[0]], scope="full", duration=3600.0)
    assert not r.stats["noop"] and r.hyps.height == 0  # BBB (d=2000) and CCC are heard but undetected
    r = silence_filter(h1, tracks, OBS, [TAUS[0]], scope="full", duration=3600.0,
                       exempt_events=[ev(1, TAUS[1]), ev(2, 1500.0)])
    assert r.hyps.height == h1.height


def test_taus_from_events():
    assert taus_from_events([ev(0, 5.0), ev(1, 9.0)]) == [5.0, 9.0]
