"""WP09: candidate grid, CPA / retarded time and per-event gates (plan 3.3.2-3.3.3, 5.2 WP09)."""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from triangulate.acoustic.cpa import (
    CPA_SCHEMA,
    EventGate,
    candidate_cells,
    compute_cpa,
    cpa_search,
    eligible_expr,
    read_cpa,
    track_segments,
    write_cpa,
)
from triangulate.acoustic.grid import (
    SEARCH_LEVEL,
    AffineMap,
    CellSet,
    densify,
    fit_lattice_map,
    observer_cells,
    region_window,
)
from triangulate.acoustic.physics import C_DEFAULT, Track, track_passes
from triangulate.acoustic.synth import circle_track, straight_pass_track
from triangulate.core.types import Event, Range
from triangulate.fusion.grid import GridSpec, pack_cells, unpack_cells

C = C_DEFAULT
BBOX = (40.80, -73.62, 41.00, -73.42)  # ~22 x 17 km around Oyster Bay
SPEC = GridSpec.for_bbox(BBOX)
ORIGIN = (SPEC.lat0, SPEC.lon0)
NX, NY = SPEC.shape(SEARCH_LEVEL)


def flat(h: float = -30.0):
    return lambda lat, lon: np.full_like(np.asarray(lat, float), h)


@pytest.fixture(scope="module")
def all_obs():
    return observer_cells(CellSet.window(SPEC, SEARCH_LEVEL, 0, 0, NX, NY), ORIGIN, flat())


def cell_index(obs, ix, iy) -> int:
    return int(np.flatnonzero((obs.cells.ix == ix) & (obs.cells.iy == iy))[0])


def rows_at(df: pl.DataFrame, ix: int, iy: int) -> pl.DataFrame:
    return df.filter((pl.col("ix") == ix) & (pl.col("iy") == iy)).sort("icao", "t_cpa")


def racetrack(center, length: float, radius: float, speed: float, t0: float, laps: float, alt: float,
              label: str = "R") -> Track:
    """Two straight legs (east/west) joined by semicircles, sampled at 1 s; finite-difference velocities."""
    per = 2 * length + 2 * math.pi * radius
    t = t0 + np.arange(0.0, laps * per / speed, 1.0)
    s = (speed * (t - t0)) % per
    e = np.empty_like(s)
    n = np.empty_like(s)
    a = s < length  # southern leg eastbound
    e[a], n[a] = -length / 2 + s[a], -radius
    b = (s >= length) & (s < length + math.pi * radius)
    th = (s[b] - length) / radius - math.pi / 2
    e[b], n[b] = length / 2 + radius * np.cos(th), radius * np.sin(th)
    c = (s >= length + math.pi * radius) & (s < 2 * length + math.pi * radius)
    e[c], n[c] = length / 2 - (s[c] - length - math.pi * radius), radius
    d = s >= 2 * length + math.pi * radius
    th = (s[d] - 2 * length - math.pi * radius) / radius + math.pi / 2
    e[d], n[d] = -length / 2 + radius * np.cos(th), radius * np.sin(th)
    pos = np.column_stack([e + center[0], n + center[1], np.full_like(e, alt)])
    return Track.from_positions(t, pos, label=label)


def to_frame(tr: Track, icao: str) -> pl.DataFrame:
    """A physics.Track as a prepared-schema frame (positions only, like adsb.tracks output)."""
    return pl.DataFrame({"icao": icao, "seg": 0, "t": tr.t, "e": tr.pos[:, 0], "n": tr.pos[:, 1],
                         "u": tr.pos[:, 2], "gs_ms": np.hypot(tr.vel[:, 0], tr.vel[:, 1])})


# --- geometry helpers -----------------------------------------------------------------------------


def test_affine_map_and_observers(all_obs):
    amap = fit_lattice_map(SPEC, SEARCH_LEVEL, ORIGIN, region_window(SPEC, SEARCH_LEVEL, 10))
    assert amap.eps_m < 5.0  # LAEA vs ENU over a regional window is near-affine
    assert abs(amap.stretch * 200 - 1) < 0.01
    u, v = amap.to_cells(all_obs.xyz[:, 0], all_obs.xyz[:, 1])
    assert np.abs(u - (all_obs.cells.ix + 0.5)).max() * 200 < amap.eps_m
    assert np.abs(v - (all_obs.cells.iy + 0.5)).max() * 200 < amap.eps_m
    e, n = amap.to_enu(u, v)
    assert np.allclose(e, all_obs.xyz[:, 0], atol=1e-6) and np.allclose(n, all_obs.xyz[:, 1], atol=1e-6)
    # observer height: HAE ground + h_obs, the ENU "up" is the local vertical (tilts away from the origin)
    assert np.allclose(np.linalg.norm(all_obs.up, axis=1), 1.0)
    far = int(np.argmax(np.hypot(all_obs.xyz[:, 0], all_obs.xyz[:, 1])))
    r = math.hypot(*all_obs.xyz[far, :2])
    assert all_obs.xyz[far, 2] < -30 + 5 - r * r / (2 * 6.4e6) + 1  # earth curvature drop
    assert abs(math.hypot(*all_obs.up[far, :2]) - r / 6.371e6) < 2e-4
    with pytest.raises(ValueError):
        AffineMap.fit(np.zeros((2, 2)), np.zeros((2, 2)))


def test_densify_and_cellset():
    fx, fy = densify(3, [10], [20], 5, halo=0)
    assert len(fx) == 16 and fx.min() == 40 and fx.max() == 43 and fy.min() == 80 and fy.max() == 83
    fx, fy = densify(3, [10, 11], [20, 20], 4, halo=1)  # 4x3 coarse cells incl. halo -> 48 children
    assert len(np.unique(pack_cells(4, fx, fy))) == len(fx) == 48
    cs = CellSet.window(SPEC, 3, 2, 3, 4, 5)
    lv, ix, iy = unpack_cells(cs.keys)
    assert len(cs) == 20 and (lv == 3).all() and ix.min() == 2 and iy.max() == 7
    with pytest.raises(ValueError):
        densify(4, [0], [0], 3)


# --- CPA accuracy vs analytic geometry -----------------------------------------------------------


@pytest.mark.parametrize("as_frame", [False, True])
def test_straight_pass_analytic(all_obs, as_frame):
    ix, iy = NX // 2, NY // 2
    j = cell_index(all_obs, ix, iy)
    X = all_obs.xyz[j]
    trs = [straight_pass_track(X, t_h=1.7889e9 + 500.0 + 37 * k, v=v, d=d, height=h, heading_deg=hd,
                               side=sd, label=f"s{k}")
           for k, (v, d, h, hd, sd) in enumerate([(60.0, 3000.0, 1000.0, 30.0, 1),
                                                  (140.0, 6150.0, 2200.0, 250.0, -1),
                                                  (100.0, 1400.0, 1400.0, 90.0, 1)])]
    src = pl.concat([to_frame(t, t.label) for t in trs]) if as_frame else trs
    df = compute_cpa(src, all_obs, [EventGate.window("all", 9000.0)])
    got = rows_at(df, ix, iy)
    assert got.height == 3
    vd = [(60, 3000), (140, 6150), (100, 1400)]
    for tr, row, (v, d) in zip(trs, got.iter_rows(named=True), vd, strict=True):
        assert abs(row["d"] - d) < 1.0
        assert abs(row["t_cpa"] - (tr.t[tr.t.size // 2])) < 0.1  # straight_pass_track centres t_c
        assert abs(row["t_h"] - (row["t_cpa"] + row["d"] / C)) < 1e-3
        assert abs(row["v"] - v) < 0.05 and abs(row["gs"] - v) < 0.05
    # heard time = the track's t_h by construction
    assert np.allclose(got["t_h"].to_numpy(), [1.7889e9 + 500.0 + 37 * k for k in range(3)], atol=0.1)


def test_circle_analytic(all_obs):
    ix, iy = NX // 2 + 7, NY // 2 - 4
    j = cell_index(all_obs, ix, iy)
    X = all_obs.xyz[j]
    ce = X + np.array([3000.0, 4000.0, 900.0])  # D = 5000 m horizontal, rho = 2000 m, H = 900 m
    rho, spd = 2000.0, 55.0
    tr = circle_track(ce, rho, spd, 0.0, 3 * 2 * math.pi * rho / spd, label="C")
    df = compute_cpa([tr], all_obs, [EventGate.window("all", 9000.0)])
    got = rows_at(df, ix, iy)
    d_true = math.hypot(5000.0 - rho, 900.0)
    ang = math.atan2(-4000.0, -3000.0) % (2 * math.pi)  # direction centre -> observer
    period = 2 * math.pi * rho / spd
    t_true = [ang / (spd / rho) + k * period for k in range(3)]
    assert got.height == 3
    assert np.abs(got["d"].to_numpy() - d_true).max() < 1.0
    assert np.abs(got["t_cpa"].to_numpy() - t_true).max() < 0.1
    assert np.abs(got["agl"].to_numpy() - (900.0 + 5.0)).max() < 2.0


def test_racetrack_multiple_minima(all_obs):
    ix, iy = NX // 2, NY // 2
    X = all_obs.xyz[cell_index(all_obs, ix, iy)]
    tr = racetrack(X[:2] + np.array([0.0, 1500.0]), 4000.0, 800.0, 45.0, 1000.0, 2.0, X[2] + 300.0,
                   label="rt")
    df = compute_cpa(to_frame(tr, "rt"), all_obs, [EventGate.window("all", 5000.0)])
    got = rows_at(df, ix, iy)
    ref = [p for p in track_passes(tr, X, r_max=5000.0) if _keep(tr, X, p.t_cpa)]
    got = _filter_rows(got, [tr], X)
    assert got.height >= 2 and got.height == len(ref)
    assert np.abs(got["t_cpa"].to_numpy() - [p.t_cpa for p in ref]).max() < 0.1
    assert np.abs(got["d"].to_numpy() - [p.d for p in ref]).max() < 1.0
    assert got["pass"].n_unique() == got.height  # pass keys are unique per (cell, flight)


def test_dem_changes_slant_distance():
    cells = CellSet.window(SPEC, SEARCH_LEVEL, NX // 2 - 3, NY // 2 - 3, 7, 7)
    lo, hi = observer_cells(cells, ORIGIN, flat(0.0)), observer_cells(cells, ORIGIN, flat(200.0))
    j = cell_index(lo, NX // 2, NY // 2)
    X = lo.xyz[j]
    # level overflight 1500 m above the low observer, horizontally offset 300 m
    tr = straight_pass_track(X, t_h=5000.0, v=70.0, d=math.hypot(1500.0, 300.0), height=1500.0, label="o")
    gate = [EventGate.window("all", 5000.0)]
    a = rows_at(compute_cpa([tr], lo, gate), NX // 2, NY // 2).row(0, named=True)
    b = rows_at(compute_cpa([tr], hi, gate), NX // 2, NY // 2).row(0, named=True)
    assert abs(a["d"] - math.hypot(1500.0, 300.0)) < 1.0
    assert abs(b["d"] - math.hypot(1300.0, 300.0)) < 1.0
    assert abs((a["t_h"] - b["t_h"]) - (a["d"] - b["d"]) / C) < 0.01  # sound delay follows the slant
    assert abs(a["agl"] - 1505.0) < 1.0 and abs(b["agl"] - 1305.0) < 1.0


# --- exhaustive comparison with the brute-force oracle ------------------------------------------


def _scene(all_obs, rng):
    X = all_obs.xyz[cell_index(all_obs, NX // 2, NY // 2)]
    z = X[2]
    trs = [
        straight_pass_track(X + [2000, -1000, 0], 3000.0, 62.0, 2500.0, height=900.0, heading_deg=40,
                            label="a"),
        straight_pass_track(X + [-3000, 2500, 0], 3400.0, 140.0, 5000.0, height=2000.0, heading_deg=200,
                            label="b"),
        straight_pass_track(X + [500, 3000, 0], 3800.0, 100.0, 1800.0, height=1200.0, heading_deg=300,
                            label="c"),
        circle_track(X + [-4000, -3000, 1100], 1800.0, 50.0, 2500.0, 1500.0, label="d"),
        racetrack(X[:2] + [3500, 3500], 3000.0, 700.0, 95.0, 3100.0, 1.5, z + 800.0, label="e"),
    ]
    # a climbing, turning track with finite-difference velocities
    t = 2000.0 + np.arange(0, 900.0)
    hd = np.radians(80) + 0.004 * (t - t[0])
    p = np.column_stack([X[0] - 6000 + np.cumsum(110 * np.sin(hd)), X[1] - 2000 + np.cumsum(110 * np.cos(hd)),
                         z + 600 + 4.0 * (t - t[0])])
    trs.append(Track.from_positions(t, p, label="f"))
    return trs


def _significant(tr: Track, X, t: float) -> bool:
    """A real pass: range still decreasing 1 s before and increasing 1 s after the minimum. Minima
    paired with a maximum inside one 1-s interval (sub-metre interpolation wiggles at curvature jumps)
    are below the kernel's node resolution by design, so both sides drop them."""
    p, v = tr.state(np.array([t - 1.0, t + 1.0]))
    g = np.sum((p - X) * v, axis=1)
    return bool(g[0] < 0 < g[1])


def _keep(tr: Track, X, t: float) -> bool:
    return tr.t[0] + 1.0 < t < tr.t[-1] - 1.0 and _significant(tr, X, t)


def _filter_rows(got: pl.DataFrame, trs, X) -> pl.DataFrame:
    by = {t.label: t for t in trs}
    keep = [_keep(by[r["icao"]], X, r["t_cpa"]) for r in got.iter_rows(named=True)]
    return got.filter(pl.Series(keep, dtype=pl.Boolean))


def _oracle(trs, X, r_max):
    out = []
    for tr in trs:
        for p in track_passes(tr, X, r_max=r_max):
            if _keep(tr, X, p.t_cpa):
                _, vv = tr.state(np.array([p.t_cpa]))
                out.append((tr.label, p, float(np.hypot(vv[0, 0], vv[0, 1]))))
    return sorted(out, key=lambda r: (r[0], r[1].t_cpa))


def test_all_passes_match_oracle(all_obs):
    rng = np.random.default_rng(1)
    trs = _scene(all_obs, rng)
    r_max = 8000.0
    df = compute_cpa(trs, all_obs, [EventGate.window("all", r_max)])
    n_ref = n_flat = 0
    for j in rng.choice(len(all_obs), 250, replace=False):
        X = all_obs.xyz[j]
        ref = [r for r in _oracle(trs, X, r_max) if abs(r[1].d - r_max) > 0.5]
        got = rows_at(df, int(all_obs.cells.ix[j]), int(all_obs.cells.iy[j]))
        got = got.filter((pl.col("d") - r_max).abs() > 0.5)
        got = _filter_rows(got, trs, X)
        assert got.height == len(ref), (j, got, ref)
        by = {t.label: t for t in trs}
        for (lb, p, _), row in zip(ref, got.iter_rows(named=True), strict=True):
            assert row["icao"] == lb
            if abs(row["t_cpa"] - p.t_cpa) >= 1e-3:
                # Very flat minimum at a curvature jump of the finite-difference racetrack: g(t) crosses
                # zero three times within ~1 s and the kernel (node resolution) keeps the wiggle minimum.
                pp, _ = by[lb].state(np.array([row["t_cpa"]]))
                assert abs(np.linalg.norm(pp[0] - X) - row["d"]) < 0.01
                assert abs(row["d"] - p.d) < 0.5 and abs(row["t_cpa"] - p.t_cpa) < 1.0
                n_flat += 1
                continue
            assert abs(row["d"] - p.d) < 0.01
            assert abs(row["t_h"] - p.t_h) < 1e-3 and abs(row["v"] - p.v) < 1e-3
        n_ref += len(ref)
    assert n_ref > 300 and n_flat < 0.02 * n_ref


def test_gates_match_oracle(all_obs):
    rng = np.random.default_rng(2)
    trs = [t for t in _scene(all_obs, rng) if t.label != "f"]  # level tracks: gs == 3-D speed
    gates = [
        EventGate.from_ranges("ev1", (130.0, 154.0), (4300.0, 7800.0)),
        EventGate.from_ranges("ev2", (88.0, 110.0), (1390.0, 4690.0), tau=(100.0, 110.0),
                              t_s=(2500.0, 3800.0)),
        EventGate(id="slow", v_lo=40.0, v_hi=70.0, d_lo=0.0, d_hi=3000.0, th_lo=2400.0, th_hi=4000.0),
    ]
    df = compute_cpa(trs, all_obs, gates)
    assert df["eligible"].min() > 0
    counts = np.zeros(3, int)
    for j in rng.choice(len(all_obs), 400, replace=False):
        X = all_obs.xyz[j]
        exp = []
        border = False
        for lb, p, gs in _oracle(trs, X, 9000.0):
            m = 0
            for k, g in enumerate(gates):
                near = (min(abs(p.d - g.d_lo), abs(p.d - g.d_hi)) < 0.5
                        or min(abs(gs - g.v_lo), abs(gs - g.v_hi)) < 0.05
                        or min(abs(p.t_h - g.th_lo), abs(p.t_h - g.th_hi)) < 0.01)
                border |= near
                if g.d_lo <= p.d <= g.d_hi and g.v_lo <= gs <= g.v_hi and g.th_lo <= p.t_h <= g.th_hi:
                    m |= 1 << k
            if m:
                exp.append((lb, round(p.t_cpa, 2), m))
        if border:
            continue
        got = _filter_rows(rows_at(df, int(all_obs.cells.ix[j]), int(all_obs.cells.iy[j])), trs, X)
        got = [(r["icao"], round(r["t_cpa"], 2), r["eligible"]) for r in got.iter_rows(named=True)]
        assert got == exp, (j, got, exp)
        for _, _, m in exp:
            counts += [(m >> k) & 1 for k in range(3)]
    assert (counts > 0).all()  # every gate was exercised


def test_gate_construction():
    ev = Event(id="e2", t_h=3073.0, t_h_range=Range(lo=3070.0, hi=3076.0), f0=130.0,
               f0_range=Range(lo=126, hi=134),
               v=99.0, v_range=Range(lo=88.0, hi=110.0), d=3000.0, d_range=Range(lo=1390.0, hi=4690.0))
    g = EventGate.from_event(ev, t_s=(1000.0, 2000.0))
    assert (g.v_lo, g.v_hi, g.d_lo, g.d_hi) == (73.0, 125.0, 1390.0, 4690.0)
    assert (g.th_lo, g.th_hi) == (1000.0 + 3070.0 - 5.0, 2000.0 + 3076.0 + 5.0)
    g = EventGate.from_event(ev)
    assert math.isinf(g.th_lo) and math.isinf(g.th_hi)
    with pytest.raises(ValueError):
        EventGate(id="x", v_lo=2, v_hi=1, d_lo=0, d_hi=1)
    with pytest.raises(ValueError):
        EventGate.from_ranges("x", (1, 2), (1, 2), t_s=(0, 1))


# --- candidate cells and the full Level-2 search --------------------------------------------------


def test_candidates_are_lossless_and_prune(all_obs, tmp_path):
    rng = np.random.default_rng(3)
    trs = _scene(all_obs, rng)
    gates = [EventGate.from_ranges("ev1", (130.0, 154.0), (4300.0, 7800.0)),
             EventGate.from_ranges("ev2", (88.0, 110.0), (1390.0, 4690.0))]
    full = compute_cpa(trs, all_obs, gates).sort("cell", "icao", "t_cpa")
    res = cpa_search(trs, ORIGIN, SPEC, gates, ground_hae=flat(), margin_m=0.0)
    assert len(res.observers) < len(all_obs)  # the speed-gated proximity mask prunes cells
    got = res.table.sort("cell", "icao", "t_cpa")
    assert got.height == full.height > 0
    assert (got["cell"] == full["cell"]).all() and (got["eligible"] == full["eligible"]).all()
    assert np.allclose(got["t_h"].to_numpy(), full["t_h"].to_numpy(), atol=1e-6)
    st = res.stats
    assert st["cells_near_tracks"] >= st["cells_candidates"] >= max(st["cells_speed_gated"].values())
    assert st["cells_eligible"]["ev1"] == full.filter(eligible_expr(0))["cell"].n_unique()
    assert st["cells_eligible_all"] <= min(st["cells_eligible"].values())
    # parquet round trip keeps the schema
    p = write_cpa(res.table, tmp_path / "cpa.parquet")
    back = read_cpa(p)
    assert back.schema == pl.Schema(CPA_SCHEMA) and back.height == res.table.height


def test_candidate_cells_radius():
    """A single slow level track: candidates are exactly the cells within d_hi + margin of it."""
    X = observer_cells(CellSet.window(SPEC, 3, NX // 2, NY // 2, 1, 1), ORIGIN, flat()).xyz[0]
    tr = straight_pass_track(X, 1000.0, 60.0, 1000.0, height=1000.0, heading_deg=0.0, half_span_s=20.0)
    segs = track_segments([tr])
    g = [EventGate.from_ranges("e", (50.0, 70.0), (500.0, 2000.0))]
    cs = candidate_cells(segs, ORIGIN, SPEC, g, margin_m=1000.0)
    obs = observer_cells(cs, ORIGIN, flat())
    # horizontal distance from each candidate to the track segment (north-south line through X)
    dn = np.clip(obs.xyz[:, 1], tr.pos[0, 1], tr.pos[-1, 1]) - obs.xyz[:, 1]
    dh = np.hypot(obs.xyz[:, 0] - X[0], dn)
    assert dh.max() <= 3000.0 + 200 and dh.max() >= 2800.0
    assert len(cs) < NX * NY / 4
    none = candidate_cells(segs, ORIGIN, SPEC, [EventGate.from_ranges("e", (100.0, 120.0), (500.0, 2000.0))])
    assert len(none) == 0  # no node passes the speed gate


def test_empty_inputs(all_obs):
    g = [EventGate.window("all", 1000.0)]
    empty = pl.DataFrame(schema={"icao": pl.String, "seg": pl.Int64, "t": pl.Float64, "e": pl.Float64,
                                 "n": pl.Float64, "u": pl.Float64})
    assert compute_cpa(empty, all_obs, g).height == 0
    res = cpa_search([], ORIGIN, SPEC, g, ground_hae=flat())
    assert res.table.height == 0 and res.stats["cells_eligible_all"] == 0


# --- performance ----------------------------------------------------------------------------------


def _bench():
    path = Path(__file__).resolve().parents[2] / "scripts" / "bench_cpa.py"
    spec = importlib.util.spec_from_file_location("bench_cpa", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_perf_smoke():
    """100 flights over 50 km for 3 h in well under the region budget (the full target is the slow test)."""
    out = _bench().run(n_flights=100, km=50.0, hours=3.0)
    assert out["rows"] > 10_000
    assert out["wall_s"] < 20.0


@pytest.mark.slow
def test_perf_region_target():
    """Plan 3.3.8: CPA stage, region mode (100 x 100 km, 3 h, 500 prop flights) <= 60 s on 4 cores."""
    t = time.perf_counter()
    out = _bench().run(n_flights=500, km=100.0, hours=3.0)
    print({k: out[k] for k in ("samples", "wall_s", "rows", "cells_candidates", "cells_eligible_all")},
          f"(incl. setup {time.perf_counter() - t:.1f} s)")
    assert out["wall_s"] <= 60.0
