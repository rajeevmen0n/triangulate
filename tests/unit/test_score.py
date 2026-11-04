"""WP11: locked scoring, null calibration, refinement and the acoustic layer on synthetic flyovers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from triangulate.acoustic.calibrate import calibrate, loglik
from triangulate.acoustic.physics import Track, track_passes
from triangulate.acoustic.score import ScoreConfig, ScoreMap, TrackBank, event_curve, score_hypotheses
from triangulate.acoustic.search import (
    SearchConfig,
    acoustic_search,
    band_maps,
    exempt_events_for,
)
from triangulate.acoustic.spectrogram import Spectrogram, compute_spectrogram
from triangulate.acoustic.synth import SynthSource, circle_track, straight_pass_track, synthesize
from triangulate.core.geo import geodetic_to_enu, haversine_m
from triangulate.core.types import Event, Range
from triangulate.fusion.grid import GridSpec, latlon_to_xy

BBOX = (40.80, -73.62, 41.00, -73.42)
SPEC = GridSpec.for_bbox(BBOX)
ORIGIN = (SPEC.lat0, SPEC.lon0)
GROUND = -30.0
T0 = 1.789e9  # true UTC epoch of audio sample 0
TRUE = (40.8935, -73.5051)  # observer (not a cell centre)
F0 = (110.0, 140.0)
DUR = 1200.0


def flat(lat, lon):
    return np.full_like(np.asarray(lat, float), GROUND)


def frame(tr: Track, icao: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "icao": icao,
            "seg": 0,
            "t": tr.t,
            "e": tr.pos[:, 0],
            "n": tr.pos[:, 1],
            "u": tr.pos[:, 2],
            "gs_ms": np.hypot(tr.vel[:, 0], tr.vel[:, 1]),
        }
    )


def shifted(tr: Track, dt: float, label: str) -> Track:
    return Track(t=tr.t + dt, pos=tr.pos, vel=tr.vel, label=label)


@dataclass
class Scene:
    X: np.ndarray
    spec: Spectrogram
    tracks: pl.DataFrame
    trs: dict[str, Track]
    events: list[Event]
    truth: list  # (tau_h, v, d) per event


def _turning_track(X: np.ndarray, t_h: float) -> Track:
    """A gentle turn (6 km radius) passing the observer at ~1.3 km slant, heard CPA at t_h."""
    R = 6000.0
    centre = X + np.array([R - 1100.0, 0.0, 700.0])
    tr = circle_track(centre, R, 65.0, 0.0, 900.0, phase0_deg=150.0, label="BBB")
    p = min(track_passes(tr, X), key=lambda q: q.d)
    return shifted(tr, t_h - p.t_h, "BBB")


@pytest.fixture(scope="module")
def scene() -> Scene:
    e, n, u = geodetic_to_enu(np.array([TRUE[0]]), np.array([TRUE[1]]), np.array([GROUND + 5.0]), *ORIGIN)
    X = np.array([e[0], n[0], u[0]])
    A = straight_pass_track(
        X, T0 + 300, 90.0, 1500.0, heading_deg=60, height=800, half_span_s=400, label="AAA"
    )
    B = _turning_track(X, T0 + 900)
    # a competing pair elsewhere with similar speeds/distances (a decoy tuple for the matcher)
    Xd = X + np.array([7000.0, -5000.0, 0.0])
    C = straight_pass_track(
        Xd, T0 + 450, 92.0, 1600.0, heading_deg=150, height=800, half_span_s=400, label="CCC"
    )
    D = straight_pass_track(
        Xd, T0 + 1050, 63.0, 1300.0, heading_deg=20, height=700, half_span_s=400, label="DDD"
    )
    rng = np.random.default_rng(3)
    others: list[Track] = []
    while len(others) < 8:  # unheard traffic: never within 5 km of the observer
        p = X + np.array([rng.uniform(-9000, 9000), rng.uniform(-7000, 7000), 0.0])
        tr = straight_pass_track(
            p,
            T0 + rng.uniform(0, DUR),
            rng.uniform(40, 90),
            rng.uniform(1500, 4000),
            heading_deg=rng.uniform(0, 360),
            half_span_s=400,
            label=f"O{len(others)}",
        )
        if np.linalg.norm(tr.pos - X, axis=1).min() > 5000:
            others.append(tr)
    trs = {t.label: t for t in [A, B, C, D, *others]}
    syn = synthesize(
        DUR,
        [SynthSource(A, f0=F0[0]), SynthSource(B, f0=F0[1])],
        X,
        t0=T0,
        noise_rms=0.4,
        hum=[(60.0, 0.05), (125.0, 0.05)],
        click_rate_per_min=2.0,
        click_amp=0.3,
        seed=1,
    )
    spec = compute_spectrogram(syn.audio)
    events, truth = [], []
    for k, st in enumerate(syn.truth):  # events as WP08 would report them (offsets inside the ranges)
        p = min(st.passes, key=lambda q: q.d)
        tau = p.t_h - T0
        truth.append((tau, p.v, p.d))
        events.append(
            Event(
                id=f"E{k}",
                t_h=tau + 0.7,
                t_h_range=Range(lo=tau - 2.3, hi=tau + 3.7),
                f0=F0[k] * 1.004,
                f0_range=Range(lo=F0[k] * 0.975, hi=F0[k] * 1.035),
                v=p.v * 1.03,
                v_range=Range(lo=p.v * 0.93, hi=p.v * 1.13),
                d=p.d * 1.1,
                d_range=Range(lo=p.d * 0.8, hi=p.d * 1.4),
                harmonics_seen=[1, 2, 3],
                snr=20.0,
            )
        )
    tracks = pl.concat([frame(t, t.label) for t in trs.values()])
    return Scene(X=X, spec=spec, tracks=tracks, trs=trs, events=events, truth=truth)


def _hyps(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "hid": np.arange(len(rows)),
            "cell": np.arange(len(rows)),
            "o": [r[2] for r in rows],
            "icao0": [r[0] for r in rows],
            "seg0": 0,
            "icao1": [r[1] for r in rows],
            "seg1": 0,
        }
    )


# --- scoring --------------------------------------------------------------------------------------


def test_truth_beats_200_decoys(scene: Scene):
    smap = ScoreMap.from_spectrogram(scene.spec, scene.events)
    bank = TrackBank.from_tracks(scene.tracks)
    rng = np.random.default_rng(7)
    o_true = T0 - 1.3  # the matcher's implied start is off by 1.3 s; dt* must recover it
    rows, obs = [("AAA", "BBB", o_true)], [scene.X]
    names = list(scene.trs)
    for _ in range(200):  # wrong places and/or wrong flights
        a, b = rng.choice(names, 2, replace=False) if rng.random() < 0.5 else ("AAA", "BBB")
        rows.append((str(a), str(b), T0 + rng.uniform(-5, 5)))
        r, phi = rng.uniform(300, 6000), rng.uniform(0, 2 * math.pi)
        obs.append(scene.X + np.array([r * math.cos(phi), r * math.sin(phi), 0.0]))
    hyps, xyz = _hyps(rows), np.array(obs)
    cal = calibrate(smap, bank, hyps, xyz)
    out = score_hypotheses(smap, bank, hyps, xyz, cal)
    assert int(out.sort("z_sum", descending=True)["hid"][0]) == 0
    t = out.row(0, named=True)
    assert abs(t["t_s"] - T0) <= 0.25  # dt* within 0.25 s
    assert abs(t["f0_0"] - F0[0]) <= 0.5 and abs(t["f0_1"] - F0[1]) <= 0.5
    assert t["z_0"] > 10 and t["z_1"] > 10
    assert t["z_sum"] - out["z_sum"][1:].max() > 3.0


def test_null_calibration(scene: Scene):
    smap = ScoreMap.from_spectrogram(scene.spec, scene.events)
    bank = TrackBank.from_tracks(scene.tracks)
    rng = np.random.default_rng(1)
    rows, obs = [], []
    for _ in range(60):
        rows.append(("AAA", "BBB", T0 + rng.uniform(-5, 5)))
        obs.append(scene.X + np.array([rng.uniform(-8000, 8000), rng.uniform(-8000, 8000), 0.0]))
    cal = calibrate(smap, bank, _hyps(rows), np.array(obs))
    for ec in cal.events:
        assert ec.n >= 60 * 10 and ec.sigma > 0 and abs(ec.mu) < 0.5  # ridge-filtered map: noise ~ 0
        assert ec.q90 < 6.0  # decoys are mostly noise
    assert cal.ell_floor == pytest.approx(math.log(0.1))
    assert loglik(10.0, cal.lam) == pytest.approx(5.0)


def test_lock_beats_unlocked(scene: Scene):
    """A decoy with the exact curve shapes but timings inconsistent by 10 s (each event individually
    perfect; the matcher's +-5 s slack admits it) beats a 40-m-off hypothesis when the events float
    independently, but not when they are locked to one T_s."""
    A, B = scene.trs["AAA"], scene.trs["BBB"]
    tracks = pl.concat(
        [scene.tracks, frame(shifted(A, 5.0, "A+5"), "A+5"), frame(shifted(B, -5.0, "B-5"), "B-5")]
    )
    smap = ScoreMap.from_spectrogram(scene.spec, scene.events)
    bank = TrackBank.from_tracks(tracks)
    near = scene.X + np.array([40.0 * math.cos(1.0), 40.0 * math.sin(1.0), 0.0])
    hyps = _hyps([("AAA", "BBB", T0), ("A+5", "B-5", T0)])
    xyz = np.array([near, scene.X])
    cal = calibrate(smap, bank, hyps, xyz)
    out = score_hypotheses(smap, bank, hyps, xyz, cal)
    near_r, decoy = out.row(0, named=True), out.row(1, named=True)
    assert decoy["z_sum_unlocked"] > near_r["z_sum_unlocked"]  # unlocked picks the decoy
    assert near_r["z_sum"] > decoy["z_sum"] + 5.0  # the shared-dt lock rejects it


def test_traced_line_event(scene: Scene):
    """A hand-traced polyline (LINE_POINTS) switches event 0 to the distance-to-line score."""
    A = scene.trs["AAA"]
    tau0 = scene.truth[0][0]
    tt = np.arange(tau0 - 40, tau0 + 40, 2.0)
    rng = np.random.default_rng(0)
    f = event_curve(A, scene.X, T0, tt, F0[0]) + rng.normal(0, 0.4, tt.size)
    ev0 = scene.events[0].model_copy(
        update={"traced_line": [(float(a), float(b)) for a, b in zip(tt, f, strict=True)]}
    )
    events = [ev0, scene.events[1]]
    smap = ScoreMap.from_spectrogram(scene.spec, events)
    assert smap.plans[0].kind == "line" and smap.plans[1].kind == "brightness"
    bank = TrackBank.from_tracks(scene.tracks)
    rows, obs = [("AAA", "BBB", T0 + 0.8)], [scene.X]
    for i in range(80):
        phi = 2 * math.pi * i / 80
        r = 400 + 60 * i
        rows.append(("AAA", "BBB", T0))
        obs.append(scene.X + np.array([r * math.cos(phi), r * math.sin(phi), 0.0]))
    hyps, xyz = _hyps(rows), np.array(obs)
    cal = calibrate(smap, bank, hyps, xyz)
    assert cal.events[0].kind == "line"
    out = score_hypotheses(smap, bank, hyps, xyz, cal)
    assert int(out.sort("z_sum", descending=True)["hid"][0]) == 0
    t = out.row(0, named=True)
    assert t["S_0"] > -0.5  # mean |error| below sigma_f / 4
    assert abs(t["t_s"] - T0) <= 0.5


# --- search -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result(scene: Scene):
    return acoustic_search(
        scene.spec, scene.events, scene.tracks, SPEC, ORIGIN, (T0 - 1800, T0 + 1800), ground_hae=flat
    )


def test_search_recovers_truth(result):
    res = result
    assert res.ok and res.tuples["tuple"][0] == "AAA/0|BBB/0"
    b = res.best()
    assert b["level"] >= 5
    d_best = haversine_m(b["lat"], b["lon"], *TRUE)
    coarse = res.top(1, level=3).row(0, named=True)
    d_coarse = haversine_m(coarse["lat"], coarse["lon"], *TRUE)
    assert d_best <= 40.0, (d_best, d_coarse)  # refined 200 m -> 25 m
    assert haversine_m(b["lat_c"], b["lon_c"], *TRUE) <= 60.0
    assert abs(b["t_s"] - T0) <= 0.25
    assert res.margin_z is not None and res.margin_z > 0
    assert set(res.hypotheses["level"].unique().to_list()) == {3, 4, 5, 6}
    st = res.stats
    assert st["hyps_after_silence"] <= st["hyps_matched"] and st["tuples_after_silence"] >= 2
    s = res.summary()
    assert s["tuples"][0]["flights"] == ["AAA", "BBB"] and s["tuples"][0]["t_s"].endswith("Z")


def test_acoustic_layer(result, tmp_path):
    res = result
    lay = res.layer
    assert lay.module == "acoustic" and lay.fill == pytest.approx(math.log(0.1))
    b = res.best()
    x, y = latlon_to_xy(SPEC, b["lat"], b["lon"])
    assert float(lay.value_at_xy(x, y)) == pytest.approx(res.config.lam * b["z_sum"], abs=1e-6)
    xf, yf = latlon_to_xy(SPEC, 40.95, -73.45)  # far from every hypothesis
    assert float(lay.value_at_xy(xf, yf)) == pytest.approx(math.log(0.1))
    a = lay.attrs_at_xy(float(x), float(y))
    assert a is not None and a["flights"] == ["AAA", "BBB"] and abs(a["t_s"] - T0) < 0.3
    paths = res.write(tmp_path)
    assert paths["scores"].exists() and paths["layer"].exists()
    mr = res.module_result(tmp_path, attribution="Flight data (c) test")
    assert mr.status == "ok" and mr.layers[0].module == "acoustic"


def test_band_maps(result):
    bm = band_maps(result, radius_m=2500.0, level=4)
    i = int(np.nanargmax(bm["z_sum"].to_numpy()))
    assert haversine_m(bm["lat"][i], bm["lon"][i], *TRUE) <= 150.0
    # each single-event map is a band: many more cells score well alone than together
    zs = np.nan_to_num(bm["z_sum"].to_numpy(), nan=-1e9)
    blob = int((zs > 0.5 * zs.max()).sum())
    for k in (0, 1):
        z = np.nan_to_num(bm[f"z_{k}"].to_numpy(), nan=-1e9)
        assert (z > 0.5 * z.max()).sum() > 2 * blob


def test_silence_full_scope_and_exemptions(scene: Scene):
    evs = [
        *scene.events,
        scene.events[0].model_copy(update={"id": "X1", "selected": False, "quality_flags": []}),
        scene.events[0].model_copy(update={"id": "X2", "selected": False, "quality_flags": ["one-sided"]}),
        scene.events[0].model_copy(update={"id": "X3", "selected": False, "quality_flags": ["non-aircraft"]}),
    ]
    assert [e.id for e in exempt_events_for(evs, "usable")] == ["X1"]
    assert [e.id for e in exempt_events_for(evs, "aircraft")] == ["X1", "X2"]
    assert [e.id for e in exempt_events_for(evs, "all")] == ["X1", "X2", "X3"]
    assert exempt_events_for(evs, "none") == []
    cfg = SearchConfig(silence_scope="full", refine_levels=(4,), top_tuples=(5,))
    res = acoustic_search(
        scene.spec,
        scene.events,
        scene.tracks,
        SPEC,
        ORIGIN,
        (T0 - 1800, T0 + 1800),
        ground_hae=flat,
        config=cfg,
    )
    assert res.ok and res.tuples["tuple"][0] == "AAA/0|BBB/0"  # unheard traffic stays > 5 km away


def test_no_hypotheses_gives_flat_floor(scene: Scene):
    res = acoustic_search(
        scene.spec, scene.events, scene.tracks, SPEC, ORIGIN, (T0 + 5000, T0 + 6000), ground_hae=flat
    )
    assert not res.ok and res.layer is not None and res.layer.fill == pytest.approx(math.log(0.1))
    assert res.layer.patches == []


def test_score_config_grid():
    cfg = ScoreConfig()
    assert cfg.n_coarse == 21 and cfg.coarse_step == 2  # +-5 s in 0.5 s steps on 0.25 s frames
