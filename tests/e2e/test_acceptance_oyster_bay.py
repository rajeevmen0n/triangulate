"""WP23 acceptance test, Oyster Bay (plan 8.2). Drives the real pipeline (the same path as the CLI/web:
`create_job` / `run_job` with the real stages) on `audio.mp3`, decoded PCM only. Container metadata is never
read (`use_metadata=False`, TRIANGULATE_SKIP_METADATA=1).

ADS-B comes from the region fixture built by `scripts/build_acceptance_fixture.py` (built on first use, which
needs the network and the adsb.lol cache); it is injected through the `SERVICES.track_provider` seam, so every
other stage is the production one. DEM tiles are fetched on first use (network) and cached.

Constraints as a user would type them: date 2026-09-08, T_s window 19:30-22:30 UTC (entered as 15:30-18:30 in
America/New_York, which is how the pipeline interprets wall-clock windows), bbox 40.50,-74.00,41.20,-73.00,
h_obs 5 m, visual off.

A1 (required)  auto-detected events: truth in top-3 candidates and best <= 500 m.
A2 (required)  events seeded from the video's ranges: top-1 <= 250 m and acoustic margin z > 0.
A3 (report)    T_s* vs 20:15:44Z (the notes' anchor; fails only if > 600 s) and vs 20:17:32Z (logged).
A4 (report)    single event at tau ~1276 s with a known clip time 20:37Z +- 90 s: logged, truth within 1 km.
A5 (optional)  unknown time, full UTC day, real provider: only with TRIANGULATE_ACCEPT_A5=1 (needs a full-day
               region extract by range requests, ~50-150 MB, minutes of runtime): truth in top-5.

Run: nix develop --command pytest -m acceptance tests/e2e/test_acceptance_oyster_bay.py -s
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from triangulate.core.geo import haversine_m
from triangulate.core.types import Event, Range, TrackSet
from triangulate.jobs import stages
from triangulate.jobs.pipeline import JobParams
from triangulate.jobs.runner import create_job, make_store, run_job
from triangulate.jobs.stages import validate_results

pytestmark = [pytest.mark.acceptance, pytest.mark.network, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
GT = (40.876722, -73.526444)
BBOX = (40.50, -74.00, 41.20, -73.00)
TAU = (2152.0, 3073.0)
NOTES_TS = datetime(2026, 9, 8, 20, 15, 44, tzinfo=UTC)  # the notes' anchor (plan A3)
TRUE_TS = datetime(2026, 9, 8, 20, 17, 32, tzinfo=UTC)  # measured: WP07/WP10/WP11 (~20:17:30-35Z)
EXPECTED = ({"a1dfeb"}, {"ace391"})


def _constraints(**kw) -> dict:
    c = {"bbox": BBOX, "date_start": "2026-09-08", "tz": "America/New_York",
         "window_start": "15:30", "window_end": "18:30", "h_obs_m": 5.0}
    c.update(kw)
    return c


def _seed(i: int, tau: float, v: tuple[float, float], d: tuple[float, float]) -> dict:
    e = Event(id=f"ev{i}", t_h=tau, t_h_range=Range(lo=tau - 3, hi=tau + 3), f0=150.0,
              f0_range=Range(lo=100, hi=200), v=sum(v) / 2, v_range=Range(lo=v[0], hi=v[1]),
              d=sum(d) / 2, d_range=Range(lo=d[0], hi=d[1]))
    return e.model_dump(mode="json")


# The video's measured ranges [N sec 2, 10:33]
SEEDED = [_seed(1, TAU[0], (130.0, 154.0), (4300.0, 7800.0)),
          _seed(2, TAU[1], (88.0, 110.0), (1390.0, 4690.0))]


class FixtureProvider:
    """Track provider serving the committed-region fixture (no network)."""

    name = "adsblol"

    def __init__(self, ts: TrackSet):
        self.ts, self.licence = ts, ts.licence

    def fetch(self, q, progress=None) -> TrackSet:
        s, w, n, e = q.bbox
        t = self.ts.tracks.filter(pl.col("t").is_between(q.t0.timestamp(), q.t1.timestamp())
                                  & pl.col("lat").is_between(s, n) & pl.col("lon").is_between(w, e))
        ac = self.ts.aircraft.join(t.select("icao").unique(), on="icao", how="semi")
        return TrackSet(tracks=t, aircraft=ac, provider=self.name, licence=self.ts.licence)


def _builder():
    spec = importlib.util.spec_from_file_location("build_acceptance_fixture",
                                                  ROOT / "scripts" / "build_acceptance_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audio() -> Path:
    p = ROOT / "audio.mp3"
    if not p.exists():
        pytest.skip("audio.mp3 not present")
    return p


@pytest.fixture(scope="module")
def fixture_provider():
    ts = _builder().load_trackset()  # reuses the fixture if present, else builds it (network)
    old = stages.SERVICES.track_provider
    stages.SERVICES.track_provider = lambda: FixtureProvider(ts)
    yield
    stages.SERVICES.track_provider = old


@pytest.fixture(autouse=True)
def _no_metadata(monkeypatch):
    monkeypatch.setenv("TRIANGULATE_SKIP_METADATA", "1")


def _run(audio: Path, params: JobParams) -> tuple[dict, float]:
    from triangulate.config import get_settings

    stages.register_real_stages()
    s = get_settings()
    store = make_store(s)
    t = time.perf_counter()
    job = run_job(store, s, create_job(store, s, params, audio))
    dt = time.perf_counter() - t
    assert job["status"] == "done", job["error"]
    res = json.loads(Path(job["result_path"]).read_text())
    validate_results(res)
    return res, dt


def _dist(c: dict) -> float:
    return float(haversine_m(c["lat"], c["lon"], *GT))


def _iso_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _report(tag: str, res: dict, dt: float) -> None:
    top = res["best"]
    ranks = [(c["rank"], round(_dist(c)), [f[:6] for f in c["flights"]]) for c in res["candidates"][:5]]
    print(f"\n{tag}: best {top['lat']:.5f},{top['lon']:.5f} dist {_dist(top):.0f} m  "
          f"flights {top['flights']}  T_s {top['t_s']}  margin {top['margin_nats']} nats / "
          f"z {res['margin']['acoustic_z']}  events {res['events']['ids_used']}  runtime {dt:.0f} s\n"
          f"   top candidates (rank, dist m, flights): {ranks}")


@pytest.fixture(scope="module")
def a1(audio, fixture_provider):
    params = JobParams(constraints=_constraints(), visual=False, use_metadata=False)
    return _run(audio, params)


@pytest.fixture(scope="module")
def a2(audio, fixture_provider):
    params = JobParams(constraints=_constraints(), visual=False, use_metadata=False, events=SEEDED)
    return _run(audio, params)


def test_a1_auto_detected(a1):
    res, dt = a1
    _report("A1 auto-detected", res, dt)
    assert res["events"]["source"] == "detected" and res["events"]["n_used"] >= 1
    top3 = res["candidates"][:3]
    truth_rank = next((c["rank"] for c in res["candidates"] if _dist(c) <= 500.0), None)
    print(f"A1 truth rank (first candidate within 500 m): {truth_rank}")
    assert any(_dist(c) <= 500.0 for c in top3), "ground truth not inside the top-3 candidates"
    assert _dist(res["best"]) <= 500.0


def test_a2_seeded(a2):
    res, dt = a2
    _report("A2 seeded", res, dt)
    assert res["events"]["source"] == "user" and res["events"]["n_used"] == 2
    got = [{f[:6] for f in res["best"]["flights"]}]
    print(f"A2 flights matched: {sorted(got[0])} (expected {sorted(EXPECTED[0] | EXPECTED[1])})")
    assert _dist(res["best"]) <= 250.0
    z = res["margin"]["acoustic_z"]
    assert z is not None and z > 0, f"margin to runner-up tuple (z) = {z}"


def test_a3_ts_estimate(a2):
    res, _ = a2
    ts = _iso_ts(res["T_s"]["best"] or res["best"]["t_s"])
    e_notes, e_true = (ts - NOTES_TS).total_seconds(), (ts - TRUE_TS).total_seconds()
    print(f"\nA3: T_s* {ts.isoformat()}  error vs 20:15:44Z (notes) {e_notes:+.1f} s "
          f"[within 120: {abs(e_notes) <= 120}; hard-fail above 600]  "
          f"error vs 20:17:32Z (measured) {e_true:+.1f} s")
    assert abs(e_notes) <= 600.0


def test_a4_single_event(audio, fixture_provider):
    """Report-only (plan A4): logs the numbers and the 1 km criterion (the video got 481 m); does not fail."""
    from triangulate.acoustic.detect import find_events
    from triangulate.acoustic.spectrogram import decode_and_whiten

    tau = 1276.0
    near = [e for e in find_events(decode_and_whiten(audio)).events if abs(e.t_h - tau) <= 30.0]
    if near:
        ev = near[0].model_copy(update={"selected": True}).model_dump(mode="json")
        src = f"auto-detected ({ev['id']} at t_h {ev['t_h']:.1f} s)"
    else:  # the committed polyline tests/fixtures/single_event_trace.json does not exist; seed instead
        ev = _seed(1, tau, (60.0, 200.0), (300.0, 8000.0))
        ev["f0_range"] = {"lo": 300.0, "hi": 600.0}
        ev["f0"] = 450.0
        src = "seeded wide ranges (detection found no event within 30 s)"
    known = {"audio_pos_s": tau, "utc": "2026-09-08T20:37:00Z", "sigma_s": 90.0}
    params = JobParams(constraints=_constraints(known_audio_time=known), visual=False, use_metadata=False,
                       events=[ev])
    res, dt = _run(audio, params)
    print(f"\nA4 event source: {src}")
    _report("A4 single event", res, dt)
    assert res["events"]["n_used"] == 1
    print(f"A4 min distance among top-3: {min(_dist(c) for c in res['candidates'][:3]):.0f} m")
    ok = _dist(res["best"]) <= 1000.0
    print(f"A4 criterion (top-1 within 1 km; report-only): {'PASS' if ok else 'FAIL (not asserted)'}")


@pytest.mark.skipif(os.environ.get("TRIANGULATE_ACCEPT_A5") != "1",
                    reason="A5 (full-day unknown time) is optional; set TRIANGULATE_ACCEPT_A5=1")
def test_a5_unknown_time_full_day(audio):
    c = _constraints(tz="UTC")
    c.pop("window_start"), c.pop("window_end")
    params = JobParams(constraints=c, visual=False, use_metadata=False, events=SEEDED)
    old = stages.SERVICES.track_provider
    stages.SERVICES.track_provider = None  # real adsb.lol provider (cached members)
    try:
        res, dt = _run(audio, params)
    finally:
        stages.SERVICES.track_provider = old
    _report("A5 full day", res, dt)
    assert any(_dist(x) <= 500.0 for x in res["candidates"][:5])
