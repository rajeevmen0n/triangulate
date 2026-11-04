"""WP25: the real stages end to end on a small synthetic recording, with an in-memory track provider
and a flat ground (no network). Asserts the artifact set and the results.json schema."""

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from wp08_helpers import make_scene

from triangulate.acoustic.synth import FS_DEFAULT, write_wav
from triangulate.config import Settings
from triangulate.core.types import AIRCRAFT_SCHEMA, TRACK_SCHEMA, TrackSet
from triangulate.jobs import pipeline as P
from triangulate.jobs import stages as ST
from triangulate.jobs.runner import create_job, make_store, run_job

BBOX = (40.80, -73.62, 40.96, -73.44)
LAT0, LON0 = 40.88, -73.53
T_S = datetime(2026, 9, 8, 20, 0, 0, tzinfo=UTC).timestamp()
PASSES = [dict(label="A", t_h=120.0, v=100.0, d=800.0, f0=140.0, snr_db=15.0, heading=90.0),
          dict(label="B", t_h=230.0, v=80.0, d=1500.0, f0=110.0, snr_db=15.0, heading=270.0)]
ARTIFACTS = ["results.json", "candidates.geojson", "heatmap.png", "heatmap.json", "spectrogram.png",
             "spectrogram.json", "events.json", "fusion.json", "posterior.tif", "posterior.npz"]


class FakeProvider:
    name, licence = "fake", "test"

    def days_available(self, t0, t1):
        return []

    def fetch(self, q, progress):
        rows = []
        m = 111_320.0
        for icao, tcpa, v, d, sgn in (("aaa111", T_S + 120 - 800 / 343, 100.0, 800.0, 1),
                                      ("bbb222", T_S + 230 - 1500 / 343, 80.0, 1500.0, -1)):
            hd = float(np.sqrt(d**2 - 300.0**2))
            for t in np.arange(tcpa - 400, tcpa + 400, 1.0):
                east = sgn * v * (t - tcpa)
                rows.append(dict(icao=icao, reg="N1", typecode="C172", category="A1", t=float(t),
                                 lat=LAT0 + hd / m, lon=LON0 + east / (m * np.cos(np.radians(LAT0))),
                                 alt_baro_ft=300 / 0.3048, alt_geom_ft=300 / 0.3048, gs_kt=v * 1.944,
                                 track_deg=90.0 if sgn > 0 else 270.0, vrate_fpm=0.0, on_ground=False,
                                 source="test"))
        tr = pl.DataFrame(rows, schema=TRACK_SCHEMA)
        ac = pl.DataFrame({"icao": ["aaa111", "bbb222"], "reg": ["N1", "N2"], "typecode": ["C172"] * 2,
                           "desc": ["", ""], "category": ["A1"] * 2, "db_flags": [0, 0], "owner_op": ["", ""],
                           "year": ["", ""]}, schema=AIRCRAFT_SCHEMA)
        return TrackSet(tracks=tr, aircraft=ac, provider="fake", licence="test")


@pytest.fixture
def real(monkeypatch, tmp_path):
    monkeypatch.setattr(ST.SERVICES, "track_provider", FakeProvider)
    monkeypatch.setattr(ST.SERVICES, "ground_hae", lambda lat, lon: np.zeros_like(np.asarray(lat, float)))
    ST.register_real_stages()
    yield Settings(data=tmp_path / "data")
    P.reset_registry()


def _run(s, f, **kw):
    st = make_store(s)
    params = P.JobParams(constraints={"bbox": BBOX, "date_start": "2026-09-08", "tz": "UTC",
                                      "window_start": "19:55", "window_end": "20:05"},
                         use_metadata=False, **kw)
    jid = create_job(st, s, params, f)
    return st, jid, run_job(st, s, jid)


def test_real_pipeline_synthetic(real, tmp_path):
    s = real
    sc = make_scene(360, PASSES, seed=3)
    f = write_wav(tmp_path / "in.wav", sc.audio, FS_DEFAULT)
    st, jid, job = _run(s, f)
    assert job["status"] == "done", job["error"]
    ws = s.jobs_dir / jid
    for a in ARTIFACTS:
        assert (ws / a).is_file(), a
    res = json.loads((ws / "results.json").read_text())
    ST.validate_results(res)
    print(json.dumps({k: res[k] for k in ("status", "best", "warnings", "events")}))
    assert res["events"]["source"] == "detected" and res["events"]["n_used"] >= 1
    assert res["modules"]["metadata"]["status"] == "skipped"
    assert res["modules"]["visual"]["status"] == "skipped"
    assert res["warnings"]
    assert not (ws / "metadata.json").exists()  # metadata never read when disabled
    ov = json.loads((ws / "spectrogram.json").read_text())
    assert ov["image"] and ov["events"]
    gj = json.loads((ws / "candidates.geojson").read_text())
    assert gj["features"]


def test_user_events_override_detection(real, tmp_path):
    s = real
    sc = make_scene(360, PASSES, seed=3)
    f = write_wav(tmp_path / "in.wav", sc.audio, FS_DEFAULT)
    _, jid, job = _run(s, f)
    evs = json.loads((s.jobs_dir / jid / "events.json").read_text())
    keep = evs[:1]
    _, jid2, job2 = _run(s, f, events=keep)
    assert job2["status"] == "done", job2["error"]
    res = json.loads((s.jobs_dir / jid2 / "results.json").read_text())
    ST.validate_results(res)
    assert res["events"]["source"] == "user" and res["events"]["n_total"] == 1
    assert any("one aircraft event" in w for w in res["warnings"])


def test_visual_needs_consent_and_key(real, tmp_path):
    s = real
    sc = make_scene(360, PASSES, seed=3)
    f = write_wav(tmp_path / "in.wav", sc.audio, FS_DEFAULT)
    _, jid, job = _run(s, f, visual=True, visual_consent=False)
    assert job["status"] == "done", job["error"]
    res = json.loads((s.jobs_dir / jid / "results.json").read_text())
    assert res["modules"]["visual"]["status"] == "skipped"
    assert res["cost_usd"] == 0
