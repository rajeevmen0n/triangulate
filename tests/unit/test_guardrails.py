"""WP24: consent gate, per-day rate limit, retention purge (path/symlink safety), attribution."""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from triangulate import cli
from triangulate.api import create_app
from triangulate.config import Settings
from triangulate.jobs import pipeline as P
from triangulate.jobs import store as S
from triangulate.jobs.retention import purge_expired
from triangulate.jobs.runner import make_store

CONS = {"bbox": [40.0, -74.0, 41.0, -73.0], "date_start": "2026-09-08"}
OLD = time.time() - 30 * 86400  # well past the 7-day default


def three():
    deps = {"ingest": (), "spectrogram": ("ingest",), "events": ("spectrogram",)}
    return [P.StubStage(n, deps[n]) for n in deps]


@pytest.fixture(autouse=True)
def _stubs():
    P.reset_registry()
    yield
    P.reset_registry()


@pytest.fixture
def settings(tmp_path):
    return Settings(data=tmp_path / "data", max_upload_bytes=1000, retention_days=7, max_jobs_per_day=2)


def add_job(store, s, jid, *, created, status=S.DONE, files=("input.wav",)):
    """Insert a job row plus a workspace with some files, as the app would have made it."""
    store.create({"constraints": CONS}, job_id=jid)
    store._exec("UPDATE jobs SET created=?, status=? WHERE id=?", (created, status, jid))
    ws = s.jobs_dir / jid
    ws.mkdir(parents=True, exist_ok=True)
    for f in files:
        (ws / f).write_bytes(b"data")
    return ws


def post(c, name="a.wav"):
    return c.post(
        "/api/jobs", data={"constraints": json.dumps(CONS)}, files={"file": (name, b"fake audio")}
    )


# -- retention purge

def test_purge_deletes_old_workspaces_and_rows_only(settings):
    st = make_store(settings)
    old = add_job(st, settings, "aaaaaaaaaaaa", created=OLD)
    new = add_job(st, settings, "bbbbbbbbbbbb", created=time.time())
    rep = purge_expired(settings, st)
    assert rep["jobs_purged"] == ["aaaaaaaaaaaa"] and rep["errors"] == []
    assert not old.exists() and st.get("aaaaaaaaaaaa") is None
    assert new.exists() and st.get("bbbbbbbbbbbb") is not None


def test_purge_dry_run_deletes_nothing(settings):
    st = make_store(settings)
    old = add_job(st, settings, "aaaaaaaaaaaa", created=OLD)
    rep = purge_expired(settings, st, dry_run=True)
    assert rep["jobs_purged"] == ["aaaaaaaaaaaa"] and rep["dry_run"] is True
    assert old.exists() and st.get("aaaaaaaaaaaa") is not None


def test_purge_disabled_when_days_zero(settings):
    st = make_store(settings)
    old = add_job(st, settings, "aaaaaaaaaaaa", created=OLD)
    rep = purge_expired(settings, st, days=0)
    assert rep["enabled"] is False and old.exists()


def test_purge_skips_queued_and_running(settings):
    st = make_store(settings)
    q = add_job(st, settings, "aaaaaaaaaaaa", created=OLD, status=S.QUEUED)
    r = add_job(st, settings, "bbbbbbbbbbbb", created=OLD, status=S.RUNNING)
    rep = purge_expired(settings, st)
    assert rep["jobs_purged"] == [] and q.exists() and r.exists()


def test_purge_never_touches_cache_or_outside_data(settings, tmp_path):
    st = make_store(settings)
    cache_file = settings.cache_subdir("stages") / "k" / "keep.bin"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep")
    # path traversal into the cache, a parent-dir id, a separator id, and a symlink to outside
    for bad in ("../cache", "..", "x/../../cache", "a/b"):
        st.create({"constraints": CONS}, job_id=bad)
        st._exec("UPDATE jobs SET created=? WHERE id=?", (OLD, bad))
    link = add_job(st, settings, "linkjob0000", created=OLD, files=())
    link.rmdir()
    link.symlink_to(outside, target_is_directory=True)

    rep = purge_expired(settings, st)

    assert cache_file.read_bytes() == b"cache"
    assert (outside / "precious.txt").read_text() == "keep"
    assert link.is_symlink()
    assert rep["jobs_purged"] == []
    assert len(rep["skipped"]) == 5
    assert all(st.get(j) is not None for j in ("../cache", "..", "x/../../cache", "a/b", "linkjob0000"))


def test_purge_symlink_inside_jobs_dir_is_not_followed(settings, tmp_path):
    st = make_store(settings)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep")
    ws = add_job(st, settings, "aaaaaaaaaaaa", created=OLD, files=())
    (ws / "inner").symlink_to(outside, target_is_directory=True)
    rep = purge_expired(settings, st)
    assert rep["jobs_purged"] == ["aaaaaaaaaaaa"]
    assert not ws.exists()
    assert (outside / "precious.txt").read_text() == "keep"  # the link was removed, not its target


def test_purge_tmp_staging_by_age_and_symlinks(settings, tmp_path):
    st = make_store(settings)
    tmp = settings.data / "tmp"
    stale, fresh = tmp / "stale", tmp / "fresh"
    stale.mkdir(parents=True)
    fresh.mkdir()
    (stale / "upload.wav").write_bytes(b"x")
    os.utime(stale, (OLD, OLD))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep")
    (tmp / "evil").symlink_to(outside, target_is_directory=True)
    os.utime(tmp / "evil", (OLD, OLD), follow_symlinks=False)
    rep = purge_expired(settings, st)
    assert rep["tmp_removed"] == ["stale"] and not stale.exists()
    assert fresh.exists()
    assert (tmp / "evil").is_symlink() and (outside / "precious.txt").read_text() == "keep"


def test_startup_purge_runs_in_create_app(settings):
    st = make_store(settings)
    old = add_job(st, settings, "aaaaaaaaaaaa", created=OLD)
    create_app(settings, stages=three(), start_runner=False)
    assert not old.exists() and st.get("aaaaaaaaaaaa") is None


def test_cli_purge(settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    st = make_store(settings)
    old = add_job(st, settings, "aaaaaaaaaaaa", created=OLD)
    assert cli.main(["purge", "--dry-run"]) == 0
    assert old.exists() and json.loads(capsys.readouterr().out)["dry_run"] is True
    assert cli.main(["purge"]) == 0
    assert not old.exists()
    assert cli.main(["purge", "--days", "0"]) == 2


# -- API: rate limit, consent gate, attribution

@pytest.fixture
def client(settings):
    app = create_app(settings, stages=three(), poll_interval=0.02)
    with TestClient(app) as c:
        yield c


def test_rate_limit_applies_to_upload_and_rerun(client):
    r1 = post(client)
    assert r1.status_code == 201
    assert post(client).status_code == 201
    r3 = post(client)
    assert r3.status_code == 429 and "per 24 hours" in r3.text
    assert client.post(f"/api/jobs/{r1.json()['id']}/rerun", json={}).status_code == 429
    assert len(client.get("/api/jobs").json()) == 2  # refused requests create no job


def test_rate_limit_zero_disables(settings):
    settings.max_jobs_per_day = 0
    with TestClient(create_app(settings, stages=three(), poll_interval=0.02)) as c:
        assert all(post(c).status_code == 201 for _ in range(3))


def test_consent_gate_upload_and_rerun(client):
    r = client.post("/api/jobs", data={"constraints": json.dumps(CONS), "visual": "true"},
                    files={"file": ("a.wav", b"x")})
    assert r.status_code == 409
    ok = post(client)
    assert ok.status_code == 201
    assert client.post(f"/api/jobs/{ok.json()['id']}/rerun", json={"visual": True}).status_code == 409


def test_config_and_result_carry_attribution(client):
    cfg = client.get("/api/config").json()
    joined = " | ".join(cfg["attribution"])
    for needle in ("adsb.lol", "ODbL", "OpenStreetMap", "Copernicus DEM", "Leaflet", "Doc 8643"):
        assert needle in joined, needle
    assert "uncertainty" in cfg["use_notice"] and "private individuals" in cfg["use_notice"]
    jid = post(client).json()["id"]
    with client.stream("GET", f"/api/jobs/{jid}/stream") as r:
        for _ in r.iter_lines():
            pass
    res = client.get(f"/api/jobs/{jid}/result")
    assert res.status_code == 200 and res.json()["attribution"] == cfg["attribution"]
