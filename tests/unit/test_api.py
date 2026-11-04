import json

import pytest
from fastapi.testclient import TestClient

from triangulate.api import create_app
from triangulate.config import Settings
from triangulate.jobs import pipeline as P

CONS = {"bbox": [40.0, -74.0, 41.0, -73.0], "date_start": "2026-09-08"}


def three():
    deps = {"ingest": (), "spectrogram": ("ingest",), "events": ("spectrogram",)}
    return [P.StubStage(n, deps[n]) for n in deps]


@pytest.fixture
def client(tmp_path):
    P.reset_registry()
    s = Settings(data=tmp_path / "data", max_upload_bytes=1000)
    app = create_app(s, stages=three(), poll_interval=0.02)
    with TestClient(app) as c:
        c.settings = s
        yield c


def post(c, data=None, content=b"fake audio", cons=CONS, name="a.wav"):
    d = {"constraints": json.dumps(cons)}
    d.update(data or {})
    return c.post("/api/jobs", data=d, files={"file": (name, content)})


def wait(c, jid):
    with c.stream("GET", f"/api/jobs/{jid}/stream") as r:
        events = []
        ev = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((ev, json.loads(line.split(":", 1)[1])))
    return events


def test_health_config(client):
    h = client.get("/api/health").json()
    assert h["llm_enabled"] is False and "cache_bytes" in h and h["providers"]["basemap"]["enabled"]
    assert client.get("/api/config").json()["attribution"]


def test_job_sse_result_artifacts(client):
    r = post(client)
    assert r.status_code == 201, r.text
    jid = r.json()["id"]
    ev = wait(client, jid)
    assert ev[-1][0] == "end" and ev[-1][1]["status"] == "done"
    progress = [d["progress"] for k, d in ev if k == "progress"]
    assert progress and progress == sorted(progress) and progress[-1] == 1.0
    seqs = [d["seq"] for k, d in ev if k != "end"]
    assert seqs == sorted(seqs)
    stages = [d["stage"] for k, d in ev if k == "progress"]
    assert stages.index("ingest") < stages.index("events")
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "done" and j["has_result"]
    assert client.get(f"/api/jobs/{jid}/result").status_code == 200
    assert any(a.startswith("input") for a in j["artifacts"])
    assert client.get(f"/api/jobs/{jid}/artifacts/{j['artifacts'][0]}").status_code == 200
    assert [x["id"] for x in client.get("/api/jobs").json()] == [jid]


def test_last_event_id_resume(client):
    jid = post(client).json()["id"]
    wait(client, jid)
    with client.stream("GET", f"/api/jobs/{jid}/stream", headers={"Last-Event-ID": "1000000"}) as r:
        body = "".join(r.iter_text())
    assert "event: end" in body and "event: progress" not in body


def test_upload_limit(client):
    r = post(client, content=b"x" * 1001)
    assert r.status_code == 413
    assert client.get("/api/jobs").json() == []
    assert post(client, content=b"x" * 1000).status_code == 201


def test_visual_requires_consent(client):
    assert post(client, {"visual": "true"}).status_code == 409
    assert post(client, {"visual": "true", "visual_consent": "false"}).status_code == 409
    r = post(client, {"visual": "true", "visual_consent": "true"})
    assert r.status_code == 201 and r.json()["params"]["visual_consent"] is True
    assert client.get("/api/jobs").json()[0]["id"] == r.json()["id"]


def test_validation(client):
    assert post(client, cons={"date_start": "2026-09-08"}).status_code == 422  # bbox required
    assert post(client, cons={"bbox": CONS["bbox"]}).status_code == 422  # date required
    bad = client.post("/api/jobs", data={"constraints": "{nope"}, files={"file": ("a.wav", b"x")})
    assert bad.status_code == 422
    assert client.get("/api/jobs/nope").status_code == 404


def test_artifact_path_traversal(client, tmp_path):
    jid = post(client).json()["id"]
    wait(client, jid)
    (client.settings.data / "secret.txt").write_text("secret")
    (client.settings.jobs_dir / "other").mkdir(exist_ok=True)
    (client.settings.jobs_dir / "other" / "x.txt").write_text("other")
    for bad in ("../../secret.txt", "..%2F..%2Fsecret.txt", "%2e%2e/%2e%2e/secret.txt",
                "../other/x.txt", "/etc/passwd", "..%2f..%2f..%2fjobs.db", "missing.txt", "%00"):
        r = client.get(f"/api/jobs/{jid}/artifacts/{bad}")
        assert r.status_code == 404, (bad, r.status_code)
        assert b"secret" not in r.content
    # symlink escaping the workspace is refused
    (client.settings.jobs_dir / jid / "link.txt").symlink_to(client.settings.data / "secret.txt")
    assert client.get(f"/api/jobs/{jid}/artifacts/link.txt").status_code == 404


def test_rerun_with_events_and_delete(client):
    jid = post(client).json()["id"]
    wait(client, jid)
    r = client.post(f"/api/jobs/{jid}/rerun", json={"events": [{"id": "e1"}]})
    assert r.status_code == 201
    j2 = r.json()
    assert j2["id"] != jid and j2["params"]["events"] == [{"id": "e1"}]
    assert wait(client, j2["id"])[-1][1]["status"] == "done"
    r = client.post(f"/api/jobs/{jid}/rerun", json={"visual": True})
    assert r.status_code == 409
    r = client.post(f"/api/jobs/{jid}/rerun", json={"constraints": {"date_start": "2026-09-08"}})
    assert r.status_code == 422
    assert client.delete(f"/api/jobs/{jid}").status_code == 204
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert not (client.settings.jobs_dir / jid).exists()


def test_cancel(tmp_path):
    P.reset_registry()
    s = Settings(data=tmp_path / "data")
    app = create_app(s, stages=three(), start_runner=False, poll_interval=0.02)  # job stays queued
    with TestClient(app) as c:
        jid = post(c).json()["id"]
        assert c.post(f"/api/jobs/{jid}/cancel").json()["status"] == "cancelled"
        assert c.post(f"/api/jobs/{jid}/cancel").status_code == 409
        assert c.get(f"/api/jobs/{jid}/result").status_code == 409
        assert wait(c, jid)[-1][1]["status"] == "cancelled"
