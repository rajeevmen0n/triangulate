import json
import threading

import pytest

from triangulate.cli import main
from triangulate.config import Settings
from triangulate.jobs import pipeline as P
from triangulate.jobs import store as S
from triangulate.jobs.runner import JobRunner, create_job, make_store, run_job

CONS = {"bbox": (40.0, -74.0, 41.0, -73.0), "date_start": "2026-09-08"}


@pytest.fixture
def env(tmp_path):
    P.reset_registry()
    s = Settings(data=tmp_path / "data")
    f = tmp_path / "a.wav"
    f.write_bytes(b"fake audio")
    return s, make_store(s), f


class Counting(P.StubStage):
    calls: list[str] = []

    def run(self, ctx, out_dir, progress):
        Counting.calls.append(self.name)
        return super().run(ctx, out_dir, progress)


def three(names=("ingest", "spectrogram", "events")):
    deps = {"ingest": (), "spectrogram": ("ingest",), "events": ("spectrogram",)}
    return [Counting(n, deps[n]) for n in names]


def test_progress_monotonic_and_cache_resume(env):
    s, st, f = env
    Counting.calls = []
    jid = create_job(st, s, P.JobParams(constraints=CONS), f)
    j = run_job(st, s, jid, three())
    assert j["status"] == "done" and j["progress"] == 1.0
    prog = [r["progress"] for r in st.log_since(jid) if r["kind"] == "progress"]
    assert prog == sorted(prog) and prog[-1] == 1.0
    assert (s.jobs_dir / jid / "results.json").exists()
    assert Counting.calls == ["ingest", "spectrogram", "events"]
    # same input + constraints in a new job: everything cached
    jid2 = create_job(st, s, P.JobParams(constraints=CONS), f)
    run_job(st, s, jid2, three())
    assert Counting.calls == ["ingest", "spectrogram", "events"]
    assert (s.jobs_dir / jid2 / "events.json").exists()
    # changed constraints invalidate
    c2 = dict(CONS, tz="America/New_York")
    jid3 = create_job(st, s, P.JobParams(constraints=c2), f)
    run_job(st, s, jid3, three())
    assert len(Counting.calls) == 5  # ingest stays cached


def test_cancel_keeps_partial(env):
    s, st, f = env
    jid = create_job(st, s, P.JobParams(constraints=CONS), f)

    class Canceller(Counting):
        def run(self, ctx, out_dir, progress):
            st.request_cancel(ctx.job_id)
            return super().run(ctx, out_dir, progress)

    stages = [Counting("ingest"), Canceller("spectrogram", ("ingest",)), Counting("events", ("spectrogram",))]
    Counting.calls = []
    j = run_job(st, s, jid, stages)
    assert j["status"] == S.CANCELLED
    assert (s.jobs_dir / jid / "ingest.json").exists()
    assert "events" not in Counting.calls


def test_failure_then_resume_after_restart(env):
    s, st, f = env
    Counting.calls = []
    jid = create_job(st, s, P.JobParams(constraints=CONS), f)

    class Boom(Counting):
        def run(self, ctx, out_dir, progress):
            raise RuntimeError("boom")

    j = run_job(st, s, jid, [Counting("ingest"), Boom("spectrogram", ("ingest",))])
    assert j["status"] == S.FAILED and "boom" in j["error"]
    # simulate a crash mid-run, then a new process
    st.set_status(jid, S.RUNNING)
    st2 = make_store(s)
    assert st2.recover_interrupted() == [jid]
    assert st2.get(jid)["status"] == S.INTERRUPTED
    r = JobRunner(st2, s, three())
    r.start()
    r.submit(jid)
    for _ in range(100):
        if st2.get(jid)["status"] == S.DONE:
            break
        threading.Event().wait(0.05)
    r.stop()
    assert st2.get(jid)["status"] == S.DONE
    assert Counting.calls.count("ingest") == 1  # resumed from cache


def test_visual_skipped_without_consent_or_key(env):
    s, st, f = env
    jid = create_job(st, s, P.JobParams(constraints=CONS, visual=True), f)
    j = run_job(st, s, jid)
    assert j["status"] == "done"
    assert json.loads((s.jobs_dir / jid / "results.json").read_text())["stages"]["visual"]["skipped"]
    big = P.JobParams(constraints=CONS, llm_budget_usd=50)
    assert P.StageContext(jid, f, "x", big, s, f).llm_budget_usd == 2.0


def test_cli_run_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRIANGULATE_DATA", str(tmp_path / "d"))
    from triangulate.config import reload_settings
    reload_settings()
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    rc = main(["run", str(f), "--date", "2026-09-08", "--bbox", "40,-74,41,-73", "--time-window",
               "19:30-22:30", "--quiet"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["status"] == "done"
    assert main(["run", str(f), "--date", "2026-09-08", "--bbox", "bad"]) == 2
    with pytest.raises(SystemExit):
        main(["run", str(f), "--bbox", "40,-74,41,-73"])
    reload_settings()
