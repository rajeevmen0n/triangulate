"""JobRunner: one in-process worker thread (concurrency 1) that executes queued jobs, plus `run_job`
for synchronous use (CLI, tests)."""

from __future__ import annotations

import queue
import shutil
import threading
import traceback
from pathlib import Path

from ..config import Settings, get_settings
from ..core.cache import file_sha256
from . import store as S
from .pipeline import (
    JobCancelled,
    JobParams,
    Stage,
    StageContext,
    get_stages,
    run_pipeline,
    workspace,
)
from .store import JobStore


def make_store(settings: Settings | None = None) -> JobStore:
    s = settings or get_settings()
    s.ensure_dirs()
    return JobStore(s.db_path)


def create_job(store: JobStore, settings: Settings, params: JobParams, input_file: str | Path) -> str:
    """Register a job and copy the input file into its workspace as input<.ext>."""
    src = Path(input_file)
    jid = store.create(params.model_copy(update={"input_name": "input" + src.suffix}).model_dump(mode="json"))
    ws = workspace(settings, jid)
    ws.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ws / ("input" + src.suffix))
    return jid


def run_job(store: JobStore, settings: Settings, job_id: str, stages: list[Stage] | None = None) -> dict:
    """Run (or resume) a job to completion in the calling thread. Returns the final job row."""
    job = store.get(job_id)
    if job is None:
        raise KeyError(job_id)
    if job["status"] in S.TERMINAL or store.is_cancel_requested(job_id):
        if store.is_cancel_requested(job_id) and job["status"] not in S.TERMINAL:
            store.set_status(job_id, S.CANCELLED)
        return store.get(job_id)  # type: ignore[return-value]
    params = JobParams.model_validate(job["params"])
    ws = workspace(settings, job_id)
    inp = ws / params.input_name
    store.clear_for_restart(job_id)
    store.set_status(job_id, S.RUNNING)
    try:
        ctx = StageContext(
            job_id=job_id, input_path=inp, input_sha256=file_sha256(inp), params=params, settings=settings,
            workdir=ws, cancel_check=lambda: store.is_cancel_requested(job_id),
        )
        def on_progress(st: str, f: float, m: str) -> None:
            store.set_progress(job_id, st, f, m)

        info = run_pipeline(ctx, stages or get_stages(), on_progress)
        store.set_progress(job_id, "done", 1.0, "finished")
        store.set_status(job_id, S.DONE, result_path=info["results_path"])
    except JobCancelled:
        store.set_status(job_id, S.CANCELLED)  # partial results (finished stage outputs) are kept
    except Exception as e:  # noqa: BLE001
        store.log(job_id, "error", message=traceback.format_exc())
        store.set_status(job_id, S.FAILED, error=f"{type(e).__name__}: {e}")
    return store.get(job_id)  # type: ignore[return-value]


class JobRunner:
    """Background thread executing queued jobs one at a time."""

    def __init__(self, store: JobStore, settings: Settings | None = None, stages: list[Stage] | None = None):
        self.store, self.settings, self.stages = store, settings or get_settings(), stages
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> list[str]:
        """Mark jobs orphaned by a previous process as interrupted, queue still-queued jobs, start the thread.
        Returns the interrupted job ids (resume them explicitly with `submit`)."""
        interrupted = self.store.recover_interrupted()
        for j in self.store.list(S.QUEUED):
            self._q.put(j["id"])
        self._thread = threading.Thread(target=self._loop, name="job-runner", daemon=True)
        self._thread.start()
        return interrupted

    def submit(self, job_id: str) -> None:
        """Queue a new job, or resume an interrupted/failed/cancelled one (finished stages are cached)."""
        j = self.store.get(job_id)
        if j and j["status"] in (S.INTERRUPTED, S.FAILED, S.CANCELLED):
            self.store.clear_for_restart(job_id)
            self.store.set_status(job_id, S.QUEUED)
        self._q.put(job_id)

    def _loop(self) -> None:
        while True:
            jid = self._q.get()
            if jid is None:
                return
            try:
                run_job(self.store, self.settings, jid, self.stages)
            except Exception:  # noqa: BLE001
                self.store.log(jid, "error", message=traceback.format_exc())

    def stop(self, timeout: float = 10) -> None:
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout)
