"""SQLite job store (plan 1.4): `jobs` plus an append-only `job_log`. One short-lived connection per call,
so it is safe to use from the runner thread and API threads at once."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

QUEUED, RUNNING, DONE, FAILED, CANCELLED, INTERRUPTED = (
    "queued", "running", "done", "failed", "cancelled", "interrupted",
)
TERMINAL = {DONE, FAILED, CANCELLED}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL, stage TEXT,
  progress REAL NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '', params_json TEXT NOT NULL, error TEXT, result_path TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS job_log(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL,
  stage TEXT, progress REAL, message TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS job_log_job ON job_log(job_id, seq);
"""


class JobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with closing(self._conn()) as c:
            return c.execute(sql, args).fetchall()

    @staticmethod
    def _job(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["params"] = json.loads(d.pop("params_json"))
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d

    # -- jobs
    def create(self, params: dict[str, Any], job_id: str | None = None) -> str:
        jid = job_id or uuid.uuid4().hex[:12]
        self._exec(
            "INSERT INTO jobs(id, created, status, params_json) VALUES(?,?,?,?)",
            (jid, time.time(), QUEUED, json.dumps(params, default=str)),
        )
        self.log(jid, "status", message=QUEUED)
        return jid

    def get(self, job_id: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._job(rows[0]) if rows else None

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._exec("SELECT * FROM jobs WHERE status=? ORDER BY created", (status,))
        else:
            rows = self._exec("SELECT * FROM jobs ORDER BY created")
        return [self._job(r) for r in rows]

    def count_since(self, since: float) -> int:
        """Number of jobs created at or after `since` (epoch seconds); used for the per-day rate limit."""
        return int(self._exec("SELECT COUNT(*) FROM jobs WHERE created>=?", (since,))[0][0])

    def delete(self, job_id: str) -> None:
        self._exec("DELETE FROM job_log WHERE job_id=?", (job_id,))
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))

    def set_status(
        self, job_id: str, status: str, error: str | None = None, result_path: str | None = None
    ) -> None:
        self._exec(
            "UPDATE jobs SET status=?, error=COALESCE(?, error),"
            " result_path=COALESCE(?, result_path) WHERE id=?",
            (status, error, result_path, job_id),
        )
        self.log(job_id, "status", message=status if error is None else f"{status}: {error}")

    def set_progress(self, job_id: str, stage: str, progress: float, message: str = "") -> None:
        self._exec(
            "UPDATE jobs SET stage=?, progress=?, message=? WHERE id=?", (stage, progress, message, job_id)
        )
        self.log(job_id, "progress", stage, progress, message)

    def clear_for_restart(self, job_id: str) -> None:
        self._exec("UPDATE jobs SET error=NULL, cancel_requested=0 WHERE id=?", (job_id,))

    # -- cancel
    def request_cancel(self, job_id: str) -> bool:
        """Flag a job for cancellation. A queued job is cancelled immediately."""
        j = self.get(job_id)
        if j is None or j["status"] in TERMINAL:
            return False
        self._exec("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
        if j["status"] in (QUEUED, INTERRUPTED):
            self.set_status(job_id, CANCELLED)
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        rows = self._exec("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,))
        return bool(rows and rows[0][0])

    # -- log
    def log(self, job_id: str, kind: str, stage: str | None = None, progress: float | None = None,
            message: str = "") -> None:
        self._exec(
            "INSERT INTO job_log(job_id, ts, kind, stage, progress, message) VALUES(?,?,?,?,?,?)",
            (job_id, time.time(), kind, stage, progress, message),
        )

    def log_since(self, job_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        rows = self._exec("SELECT * FROM job_log WHERE job_id=? AND seq>? ORDER BY seq", (job_id, after_seq))
        return [dict(r) for r in rows]

    # -- restart recovery
    def recover_interrupted(self) -> list[str]:
        """Call once at process start: jobs left `running` by a dead process become `interrupted`
        (resumable: stage outputs are cached, so a resume skips finished stages)."""
        ids = [r["id"] for r in self._exec("SELECT id FROM jobs WHERE status=?", (RUNNING,))]
        for i in ids:
            self.set_status(i, INTERRUPTED, error="process restarted while job was running")
        return ids
