"""Retention purge (plan section 7): job workspaces, job rows and stale upload staging older than
`retention_days` are deleted. Runs on API startup and via `triangulate purge`.

Safety rules:
- Only entries directly inside `data/jobs` (workspaces, named by a tracked job id) and `data/tmp`
  (upload staging) are ever removed. Names containing separators, `.` or `..` are refused.
- Symlinks are never followed and never removed. A symlink in place of a workspace is skipped.
- `data/cache` (and anything else outside `data/`) is never touched.
- Jobs that are queued or running are never purged.
- Untracked entries in `data/jobs` (no job row) are left alone.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from . import store as S
from .store import JobStore

DAY = 86400.0
_KEEP_STATUSES = {S.QUEUED, S.RUNNING}


def _within(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _entry(root: Path, name: str) -> Path | None:
    """`root/name` if `name` is a single safe path component and the entry is not a symlink; else None.
    The entry need not exist. The caller checks existence."""
    if not name or name in (".", "..") or any(c in name for c in ("/", "\\", "\x00")):
        return None
    p = root / name
    if p.is_symlink():
        return None
    if p.resolve().parent != root.resolve():
        return None
    return p


def _root_ok(s: Settings, root: Path) -> bool:
    """The purge root must lie inside data/ and must not be, or contain, the stage cache."""
    r, data, cache = root.resolve(), s.data.resolve(), s.cache_dir.resolve()
    return _within(r, data) and r != data and not (r == cache or _within(r, cache) or _within(cache, r))


def _remove(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)  # does not follow symlinks inside the tree
    else:
        p.unlink()


def purge_expired(
    s: Settings, store: JobStore, days: float | None = None, *, dry_run: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Delete jobs created more than `days` ago (default `s.retention_days`) and stale upload staging.
    `days <= 0` disables the purge. Returns a JSON-able report; per-item failures go to `errors`."""
    days = s.retention_days if days is None else days
    report: dict[str, Any] = {
        "enabled": days > 0, "retention_days": days, "dry_run": dry_run,
        "jobs_purged": [], "tmp_removed": [], "skipped": [], "errors": [],
    }
    if days <= 0:
        return report
    cutoff = (time.time() if now is None else now) - days * DAY
    report["cutoff_epoch"] = cutoff
    jobs_root, tmp_root = s.jobs_dir, s.data / "tmp"

    if jobs_root.is_dir() and _root_ok(s, jobs_root):
        for j in store.list():
            jid = j["id"]
            if j["created"] >= cutoff:
                continue
            if j["status"] in _KEEP_STATUSES:
                report["skipped"].append({"job": jid, "reason": f"status {j['status']}"})
                continue
            ws = _entry(jobs_root, jid)
            if ws is None:
                report["skipped"].append({"job": jid, "reason": "unsafe workspace name or symlink"})
                continue
            try:
                if ws.exists() and not dry_run:
                    _remove(ws)
                if not dry_run:
                    store.delete(jid)
                report["jobs_purged"].append(jid)
            except OSError as e:
                report["errors"].append({"job": jid, "error": f"{type(e).__name__}: {e}"})
    else:
        report["skipped"].append({"root": str(jobs_root), "reason": "jobs directory unavailable or unsafe"})

    if tmp_root.is_dir() and _root_ok(s, tmp_root):
        for child in tmp_root.iterdir():
            p = _entry(tmp_root, child.name)
            if p is None:
                report["skipped"].append({"tmp": child.name, "reason": "symlink or unsafe name"})
                continue
            try:
                if p.lstat().st_mtime >= cutoff:
                    continue
                if not dry_run:
                    _remove(p)
                report["tmp_removed"].append(child.name)
            except OSError as e:
                report["errors"].append({"tmp": child.name, "error": f"{type(e).__name__}: {e}"})
    return report
