"""Command line entry point. `run`/`serve` are implemented by WP18/WP19."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import time
from pathlib import Path

from . import __version__
from .config import get_settings


def _cmd_info(args: argparse.Namespace) -> int:
    s = get_settings()
    print(
        json.dumps(
            {
                "version": __version__,
                "data_dir": str(s.data),
                "llm": "enabled" if s.llm_enabled else "disabled",
                "user_agent": s.user_agent,
                "basemap_enabled": s.basemap_enabled,
            },
            indent=2,
        )
    )
    return 0


def _parse_window(w: str) -> tuple[time, time]:
    a, _, b = w.partition("-")
    return time.fromisoformat(a.strip()), time.fromisoformat(b.strip())


def _cmd_run(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from .jobs.pipeline import JobParams
    from .jobs.runner import create_job, make_store, run_job
    from .jobs.stages import register_real_stages

    register_real_stages()
    path = Path(args.file)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    try:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        d0, _, d1 = args.date.partition("..")
        cons: dict = {"bbox": bbox, "date_start": d0, "date_end": d1 or None, "tz": args.tz}
        if args.time_window:
            cons["window_start"], cons["window_end"] = _parse_window(args.time_window)
        events = json.loads(Path(args.events).read_text()) if args.events else None
        if isinstance(events, dict):
            events = events.get("events", [])
        params = JobParams(
            constraints=cons, visual=args.visual, visual_consent=args.visual, events=events,
            use_metadata=not args.no_metadata, trust_gps=args.trust_gps,
        )
    except (ValueError, ValidationError, OSError) as e:
        print(f"error: invalid arguments: {e}", file=sys.stderr)
        return 2
    s = get_settings()
    if args.visual:
        key = "" if s.llm_enabled else " ANTHROPIC_API_KEY is not set, so the visual module is skipped."
        print(
            f"NOTICE: --visual sends video frames to the Claude API (budget cap ${s.llm_budget_usd:.2f}/job)."
            + key,
            file=sys.stderr,
        )
    store = make_store(s)
    jid = create_job(store, s, params, path)
    if not args.quiet:
        orig = store.set_progress

        def show(job_id, stage, progress, message=""):
            orig(job_id, stage, progress, message)
            print(f"[{progress * 100:5.1f}%] {stage}: {message}", file=sys.stderr)

        store.set_progress = show  # type: ignore[method-assign]
    job = run_job(store, s, jid)
    summary = {
        "job_id": jid, "status": job["status"], "error": job["error"], "progress": job["progress"],
        "workspace": str(s.jobs_dir / jid), "result_path": job["result_path"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if job["status"] == "done" else 1


def _cmd_purge(args: argparse.Namespace) -> int:
    """Delete job workspaces and upload staging older than the retention period (WP24)."""
    from .jobs.retention import purge_expired
    from .jobs.runner import make_store

    if args.days is not None and args.days < 1:
        print("error: --days must be at least 1 (use TRIANGULATE_RETENTION_DAYS=0 to disable the purge)",
              file=sys.stderr)
        return 2
    s = get_settings()
    report = purge_expired(s, make_store(s), args.days, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, default=str))
    return 1 if report["errors"] else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app
    from .jobs.stages import register_real_stages

    register_real_stages()
    s = get_settings()
    host, port = args.host or s.host, args.port or s.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: {host} exposes this unauthenticated API beyond this machine.", file=sys.stderr)
    uvicorn.run(create_app(s), host=host, port=port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triangulate", description="Estimate where an audio/video recording was made."
    )
    p.add_argument("--version", action="version", version=f"triangulate {__version__}")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("info", help="show effective settings (no secrets)").set_defaults(func=_cmd_info)

    run = sub.add_parser("run", help="run the pipeline synchronously on a file (WP18)")
    run.add_argument("file")
    run.add_argument("--date", required=True, help="recording date YYYY-MM-DD, or START..END (required)")
    run.add_argument("--bbox", required=True, help="region S,W,N,E in degrees (required)")
    run.add_argument("--time-window", help="local or UTC window, e.g. 19:30-22:30")
    run.add_argument("--tz", help="IANA timezone")
    run.add_argument("--visual", action="store_true", help="opt in to the Claude visual module")
    run.add_argument("--quiet", action="store_true", help="no progress on stderr")
    run.add_argument("--no-metadata", action="store_true", help="do not read container metadata")
    run.add_argument("--trust-gps", action="store_true", help="pin the file's GPS (if any) as candidate #0")
    run.add_argument("--events", help="events.json to use instead of auto-detection")
    run.set_defaults(func=_cmd_run)

    pg = sub.add_parser("purge", help="delete job data older than the retention period")
    pg.add_argument("--days", type=float, help="override TRIANGULATE_RETENTION_DAYS (>= 1)")
    pg.add_argument("--dry-run", action="store_true", help="report what would be deleted")
    pg.set_defaults(func=_cmd_purge)

    sv = sub.add_parser("serve", help="start the web UI/API (binds to 127.0.0.1 by default)")
    sv.add_argument("--host", help="bind address (default: settings.host, 127.0.0.1)")
    sv.add_argument("--port", type=int, help="port (default: settings.port, 8000)")
    sv.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)
