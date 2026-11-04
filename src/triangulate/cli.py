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
            constraints=cons, visual=args.visual, visual_consent=args.visual, events=events
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


def _not_yet(name: str, wp: str):
    def run(args: argparse.Namespace) -> int:
        print(f"`{name}` is not implemented yet ({wp}).", file=sys.stderr)
        return 2

    return run


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
    run.add_argument("--events", help="events.json to use instead of auto-detection")
    run.set_defaults(func=_cmd_run)

    sub.add_parser("serve", help="start the web UI/API (WP19)").set_defaults(
        func=_not_yet("serve", "WP19")
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)
