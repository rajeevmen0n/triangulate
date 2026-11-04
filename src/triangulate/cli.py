"""Command line entry point. `run`/`serve` are implemented by WP18/WP19."""

from __future__ import annotations

import argparse
import json
import sys

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
    run.add_argument("--date", required=True, help="recording date, YYYY-MM-DD (required)")
    run.add_argument("--bbox", required=True, help="region S,W,N,E in degrees (required)")
    run.add_argument("--time-window", help="local or UTC window, e.g. 19:30-22:30")
    run.add_argument("--tz", help="IANA timezone")
    run.add_argument("--visual", action="store_true", help="opt in to the Claude visual module")
    run.add_argument("--events", help="events.json to use instead of auto-detection")
    run.set_defaults(func=_not_yet("run", "WP18"))

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
