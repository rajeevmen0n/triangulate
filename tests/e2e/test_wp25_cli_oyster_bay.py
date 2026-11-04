"""WP25: `triangulate run` on audio.mp3 through the real stages. Reports the top candidate's distance
from the ground truth; asserts only loosely (WP23 owns the acceptance thresholds). Container metadata of
audio.mp3 is never read (`--no-metadata`; only samples are decoded)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from triangulate.core.geo import haversine_m
from triangulate.jobs.stages import validate_results

pytestmark = [pytest.mark.network, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
GT = (40.876722, -73.526444)


def test_cli_run_audio_mp3():
    audio = ROOT / "audio.mp3"
    if not audio.exists():
        pytest.skip("audio.mp3 not present")
    t = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-m", "triangulate", "run", str(audio), "--date", "2026-09-08",
         "--bbox", "40.58,-73.83,41.18,-73.23", "--time-window", "19:30-22:30", "--tz", "UTC",
         "--no-metadata", "--quiet"],
        capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - t
    assert p.returncode == 0, p.stderr[-2000:]
    summary = json.loads(p.stdout)
    res = json.loads(Path(summary["result_path"]).read_text())
    validate_results(res)
    assert res["candidates"]
    top = res["best"]
    dist = haversine_m(top["lat"], top["lon"], *GT)
    print(f"\nWP25 CLI: top candidate {top['lat']:.5f},{top['lon']:.5f}  distance {dist:.0f} m  "
          f"flights {top['flights']}  T_s {top['t_s']}  margin {top['margin_nats']}  runtime {elapsed:.0f} s")
    assert res["events"]["n_used"] >= 1
