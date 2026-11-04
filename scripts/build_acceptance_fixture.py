"""Build the Oyster Bay acceptance ADS-B region fixture (plan 8.2, "ADS-B data needs").

Builds (or reuses) the adsb.lol day index for 2026-09-08 and writes

    data/cache/fixtures/adsb_2026-09-08_LI_1930-2300.parquet          tracks (all aircraft, jets included)
    data/cache/fixtures/adsb_2026-09-08_LI_1930-2300.aircraft.parquet aircraft table
    data/cache/fixtures/adsb_2026-09-08_LI_1930-2300.manifest.json    sha256 of both files + the query

for bbox (40.50, -74.00, 41.20, -73.00) + 15 km, 19:30-23:00 UTC. The existing adsb cache is reused; the
~3 GB index stream only happens if the index is missing. Usage:

    nix develop --command python scripts/build_acceptance_fixture.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "data" / "cache" / "fixtures"
STEM = "adsb_2026-09-08_LI_1930-2300"
TRACKS = FIXTURE_DIR / f"{STEM}.parquet"
AIRCRAFT = FIXTURE_DIR / f"{STEM}.aircraft.parquet"
MANIFEST = FIXTURE_DIR / f"{STEM}.manifest.json"

BBOX = (40.50, -74.00, 41.20, -73.00)  # S, W, N, E (the user's region)
MARGIN_M = 15_000.0
T0 = datetime(2026, 9, 8, 19, 30, tzinfo=UTC)
T1 = datetime(2026, 9, 8, 23, 0, tzinfo=UTC)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(force: bool = False) -> Path:
    """Return the tracks fixture path, building it first if absent (or `force`)."""
    if TRACKS.exists() and AIRCRAFT.exists() and MANIFEST.exists() and not force:
        return TRACKS
    from triangulate.adsb.adsblol import AdsbLolProvider
    from triangulate.core.geo import bbox_expand_m
    from triangulate.core.types import TrackQuery

    qbox = bbox_expand_m(BBOX, MARGIN_M)
    prov = AdsbLolProvider()
    ts = prov.fetch(TrackQuery(bbox=qbox, t0=T0, t1=T1, include_ground=False),
                    lambda s, f, m="": print(f"[{f * 100:5.1f}%] {s} {m}", file=sys.stderr))
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    ts.tracks.write_parquet(TRACKS, compression="zstd")
    ts.aircraft.write_parquet(AIRCRAFT, compression="zstd")
    MANIFEST.write_text(json.dumps({
        "provider": ts.provider, "licence": ts.licence,
        "query": {"bbox_user": list(BBOX), "margin_m": MARGIN_M, "bbox_query": list(qbox),
                  "t0": T0.isoformat(), "t1": T1.isoformat(), "include_ground": False,
                  "jets_included": True},
        "built_utc": datetime.now(UTC).isoformat(),
        "rows": ts.tracks.height, "aircraft": ts.aircraft.height,
        "files": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in (TRACKS, AIRCRAFT)},
    }, indent=2))
    return TRACKS


def load_trackset():
    """The fixture as a TrackSet (builds it if missing)."""
    from triangulate.core.types import TrackSet

    build()
    return TrackSet(tracks=pl.read_parquet(TRACKS), aircraft=pl.read_parquet(AIRCRAFT),
                    provider="adsblol", licence=json.loads(MANIFEST.read_text())["licence"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if the fixture exists")
    a = ap.parse_args()
    p = build(a.force)
    m = json.loads(MANIFEST.read_text())
    print(json.dumps({"fixture": str(p), "rows": m["rows"], "aircraft": m["aircraft"],
                      "files": m["files"]}, indent=2))
