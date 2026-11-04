"""ICAO type designator -> description code (ICAO Doc 8643) and jet exclusion.

Sources, in order of precedence:
1. `data/cache/types/icao_aircraft_types2.js` (wiedehopf/tar1090-db), only if present. It is
   fetched by `download_types()` (explicit, opt-in) for local use and never redistributed.
2. The bundled `doc8643/icao_doc8643_fallback.csv`: designator facts from ICAO Doc 8643,
   transcribed by hand (a few hundred common types).

Licence finding (checked 2026-11): tar1090-db has no LICENSE file and GitHub reports no licence;
its README says the data comes from the Mictronics database. Terms are therefore UNCLEAR, so the
default path never downloads or ships it (R3 in the plan); the manifest records this.

Engine code (3rd character of the description code): J jet, T turboprop/turboshaft, P piston,
E electric, R rocket. Class (1st character): L landplane, S seaplane, A amphibian, H helicopter,
G gyrocopter, T tilt-rotor.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings

TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js"
TYPES_FILE = "icao_aircraft_types2.js"
FALLBACK_CSV = Path(__file__).parent / "doc8643" / "icao_doc8643_fallback.csv"
LICENCE_STATUS = "unclear"
LICENCE_NOTE = (
    "wiedehopf/tar1090-db has no LICENSE file (GitHub API: license=null); data derives from the "
    "Mictronics aircraft database and ICAO Doc 8643. Not redistributed; downloaded on explicit "
    "request for local use only. Bundled fallback is hand-transcribed Doc 8643 designators."
)
JET_ENGINE = "J"
FALLBACK_JET_CATEGORIES = frozenset({"A3", "A4", "A5"})
FALLBACK_GS_KT = 250.0

KEEP, DROP = True, False


@dataclass(frozen=True)
class TypeInfo:
    typecode: str
    desc_code: str  # e.g. "L2J"
    description: str = ""

    @property
    def klass(self) -> str:
        return self.desc_code[:1]

    @property
    def engine(self) -> str:
        return self.desc_code[2:3]

    @property
    def is_helicopter(self) -> bool:
        return self.klass == "H"


class AircraftTypes:
    def __init__(self, table: dict[str, TypeInfo], source: str = "", manifest: dict[str, Any] | None = None):
        self.table = table
        self.source = source
        self.manifest = manifest or {}

    def __len__(self) -> int:
        return len(self.table)

    def lookup(self, typecode: str | None) -> TypeInfo | None:
        if not typecode:
            return None
        return self.table.get(typecode.strip().upper())

    def desc_code(self, typecode: str | None) -> str | None:
        ti = self.lookup(typecode)
        return ti.desc_code if ti else None

    def engine(self, typecode: str | None) -> str | None:
        ti = self.lookup(typecode)
        return ti.engine or None if ti else None

    def classify(
        self,
        typecode: str | None,
        category: str | None = None,
        max_gs_kt: float | None = None,
        exclude_helicopters: bool = False,
    ) -> tuple[bool, str]:
        """Decide keep/drop for one aircraft. Returns (keep, reason)."""
        ti = self.lookup(typecode)
        if ti is not None:
            if ti.engine == JET_ENGINE:
                return DROP, f"jet:{ti.desc_code}"
            if ti.is_helicopter:
                return (not exclude_helicopters), f"helicopter:{ti.desc_code}"
            return KEEP, f"type:{ti.desc_code}"
        cat = (category or "").strip().upper()
        if cat in FALLBACK_JET_CATEGORIES and max_gs_kt is not None and max_gs_kt > FALLBACK_GS_KT:
            return DROP, f"no-type:{cat}@{max_gs_kt:.0f}kt"
        if cat == "A7":
            return (DROP, "no-type:A7") if exclude_helicopters else (KEEP, "no-type:A7")
        return KEEP, "unknown"


def _read_fallback(path: Path = FALLBACK_CSV) -> dict[str, TypeInfo]:
    out: dict[str, TypeInfo] = {}
    with path.open(newline="") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for r in rows:
            out[r["typecode"].upper()] = TypeInfo(r["typecode"].upper(), r["desc_code"].strip().upper())
    return out


def _parse_tar1090(raw: bytes) -> dict[str, TypeInfo]:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    doc = json.loads(raw)
    out: dict[str, TypeInfo] = {}
    for code, v in doc.items():
        if isinstance(v, list) and len(v) >= 2 and isinstance(v[1], str) and v[1]:
            out[code.upper()] = TypeInfo(code.upper(), v[1].upper(), str(v[0]))
    return out


def types_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).cache_subdir("types")


def download_types(settings: Settings | None = None, client: Any = None) -> Path:
    """Fetch tar1090-db's type table into the cache and write the manifest (opt-in; see licence)."""
    import httpx

    s = settings or get_settings()
    d = types_dir(s)
    d.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": s.user_agent}, follow_redirects=True, timeout=60.0)
    try:
        r = client.get(TYPES_URL)
        r.raise_for_status()
        raw = r.content
    finally:
        if own:
            client.close()
    table = _parse_tar1090(raw)  # validates before writing
    path = d / TYPES_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    manifest = {
        "source": "tar1090-db", "url": TYPES_URL, "sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), "n_types": len(table),
        "licence_status": LICENCE_STATUS, "licence_note": LICENCE_NOTE,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return path


def load_types(settings: Settings | None = None, download: bool = False) -> AircraftTypes:
    """Load the fallback table, overlaid by the cached tar1090-db table when present."""
    s = settings or get_settings()
    path = types_dir(s) / TYPES_FILE
    if download and not path.exists():
        download_types(s)
    table = _read_fallback()
    source = "doc8643-fallback"
    manifest: dict[str, Any] = {"source": source, "licence_status": "own transcription of public designators",
                                "n_types": len(table)}
    if path.exists():
        try:
            table.update(_parse_tar1090(path.read_bytes()))
            source = "tar1090-db+fallback"
            mp = path.parent / "manifest.json"
            manifest = json.loads(mp.read_text()) if mp.exists() else {
                "licence_status": LICENCE_STATUS, "licence_note": LICENCE_NOTE}
            manifest["source"] = source
            manifest["n_types"] = len(table)
        except (ValueError, OSError):
            pass  # corrupt cache: stay on the fallback
    return AircraftTypes(table, source, manifest)
