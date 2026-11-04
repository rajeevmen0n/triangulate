# ruff: noqa: E501
"""Decoding of readsb/adsb.lol `trace_full_<icao>.json` members (gzip bytes despite the name).

Row layout (readsb README-json, trace rows):
  [dt_s, lat, lon, alt_baro_ft | "ground" | None, gs_kt, track_deg, flags, vrate_fpm,
   detail{...} | None, source, alt_geom_ft, geom_rate_fpm, ias_kt, roll_deg]
Absolute time = header["timestamp"] + dt_s. Rows may be shorter than 14 entries.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter
from typing import Any

import numpy as np
import orjson
import polars as pl

from ..core.types import AIRCRAFT_SCHEMA, TRACK_SCHEMA

# `flags` bits (readsb README-json)
FLAG_STALE = 1  # position is stale (older than the 'recent' window)
FLAG_NEW_LEG = 2  # start of a new leg (gap/landing before it)
FLAG_GEOM_VRATE = 4  # the vertical rate column is geometric (else barometric)
FLAG_GEOM_ALT = 8  # altitude column is geometric (else barometric)

BUCKET_S = 1800  # coarse index: 1 deg x 1 deg tiles x 30 min buckets
N_TILES = 180 * 360
_B_MIN, _B_MAX = -1, 48


def parse_trace(raw: bytes) -> dict[str, Any]:
    """Gunzip (when gzip magic is present) and parse one member."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return orjson.loads(raw)


def _col(rows: list[list[Any]], i: int) -> list[Any]:
    return [r[i] if len(r) > i else None for r in rows]


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def aircraft_row(doc: dict[str, Any], category: str | None = None) -> dict[str, Any]:
    return {
        "icao": str(doc.get("icao", "")).lower(),
        "reg": doc.get("r") or None,
        "typecode": doc.get("t") or None,
        "desc": doc.get("desc") or None,
        "category": category,
        "db_flags": int(doc["dbFlags"]) if doc.get("dbFlags") is not None else None,
        "owner_op": doc.get("ownOp") or None,
        "year": str(doc["year"]) if doc.get("year") not in (None, "") else None,
    }


def trace_frame(doc: dict[str, Any], with_flags: bool = False) -> pl.DataFrame:
    """Decode all rows into the TRACK_SCHEMA columns (plus `flags` if requested)."""
    rows: list[list[Any]] = doc.get("trace") or []
    base = float(doc.get("timestamp", 0))
    icao = str(doc.get("icao", "")).lower()
    n = len(rows)
    alt = _col(rows, 3)
    details = _col(rows, 8)
    cats: list[str | None] = []
    cur: str | None = None
    for d in details:
        if isinstance(d, dict) and d.get("category"):
            cur = d["category"]
        cats.append(cur)
    vr = _col(rows, 7)
    vr2 = _col(rows, 11)
    data: dict[str, Any] = {
        "icao": [icao] * n,
        "reg": [doc.get("r") or None] * n,
        "typecode": [doc.get("t") or None] * n,
        "category": cats,
        "t": [base + float(r[0]) for r in rows],
        "lat": [_num(v) for v in _col(rows, 1)],
        "lon": [_num(v) for v in _col(rows, 2)],
        "alt_baro_ft": [None if a == "ground" else _num(a) for a in alt],
        "alt_geom_ft": [_num(v) for v in _col(rows, 10)],
        "gs_kt": [_num(v) for v in _col(rows, 4)],
        "track_deg": [_num(v) for v in _col(rows, 5)],
        "vrate_fpm": [_num(a) if _num(a) is not None else _num(b) for a, b in zip(vr, vr2, strict=True)],
        "on_ground": [a == "ground" for a in alt],
        "source": [s if isinstance(s, str) else None for s in _col(rows, 9)],
    }
    df = pl.DataFrame(data, schema=TRACK_SCHEMA)
    if with_flags:
        df = df.with_columns(pl.Series("flags", [int(f) if isinstance(f, int) else 0 for f in _col(rows, 6)], dtype=pl.Int64))
    return df


def decode_trace(raw: bytes, with_flags: bool = False) -> tuple[dict[str, Any], pl.DataFrame]:
    """Decode a member into (aircraft dict following AIRCRAFT_SCHEMA, track frame)."""
    doc = parse_trace(raw)
    df = trace_frame(doc, with_flags=with_flags)
    cat = _mode([c for c in df["category"].to_list() if c])
    ac = aircraft_row(doc, cat)
    assert set(ac) == set(AIRCRAFT_SCHEMA)
    return ac, df


def _mode(vals: list[str]) -> str | None:
    return Counter(vals).most_common(1)[0][0] if vals else None


# --- coarse cell index -------------------------------------------------------------------------


def cells_for(lat: np.ndarray, lon: np.ndarray, t: np.ndarray, day_start: float) -> np.ndarray:
    """int32 cell ids = (bucket+1) * N_TILES + tile for points (all arrays finite)."""
    flat = np.clip(np.floor(lat), -90, 89).astype(np.int64)
    flon = np.clip(np.floor(lon), -180, 179).astype(np.int64)
    tile = (flat + 90) * 360 + (flon + 180)
    b = np.clip(np.floor((t - day_start) / BUCKET_S), _B_MIN, _B_MAX).astype(np.int64)
    return ((b + 1) * N_TILES + tile).astype(np.int32)


def query_cells(bbox: tuple[float, float, float, float], t0: float, t1: float, day_start: float) -> np.ndarray:
    """All cell ids a query bbox x [t0, t1] (epoch s) can touch within one UTC day."""
    s, w, n, e = bbox
    lats = np.arange(math.floor(s), math.floor(n) + 1)
    lons = np.arange(math.floor(w), math.floor(e) + 1)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    tile = (np.clip(la, -90, 89).ravel() + 90) * 360 + (np.clip(lo, -180, 179).ravel() + 180)
    b0 = int(np.clip(math.floor((t0 - day_start) / BUCKET_S), _B_MIN, _B_MAX))
    b1 = int(np.clip(math.floor((t1 - day_start) / BUCKET_S), _B_MIN, _B_MAX))
    bs = np.arange(b0, b1 + 1)
    return ((bs[:, None] + 1) * N_TILES + tile[None, :]).ravel().astype(np.int32)


def summarize_trace(raw: bytes, day_start: float, qcells: np.ndarray | None = None) -> dict[str, Any] | None:
    """Index summary of one member (fast path, runs in worker processes).

    Returns None if the member is unparsable or has no positions.
    """
    try:
        doc = parse_trace(raw)
        rows = doc.get("trace") or []
        if not rows:
            return None
        base = float(doc.get("timestamp", day_start))
        try:
            dt = np.array([r[0] for r in rows], dtype=np.float64)
            lat = np.array([r[1] for r in rows], dtype=np.float64)
            lon = np.array([r[2] for r in rows], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            dt = np.array([_num(r[0]) if r else None for r in rows], dtype=np.float64)
            lat = np.array([_num(r[1]) if len(r) > 1 else None for r in rows], dtype=np.float64)
            lon = np.array([_num(r[2]) if len(r) > 2 else None for r in rows], dtype=np.float64)
        ok = np.isfinite(dt) & np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            return None
        t = base + dt[ok]
        la, lo = lat[ok], lon[ok]
        cells = np.unique(cells_for(la, lo, t, day_start))
        cats = [d["category"] for r in rows if len(r) > 8 and isinstance((d := r[8]), dict) and d.get("category")]
        ac = aircraft_row(doc, _mode(cats))
        ac.update(
            tmin=float(t.min()), tmax=float(t.max()),
            lat_min=float(la.min()), lat_max=float(la.max()),
            lon_min=float(lo.min()), lon_max=float(lo.max()),
            n_points=int(ok.sum()), cells=cells.tolist(), version=doc.get("version"),
            hit=bool(qcells is not None and np.isin(cells, qcells).any()),
        )
        return ac
    except Exception:  # noqa: BLE001 - corrupt member: skip, caller counts it
        return None
