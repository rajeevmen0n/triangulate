"""Aircraft gating (jet exclusion) and 1-s ENU track preparation (plan 3.3.2).

Public API
- `classify_aircraft(trackset, types, exclude_helicopters)` -> per-aircraft keep/drop table.
- `exclude_jets(trackset, types, exclude_helicopters)` -> TrackSet without dropped aircraft.
- `prepare_tracks(tracks, origin, ...)` -> 1-s ENU DataFrame (`PREPARED_SCHEMA`).
- `prepare_trackset(trackset, origin, types, ...)` -> (prepared frame, classification table).

The readsb `flags & 2` new-leg bit is not carried in TRACK_SCHEMA, so legs are split on gaps
> `max_gap_s` only (provider-side bbox filtering already creates gaps when a flight leaves it).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.interpolate import PchipInterpolator

from ..core.geo import geodetic_to_enu
from ..core.types import TrackSet
from .aircraft_types import AircraftTypes, load_types

FT_M = 0.3048
KT_MS = 1852.0 / 3600.0
SIGMA_GEOM_M = 10.0
SIGMA_BARO_M = 100.0  # pressure altitude is not QNH-corrected
MAX_GAP_S = 90.0

PREPARED_SCHEMA: dict[str, type[pl.DataType]] = {
    "icao": pl.String, "seg": pl.Int64, "t": pl.Float64,
    "e": pl.Float64, "n": pl.Float64, "u": pl.Float64,  # ENU metres, u = height above ellipsoid
    "gs_ms": pl.Float64, "alt_sigma_m": pl.Float64, "alt_src": pl.String,  # "geom" | "baro"
}


def classify_aircraft(
    ts: TrackSet, types: AircraftTypes | None = None, exclude_helicopters: bool = False
) -> pl.DataFrame:
    """One row per aircraft: icao, typecode, category, desc_code, max_gs_kt, keep, reason."""
    types = types or load_types()
    if ts.tracks.height == 0:
        return pl.DataFrame(schema={"icao": pl.String, "typecode": pl.String, "category": pl.String,
                                    "desc_code": pl.String, "max_gs_kt": pl.Float64,
                                    "keep": pl.Boolean, "reason": pl.String})
    # Type/category: prefer the aircraft table, else the per-row values on the tracks.
    per = ts.tracks.group_by("icao").agg(
        pl.col("gs_kt").max().alias("max_gs_kt"),
        pl.col("typecode").drop_nulls().first().alias("t_type"),
        pl.col("category").drop_nulls().first().alias("t_cat"),
    )
    ac = ts.aircraft.select("icao", "typecode", "category") if ts.aircraft.height else pl.DataFrame(
        schema={"icao": pl.String, "typecode": pl.String, "category": pl.String})
    df = per.join(ac, on="icao", how="left").with_columns(
        pl.coalesce(pl.col("typecode").replace("", None), pl.col("t_type")).alias("typecode"),
        pl.coalesce(pl.col("category").replace("", None), pl.col("t_cat")).alias("category"),
    ).drop("t_type", "t_cat")
    rows = []
    for r in df.sort("icao").iter_rows(named=True):
        keep, reason = types.classify(r["typecode"], r["category"], r["max_gs_kt"], exclude_helicopters)
        rows.append({**r, "desc_code": types.desc_code(r["typecode"]), "keep": keep, "reason": reason})
    return pl.DataFrame(rows, schema={
        "icao": pl.String, "typecode": pl.String, "category": pl.String, "max_gs_kt": pl.Float64,
        "desc_code": pl.String, "keep": pl.Boolean, "reason": pl.String,
    }).select("icao", "typecode", "category", "desc_code", "max_gs_kt", "keep", "reason")


def exclude_jets(
    ts: TrackSet, types: AircraftTypes | None = None, exclude_helicopters: bool = False
) -> TrackSet:
    cls = classify_aircraft(ts, types, exclude_helicopters)
    keep = cls.filter(pl.col("keep"))["icao"].to_list()
    return ts.model_copy(update={
        "tracks": ts.tracks.filter(pl.col("icao").is_in(keep)),
        "aircraft": ts.aircraft.filter(pl.col("icao").is_in(keep)),
    })


def _interp_segment(icao: str, seg: int, d: pl.DataFrame, origin: tuple[float, float]):
    t = d["t"].to_numpy()
    geom = d["alt_geom_ft"].to_numpy() * FT_M
    baro = d["alt_baro_ft"].to_numpy() * FT_M
    has_g, has_b = ~np.isnan(geom), ~np.isnan(baro)
    ok = has_g | has_b
    if ok.sum() < 2:
        return None
    t, geom, baro, has_g, has_b = t[ok], geom[ok], baro[ok], has_g[ok], has_b[ok]
    both = has_g & has_b
    off = float(np.median(geom[both] - baro[both])) if both.any() else 0.0
    u = np.where(has_g, geom, baro + off)  # geom first, else baro shifted onto the geom datum
    lat, lon = d["lat"].to_numpy()[ok], d["lon"].to_numpy()[ok]
    e, n, up = geodetic_to_enu(lat, lon, u, origin[0], origin[1])
    gs = d["gs_kt"].to_numpy()[ok] * KT_MS
    # strictly increasing t
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, e, n, up, gs, has_g = t[keep], e[keep], n[keep], up[keep], gs[keep], has_g[keep]
    if len(t) < 2:
        return None
    tg = np.arange(math.ceil(t[0]), math.floor(t[-1]) + 1, dtype=float)
    if len(tg) == 0:
        return None
    xyz = np.stack([e, n, up], axis=1)
    out = PchipInterpolator(t, xyz, axis=0)(tg)
    g_ok = ~np.isnan(gs)
    gs_i = np.interp(tg, t[g_ok], gs[g_ok]) if g_ok.any() else np.full(len(tg), np.nan)
    geom_frac = np.interp(tg, t, has_g.astype(float))
    is_geom = geom_frac >= 0.5
    return pl.DataFrame({
        "icao": icao, "seg": seg, "t": tg, "e": out[:, 0], "n": out[:, 1], "u": out[:, 2],
        "gs_ms": gs_i, "alt_sigma_m": np.where(is_geom, SIGMA_GEOM_M, SIGMA_BARO_M),
        "alt_src": np.where(is_geom, "geom", "baro"),
    }, schema=PREPARED_SCHEMA)


def prepare_tracks(
    tracks: pl.DataFrame, origin: tuple[float, float], max_gap_s: float = MAX_GAP_S
) -> pl.DataFrame:
    """Split at gaps > max_gap_s and PCHIP-resample to a 1-s grid in ENU about `origin` (lat, lon).

    Altitude precedence: geometric (HAE) when present, else barometric (shifted by the segment's
    median geom-baro offset when both exist) flagged with `alt_sigma_m` = 100 m. Ground rows and
    rows with no altitude are dropped; segments with < 2 usable samples are dropped.
    """
    empty = pl.DataFrame(schema=PREPARED_SCHEMA)
    if tracks.height == 0:
        return empty
    df = tracks.filter(~pl.col("on_ground").fill_null(False) & pl.col("lat").is_not_null()
                       & pl.col("lon").is_not_null() & pl.col("t").is_not_null()).sort("icao", "t")
    df = df.with_columns(
        (pl.col("t").diff().over("icao").fill_null(0.0) > max_gap_s)
        .cast(pl.Int64).cum_sum().over("icao").alias("seg")
    ).with_columns(
        pl.col("alt_geom_ft").fill_null(float("nan")), pl.col("alt_baro_ft").fill_null(float("nan")),
    )
    parts = []
    for (icao, seg), d in df.partition_by(["icao", "seg"], as_dict=True, maintain_order=True).items():
        p = _interp_segment(icao, int(seg), d, origin)
        if p is not None:
            parts.append(p)
    return pl.concat(parts) if parts else empty


def prepare_trackset(
    ts: TrackSet,
    origin: tuple[float, float],
    types: AircraftTypes | None = None,
    exclude_helicopters: bool = False,
    exclude_jets_: bool = True,
    max_gap_s: float = MAX_GAP_S,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Classify, optionally drop jets, and prepare 1-s ENU tracks. Returns (prepared, classification)."""
    cls = classify_aircraft(ts, types, exclude_helicopters)
    tracks = ts.tracks
    if exclude_jets_:
        tracks = tracks.filter(pl.col("icao").is_in(cls.filter(pl.col("keep"))["icao"].to_list()))
    return prepare_tracks(tracks, origin, max_gap_s), cls
