"""Shared data contracts. Change only via the owning WP or coordinated edits (plan section 5).

Conventions: bbox = (south, west, north, east) degrees; times are timezone-aware UTC; distances in
metres, speeds in m/s unless a field name says otherwise (gs_kt, alt_*_ft follow ADS-B units).
v1 is region search only: a bbox and a date are REQUIRED.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal, Protocol

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .geo import BBox, validate_bbox
from .progress import Progress  # re-export

__all__ = [
    "Progress", "Range", "KnownAudioTime", "Constraints", "Event", "TrackQuery", "TrackSet",
    "TRACK_SCHEMA", "AIRCRAFT_SCHEMA", "TrackProvider", "DemProvider", "GridSpec", "CellId",
    "LayerRef", "ModuleResult", "Candidate", "Hypothesis", "GRID_LEVELS_M",
]


class Range(BaseModel):
    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.lo > self.hi:
            raise ValueError("lo > hi")
        return self

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def overlaps(self, other: Range) -> bool:
        return self.lo <= other.hi and other.lo <= self.hi


class KnownAudioTime(BaseModel):
    """The user knows the absolute UTC time of an audio position."""

    audio_pos_s: float
    utc: datetime
    sigma_s: float = 60.0


class Constraints(BaseModel):
    """What the user tells us. `bbox` and a date are required (region search only)."""

    model_config = ConfigDict(extra="forbid")

    bbox: BBox  # S, W, N, E
    date_start: date  # local date (in `tz`) of the recording, or first of a range
    date_end: date | None = None  # inclusive; None = single day
    tz: str | None = None  # IANA; None = infer from bbox centre
    window_start: time | None = None  # local wall-clock window for the recording start
    window_end: time | None = None
    known_audio_time: KnownAudioTime | None = None
    h_obs_m: float = 5.0
    temperature_c: float | None = None  # None -> c = 343 m/s
    max_range_days: int = 7

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, v: BBox) -> BBox:
        return validate_bbox(v)

    @model_validator(mode="after")
    def _dates(self) -> Constraints:
        if self.date_end is not None:
            if self.date_end < self.date_start:
                raise ValueError("date_end before date_start")
            if (self.date_end - self.date_start).days + 1 > self.max_range_days:
                raise ValueError(f"date range longer than {self.max_range_days} days")
        return self


class Event(BaseModel):
    """One aircraft Doppler event in the audio. t_h is audio position (s) of the heard CPA."""

    id: str
    t_h: float
    t_h_range: Range
    f0: float
    f0_range: Range
    v: float  # m/s
    v_range: Range
    d: float  # CPA slant distance, m
    d_range: Range
    snr: float = 0.0
    harmonics_seen: list[int] = Field(default_factory=lambda: [1])
    ridge_points: list[tuple[float, float]] = Field(default_factory=list)  # (t, f) pairs
    traced_line: list[tuple[float, float]] | None = None  # user polyline (LINE_POINTS)
    selected: bool = True
    quality_flags: list[str] = Field(default_factory=list)  # e.g. "non-aircraft"


class TrackQuery(BaseModel):
    bbox: BBox  # already includes any margin
    t0: datetime
    t1: datetime
    include_ground: bool = False


# Polars schemas (plan 1.5). `t` is float64 epoch seconds.
TRACK_SCHEMA: dict[str, type[pl.DataType]] = {
    "icao": pl.String, "reg": pl.String, "typecode": pl.String, "category": pl.String,
    "t": pl.Float64, "lat": pl.Float64, "lon": pl.Float64,
    "alt_baro_ft": pl.Float64, "alt_geom_ft": pl.Float64, "gs_kt": pl.Float64,
    "track_deg": pl.Float64, "vrate_fpm": pl.Float64, "on_ground": pl.Boolean, "source": pl.String,
}
AIRCRAFT_SCHEMA: dict[str, type[pl.DataType]] = {
    "icao": pl.String, "reg": pl.String, "typecode": pl.String, "desc": pl.String,
    "category": pl.String, "db_flags": pl.Int64, "owner_op": pl.String, "year": pl.String,
}


class TrackSet(BaseModel):
    """Tracks plus aircraft table; frames follow TRACK_SCHEMA / AIRCRAFT_SCHEMA."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tracks: pl.DataFrame
    aircraft: pl.DataFrame
    provider: str = ""
    licence: str = ""

    @classmethod
    def empty(cls, provider: str = "", licence: str = "") -> TrackSet:
        return cls(
            tracks=pl.DataFrame(schema=TRACK_SCHEMA),
            aircraft=pl.DataFrame(schema=AIRCRAFT_SCHEMA),
            provider=provider,
            licence=licence,
        )


class TrackProvider(Protocol):
    name: str  # "adsblol" | "opensky" | "fr24"
    licence: str

    def days_available(self, t0: datetime, t1: datetime) -> list[date]: ...
    def fetch(self, q: TrackQuery, progress: Progress) -> TrackSet: ...


class DemProvider(Protocol):
    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Metres, orthometric (EGM2008)."""
        ...


# --- Grid / fusion contracts (plan 1.3) ---------------------------------------------------------

GRID_LEVELS_M: tuple[int, ...] = (1600, 800, 400, 200, 100, 50, 25)  # level index -> cell size

CellId = tuple[int, int, int]  # (level, ix, iy)


class GridSpec(BaseModel):
    """LAEA grid centred on the search region. Cell (level, ix, iy) covers
    x in [x0 + ix*s, x0 + (ix+1)*s), y in [y0 + iy*s, y0 + (iy+1)*s) with s = GRID_LEVELS_M[level].
    x0, y0 are the lower-left corner of the region in LAEA metres, aligned to the coarsest size.
    """

    lat0: float
    lon0: float
    x0: float
    y0: float
    width_m: float
    height_m: float
    base_level: int = 3  # dense layer level (index into GRID_LEVELS_M)
    max_dense_cells: int = 4_000_000

    @staticmethod
    def cell_size(level: int) -> int:
        return GRID_LEVELS_M[level]

    def shape(self, level: int) -> tuple[int, int]:
        """(nx, ny) at a level."""
        s = GRID_LEVELS_M[level]
        return int(np.ceil(self.width_m / s)), int(np.ceil(self.height_m / s))

    def cell_center_xy(self, cell: CellId) -> tuple[float, float]:
        lv, ix, iy = cell
        s = GRID_LEVELS_M[lv]
        return self.x0 + (ix + 0.5) * s, self.y0 + (iy + 0.5) * s

    def xy_to_cell(self, x: float, y: float, level: int) -> CellId:
        s = GRID_LEVELS_M[level]
        return level, int(np.floor((x - self.x0) / s)), int(np.floor((y - self.y0) / s))

    @classmethod
    def for_bbox(cls, bbox: BBox, max_dense_cells: int = 4_000_000) -> GridSpec:
        """Choose the finest level that keeps the bbox within max_dense_cells."""
        from .geo import bbox_center, to_laea

        s, w, n, e = bbox
        lat0, lon0 = bbox_center(bbox)
        corners_lat = [s, s, n, n, (s + n) / 2, (s + n) / 2, s, n]
        corners_lon = [w, e, w, e, w, e, lon0, lon0]
        xs, ys = to_laea(np.array(corners_lat), np.array(corners_lon), lat0, lon0)
        c = GRID_LEVELS_M[0]
        x0 = float(np.floor(xs.min() / c) * c)
        y0 = float(np.floor(ys.min() / c) * c)
        spec = cls(
            lat0=lat0, lon0=lon0, x0=x0, y0=y0,
            width_m=float(xs.max() - x0), height_m=float(ys.max() - y0),
            max_dense_cells=max_dense_cells,
        )
        level = 0
        for lv in range(len(GRID_LEVELS_M)):
            nx, ny = spec.shape(lv)
            if nx * ny <= max_dense_cells:
                level = lv
        return spec.model_copy(update={"base_level": level})


class LayerRef(BaseModel):
    """Pointer to a log-density layer produced by a module."""

    name: str
    module: Literal["acoustic", "visual", "metadata"]
    path: str | None = None  # file in the job workspace (npz/tif)
    level: int | None = None
    weight: float = 1.0
    description: str = ""


class ModuleResult(BaseModel):
    module: Literal["acoustic", "visual", "metadata"]
    status: Literal["ok", "skipped", "degraded", "failed"] = "ok"
    message: str = ""  # why skipped/degraded
    layers: list[LayerRef] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    cost_usd: float = 0.0  # Claude API spend
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> relative path


class Hypothesis(BaseModel):
    """A cell plus one (flight, pass) per selected event, consistent in implied start offset."""

    cell: CellId
    offset_s: float  # o = t_h - tau: implied UTC epoch of audio sample 0
    flights: list[str]  # icao per event
    passes: list[int]
    d: list[float]
    v: list[float]
    score: float | None = None  # locked forward-model score
    dt_star: float | None = None
    f0_star: list[float] | None = None
    z: list[float] | None = None  # calibrated per-event z


class Candidate(BaseModel):
    rank: int
    lat: float
    lon: float
    log_density: float
    prob_mass: float | None = None
    contributions: dict[str, float] = Field(default_factory=dict)  # module -> weighted log-lik
    flights: list[str] = Field(default_factory=list)
    t_s: datetime | None = None  # best audio-sample-0 UTC time
    margin_to_runner_up: float | None = None
    hpd_level: float | None = None  # smallest HPD mass containing this point
    notes: list[str] = Field(default_factory=list)
