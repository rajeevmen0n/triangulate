"""Metadata extraction (plan 3.5). Output is an untrusted, optional hint, never ground truth.

Every hint has `trust="low"` (or "very_low" when re-encode markers are found) and a reason. Nothing
here overrides user constraints; consumers (fusion/pipeline) turn hints into a GPS mixture prior
(alpha = 0.6, sigma = max(accuracy, 30 m)) and a mixture prior on T_s.

Public API:
    extract_metadata(path) -> MetadataReport      (never raises for missing/odd metadata)
    time_components(report, tz=None, ...) -> list[TimeComponent]   (weights sum to 1)
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from ..core.timeutil import UTC

Trust = Literal["low", "very_low"]

REENCODE_ENCODERS = ("lavf", "lavc", "yt-dlp", "youtube", "handbrake", "libx264", "x264", "ffmpeg")
GPS_SIGMA_FLOOR_M = 30.0
GPS_ALPHA = 0.6


class GpsHint(BaseModel):
    lat: float
    lon: float
    alt_m: float | None = None
    accuracy_m: float | None = None
    source: str
    trust: Trust = "low"
    reason: str = "metadata is untrusted"

    @property
    def sigma_m(self) -> float:
        return max(self.accuracy_m or 0.0, GPS_SIGMA_FLOOR_M)


class TimeHint(BaseModel):
    """A raw timestamp from metadata. `naive` is the wall clock as written; `offset_min` is an
    explicit UTC offset when the tag carried one (else None: ambiguous)."""

    field: str
    naive: datetime  # tz-naive wall clock
    offset_min: int | None = None
    source: str
    trust: Trust = "low"
    reason: str = "metadata is untrusted"


class TimeComponent(BaseModel):
    """One component of the mixture prior on T_s (start of recording, UTC)."""

    t_s: datetime  # aware UTC
    weight: float
    sigma_s: float = 60.0
    label: str
    trust: Trust = "low"


class MetadataReport(BaseModel):
    path: str
    gps: list[GpsHint] = Field(default_factory=list)
    times: list[TimeHint] = Field(default_factory=list)
    make: str | None = None
    model: str | None = None
    encoder: str | None = None
    duration_s: float | None = None
    sample_rate: int | None = None
    reencode_markers: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def has_hints(self) -> bool:
        return bool(self.gps or self.times)

    @property
    def best_gps(self) -> GpsHint | None:
        return self.gps[0] if self.gps else None


# ---------------------------------------------------------------- parsing helpers

_ISO6709 = re.compile(r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?(?:CRS\w+)?/?\s*$")


def parse_iso6709(text: str) -> tuple[float, float, float | None] | None:
    """Decimal-degree ISO 6709 (`+40.8767-073.5264/`, optional altitude). DMS forms are not handled."""
    m = _ISO6709.match(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    alt = float(m.group(3)) if m.group(3) else None
    return (lat, lon, alt) if _valid(lat, lon) else None


def _valid(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0)


_DT_FORMS = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S",
             "%Y:%m:%d %H:%M:%S.%f")


def parse_datetime(text: str) -> tuple[datetime, int | None] | None:
    """Parse ISO/EXIF-style timestamps -> (naive wall clock, offset minutes or None)."""
    s = str(text).strip()
    if not s or s.startswith("0000") or s.startswith("1904") or s.startswith("1970-01-01"):
        return None
    offset: int | None = None
    m = re.search(r"(Z|[+-]\d{2}:?\d{2})$", s)
    if m:
        z = m.group(1)
        if z == "Z":
            offset = 0
        else:
            sign = -1 if z[0] == "-" else 1
            digits = z[1:].replace(":", "")
            offset = sign * (int(digits[:2]) * 60 + int(digits[2:]))
        s = s[: m.start()].strip()
    for f in _DT_FORMS:
        try:
            return datetime.strptime(s, f), offset
        except ValueError:
            continue
    return None


def _parse_offset(text: Any) -> int | None:
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", str(text).strip())
    if not m:
        return None
    return (-1 if m.group(1) == "-" else 1) * (int(m.group(2)) * 60 + int(m.group(3)))


# ---------------------------------------------------------------- tools

def _run_json(cmd: list[str], timeout: float = 60.0) -> Any:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if not r.stdout.strip():
        raise RuntimeError(f"{cmd[0]}: no output (rc={r.returncode}) {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)


def _ffprobe(path: Path) -> dict[str, Any]:
    return _run_json(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
                      str(path)])


def _exiftool(path: Path) -> dict[str, Any]:
    data = _run_json(["exiftool", "-json", "-n", "-G1", "-api", "largefilesupport=1", str(path)])
    return data[0] if data else {}


def _short(key: str) -> str:
    return key.split(":", 1)[-1]


def _lower_tags(tags: dict[str, Any] | None) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (tags or {}).items()}


# ---------------------------------------------------------------- main entry

def extract_metadata(path: str | Path) -> MetadataReport:
    """Read ffprobe + exiftool metadata into untrusted hints. Never raises on missing metadata or
    unavailable/failing tools (problems are recorded in `errors`)."""
    p = Path(path)
    rep = MetadataReport(path=str(p))
    probe: dict[str, Any] = {}
    exif: dict[str, Any] = {}
    try:
        probe = _ffprobe(p)
    except Exception as e:  # noqa: BLE001 - metadata is optional
        rep.errors.append(f"ffprobe: {e}")
    try:
        exif = _exiftool(p)
    except Exception as e:  # noqa: BLE001
        rep.errors.append(f"exiftool: {e}")

    fmt = probe.get("format", {}) or {}
    streams = probe.get("streams", []) or []
    ftags = _lower_tags(fmt.get("tags"))
    stags = [_lower_tags(s.get("tags")) for s in streams]

    try:
        rep.duration_s = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
    except (TypeError, ValueError):
        pass
    for s in streams:
        if s.get("codec_type") == "audio" and s.get("sample_rate"):
            rep.sample_rate = int(s["sample_rate"])
            break

    # --- GPS (preference: EXIF/exiftool composite, then QuickTime location tags)
    gps: list[GpsHint] = []
    lat, lon = exif.get("Composite:GPSLatitude"), exif.get("Composite:GPSLongitude")
    if lat is None:
        for k, v in exif.items():
            if _short(k) == "GPSLatitude" and isinstance(v, (int, float)):
                lat = v
            if _short(k) == "GPSLongitude" and isinstance(v, (int, float)):
                lon = v
        ref_lat = next((v for k, v in exif.items() if _short(k) == "GPSLatitudeRef"), None)
        ref_lon = next((v for k, v in exif.items() if _short(k) == "GPSLongitudeRef"), None)
        if lat is not None and ref_lat == "S":
            lat = -abs(lat)
        if lon is not None and ref_lon == "W":
            lon = -abs(lon)
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and _valid(lat, lon):
        alt = exif.get("Composite:GPSAltitude")
        acc = next((float(v) for k, v in exif.items()
                    if _short(k) in ("GPSHPositioningError", "GPSHorizontalPositioningError")
                    and isinstance(v, (int, float))), None)
        gps.append(GpsHint(lat=float(lat), lon=float(lon),
                           alt_m=float(alt) if isinstance(alt, (int, float)) else None,
                           accuracy_m=acc, source="exif_gps"))
    candidates: list[tuple[str, str]] = []
    for tags in [ftags, *stags]:
        for key in ("location", "location-eng", "com.apple.quicktime.location.iso6709", "©xyz", "\xa9xyz"):
            if key in tags:
                candidates.append((key, tags[key]))
    for k, v in exif.items():
        if _short(k) in ("GPSCoordinates", "LocationInformation") or "iso6709" in _short(k).lower():
            candidates.append((_short(k), str(v)))
    for key, text in candidates:
        parsed = parse_iso6709(text)
        if parsed and not any(abs(g.lat - parsed[0]) < 1e-4 and abs(g.lon - parsed[1]) < 1e-4 for g in gps):
            gps.append(GpsHint(lat=parsed[0], lon=parsed[1], alt_m=parsed[2], source=f"container:{key}"))
    rep.gps = gps

    # --- times
    offsets = {_short(k): _parse_offset(v) for k, v in exif.items() if _short(k).startswith("OffsetTime")}
    times: list[TimeHint] = []

    def add_time(field: str, text: Any, source: str, offset: int | None = None) -> None:
        got = parse_datetime(text)
        if not got:
            return
        naive, off = got
        off = off if off is not None else offset
        if any(t.naive == naive and t.offset_min == off for t in times):
            return
        times.append(TimeHint(field=field, naive=naive, offset_min=off, source=source))

    for k, v in exif.items():
        s = _short(k)
        if s in ("DateTimeOriginal", "CreateDate", "SubSecDateTimeOriginal") and isinstance(v, str):
            off = offsets.get("OffsetTimeOriginal") if "Original" in s else offsets.get("OffsetTimeDigitized")
            add_time(s, v, "exif", off)
    for k, v in exif.items():
        if _short(k) in ("CreationDate",) and isinstance(v, str):  # QuickTime Keys:CreationDate (with offset)
            add_time("CreationDate", v, "quicktime")
    ct_values = [t["creation_time"] for t in [ftags, *stags] if "creation_time" in t]
    for i, v in enumerate(ct_values):
        add_time("creation_time", v, "container" if i == 0 else "stream")
    # QuickTime CreateDate from exiftool is UTC wall clock without a zone for mp4; skip if ffprobe had it.
    if not ct_values:
        for k, v in exif.items():
            if k.startswith("QuickTime:") and _short(k) == "CreateDate" and isinstance(v, str):
                add_time("CreateDate", v, "quicktime")

    # --- make/model/encoder and re-encode markers
    for k, v in exif.items():
        s = _short(k)
        if s == "Make" and not rep.make:
            rep.make = str(v)
        elif s == "Model" and not rep.model:
            rep.model = str(v)
        elif s == "Encoder" and not rep.encoder:
            rep.encoder = str(v)
    for tags in [ftags, *stags]:
        rep.make = rep.make or tags.get("make") or tags.get("com.apple.quicktime.make")
        rep.model = rep.model or tags.get("model") or tags.get("com.apple.quicktime.model")
        rep.encoder = rep.encoder or tags.get("encoder")
    markers: list[str] = []
    enc = (rep.encoder or "").lower()
    if enc and any(e in enc for e in REENCODE_ENCODERS):
        markers.append(f"encoder string suggests re-encode: {rep.encoder}")
    if (times or gps) and not (rep.make or rep.model):
        markers.append("no camera make/model")
    if len(ct_values) > 1 and len(set(ct_values)) == 1:
        markers.append("creation_time identical across streams")
    rep.reencode_markers = markers
    trust: Trust = "very_low" if markers else "low"
    reason = "metadata is untrusted" + (f"; lowered: {'; '.join(markers)}" if markers else "")
    for h in [*gps, *times]:
        h.trust = trust
        h.reason = reason
    rep.times = times
    return rep


# ---------------------------------------------------------------- hint -> evidence

def time_components(
    report: MetadataReport,
    tz: str | None = None,
    *,
    duration_s: float | None = None,
    sigma_s: float = 60.0,
    dst_error: bool = True,
) -> list[TimeComponent]:
    """Mixture prior on T_s (UTC start of the recording) from the best time hint.

    Interpretations of a wall clock without an explicit offset: as UTC; as local time in `tz`
    (if given); each with +-1 h DST-error variants. Each is also taken as the *end* of the recording
    (T_s = t - duration). A tag with an explicit offset gives that single interpretation (+ end).
    Weights are equal and sum to 1. Empty list when there is no time hint. All trust low.
    """
    if not report.times:
        return []
    dur = duration_s if duration_s is not None else report.duration_s
    hint = report.times[0]
    bases: list[tuple[datetime, str]] = []
    if hint.offset_min is not None:
        tzinfo = timezone(timedelta(minutes=hint.offset_min))
        t0 = hint.naive.replace(tzinfo=tzinfo).astimezone(UTC)
        bases.append((t0, f"{hint.field} with explicit offset"))
    else:
        bases.append((hint.naive.replace(tzinfo=UTC), f"{hint.field} as UTC"))
        if tz and tz != "UTC":
            local = hint.naive.replace(tzinfo=ZoneInfo(tz)).astimezone(UTC)
            bases.append((local, f"{hint.field} as local time ({tz})"))
    variants: list[tuple[datetime, str]] = []
    for t, lab in bases:
        variants.append((t, lab))
        if dst_error and hint.offset_min is None:
            variants.append((t + timedelta(hours=1), lab + " +1h (DST error)"))
            variants.append((t - timedelta(hours=1), lab + " -1h (DST error)"))
    if dur:
        variants += [(t - timedelta(seconds=dur), lab + " (timestamp = end of recording)")
                     for t, lab in list(variants)]
    w = 1.0 / len(variants)
    return [TimeComponent(t_s=t, weight=w, sigma_s=sigma_s, label=lab, trust=hint.trust)
            for t, lab in variants]
