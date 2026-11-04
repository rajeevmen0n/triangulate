"""Time handling. All internal times are UTC. A local date maps to one or two UTC dump days."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

UTC = UTC


def parse_utc(text: str) -> datetime:
    """ISO-8601 -> aware UTC datetime (naive input is taken as UTC)."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def infer_tz(lat: float, lon: float) -> str:
    """IANA zone name at a point (timezonefinder). Falls back to 'UTC'."""
    from timezonefinder import TimezoneFinder

    return TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"


def local_to_utc(d: date, t: time, tz: str) -> datetime:
    """Wall-clock local date+time in an IANA zone -> UTC. Nonexistent/ambiguous times use fold=0."""
    return datetime.combine(d, t, tzinfo=ZoneInfo(tz)).astimezone(UTC)


def local_date_to_utc_interval(d: date, tz: str) -> tuple[datetime, datetime]:
    """[local midnight, next local midnight) as UTC. Correct across DST days (23/25 h)."""
    start = local_to_utc(d, time(0, 0), tz)
    end = local_to_utc(d + timedelta(days=1), time(0, 0), tz)
    return start, end


def local_window_to_utc(
    d: date, tz: str, start: time | None = None, end: time | None = None
) -> tuple[datetime, datetime]:
    """Local date + optional wall-clock window -> UTC interval. No window = whole local day.
    If end <= start the window is taken to cross midnight (end on the next day)."""
    if start is None and end is None:
        return local_date_to_utc_interval(d, tz)
    start = start or time(0, 0)
    t0 = local_to_utc(d, start, tz)
    if end is None:
        return t0, local_date_to_utc_interval(d, tz)[1]
    end_day = d + timedelta(days=1) if end <= start else d
    return t0, local_to_utc(end_day, end, tz)


def utc_days(t0: datetime, t1: datetime) -> list[date]:
    """UTC calendar days touched by [t0, t1) (t1 exclusive; if t1 == t0 the single day)."""
    t0, t1 = t0.astimezone(UTC), t1.astimezone(UTC)
    last = t1 - timedelta(microseconds=1) if t1 > t0 else t0
    out, d = [], t0.date()
    while d <= last.date():
        out.append(d)
        d += timedelta(days=1)
    return out


def local_date_to_utc_days(d: date, tz: str) -> list[date]:
    """UTC dump days needed for a whole local date, e.g. 2026-09-08 America/New_York ->
    [2026-09-08, 2026-09-09]."""
    return utc_days(*local_date_to_utc_interval(d, tz))


def date_range(d0: date, d1: date) -> list[date]:
    if d1 < d0:
        raise ValueError("end date before start date")
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
