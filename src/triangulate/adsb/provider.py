"""Track-provider plumbing: shared errors, UTC-day helper and a provider factory.

The Protocols themselves (`TrackQuery`, `TrackSet`, `TrackProvider`) live in `core/types.py`
and are re-exported here for convenience.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..core.types import AIRCRAFT_SCHEMA, TRACK_SCHEMA, TrackProvider, TrackQuery, TrackSet

__all__ = [
    "AIRCRAFT_SCHEMA", "TRACK_SCHEMA", "TrackProvider", "TrackQuery", "TrackSet",
    "ProviderError", "ReleaseNotFound", "ReleaseNotPublished", "utc_days", "get_provider",
]


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ReleaseNotFound(ProviderError):
    """The upstream dataset has no release for that day."""


class ReleaseNotPublished(ReleaseNotFound):
    """The day is too recent: adsb.lol publishes about 3 h after the end of the UTC day."""


def utc_days(t0: datetime, t1: datetime) -> list[date]:
    """UTC calendar days touched by [t0, t1] (inclusive)."""
    d0 = t0.astimezone(UTC).date()
    d1 = t1.astimezone(UTC).date()
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def get_provider(name: str = "adsblol", **kwargs: Any) -> TrackProvider:
    if name == "adsblol":
        from .adsblol import AdsbLolProvider

        return AdsbLolProvider(**kwargs)
    if name == "opensky":
        from .opensky import OpenSkyProvider

        return OpenSkyProvider(**kwargs)
    if name == "fr24":
        from .fr24 import Fr24Provider

        return Fr24Provider(**kwargs)
    raise ValueError(f"unknown track provider {name!r}")
