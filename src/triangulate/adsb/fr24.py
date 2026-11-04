"""Flightradar24 provider (stub).

Intended API: the paid FR24 API (`fr24api.flightradar24.com`), endpoints `flight-positions/light`
and `flight-tracks` (Bearer token, per-credit billing). A query would list flights inside the bbox
at sampled instants, fetch their tracks, and map the points onto `core.types.TRACK_SCHEMA`.
Licence: FR24 commercial terms; data may not be redistributed.
"""

from __future__ import annotations

from datetime import date, datetime

from ..core.progress import Progress
from ..core.types import TrackQuery, TrackSet


class Fr24Provider:
    name = "fr24"
    licence = "Flightradar24 API terms (no redistribution)"

    def __init__(self, **_: object) -> None:
        pass

    def days_available(self, t0: datetime, t1: datetime) -> list[date]:
        raise NotImplementedError("FR24 provider is not implemented in v1; see module docstring")

    def fetch(self, q: TrackQuery, progress: Progress) -> TrackSet:
        raise NotImplementedError("FR24 provider is not implemented in v1; see module docstring")
