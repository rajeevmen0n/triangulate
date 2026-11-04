"""OpenSky Network provider (stub; fallback if adsb.lol lacks the demo aircraft, plan Decisions 1).

Intended API: the OpenSky historical database via Trino (`trino.opensky-network.org`, table
`minio.osky.state_vectors_data4`), which needs a research account (OAuth2 client credentials).
A query would select state vectors in [t0, t1] within the bbox (columns time, icao24, lat, lon,
baroaltitude, geoaltitude, velocity, heading, vertrate, onground, callsign), group by icao24 and map
them onto `core.types.TRACK_SCHEMA`. State vectors are 1 s or 10 s resolution; type codes come from
the aircraft database (`metadata/aircraft`). Licence: OpenSky terms of use (research use, cite the
OpenSky paper).
"""

from __future__ import annotations

from datetime import date, datetime

from ..core.progress import Progress
from ..core.types import TrackQuery, TrackSet


class OpenSkyProvider:
    name = "opensky"
    licence = "OpenSky Network terms of use (research/non-commercial; cite Schafer et al. 2014)"

    def __init__(self, **_: object) -> None:
        pass

    def days_available(self, t0: datetime, t1: datetime) -> list[date]:
        raise NotImplementedError("OpenSky provider is not implemented in v1; see module docstring")

    def fetch(self, q: TrackQuery, progress: Progress) -> TrackSet:
        raise NotImplementedError("OpenSky provider is not implemented in v1; see module docstring")
