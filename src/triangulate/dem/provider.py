"""DEM provider facade. `get_dem()` returns a `DemProvider` (metres, orthometric/EGM2008)."""

from __future__ import annotations

from .copernicus import CopernicusDem
from .geoid import geoid_N

__all__ = ["CopernicusDem", "geoid_N", "get_dem", "elevation", "ellipsoidal_height"]

_providers: dict[int, CopernicusDem] = {}


def get_dem(resolution_m: int = 30) -> CopernicusDem:
    """Shared provider for GLO-30 (default) or GLO-90 (`resolution_m=90`, coarse stages)."""
    if resolution_m not in _providers:
        _providers[resolution_m] = CopernicusDem(resolution_m)
    return _providers[resolution_m]


def elevation(lat, lon, resolution_m: int = 30):
    return get_dem(resolution_m).elevation(lat, lon)


def ellipsoidal_height(lat, lon, resolution_m: int = 30):
    """DEM elevation + geoid undulation N (HAE, comparable to ADS-B alt_geom)."""
    return elevation(lat, lon, resolution_m) + geoid_N(lat, lon)
