"""Geodesy helpers. bbox convention everywhere: (south, west, north, east) in degrees."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer

EARTH_R = 6_371_008.8  # mean radius, m

BBox = tuple[float, float, float, float]


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (scalars or numpy arrays)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def validate_bbox(b: BBox) -> BBox:
    s, w, n, e = (float(x) for x in b)
    if not (-90 <= s < n <= 90):
        raise ValueError(f"invalid bbox latitudes: south={s}, north={n}")
    if not (-180 <= w <= 180 and -180 <= e <= 180) or w == e:
        raise ValueError(f"invalid bbox longitudes: west={w}, east={e}")
    return (s, w, n, e)


def parse_bbox(text: str) -> BBox:
    """'S,W,N,E' -> bbox."""
    parts = [float(x) for x in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'S,W,N,E'")
    return validate_bbox((parts[0], parts[1], parts[2], parts[3]))


def bbox_center(b: BBox) -> tuple[float, float]:
    s, w, n, e = b
    lon_span = (e - w) % 360 if e < w else e - w  # antimeridian-safe
    lon = w + lon_span / 2
    if lon > 180:
        lon -= 360
    return ((s + n) / 2, lon)


def bbox_contains(b: BBox, lat, lon):
    s, w, n, e = b
    lat, lon = np.asarray(lat), np.asarray(lon)
    in_lon = (lon >= w) & (lon <= e) if w <= e else (lon >= w) | (lon <= e)
    return (lat >= s) & (lat <= n) & in_lon


def bbox_expand_m(b: BBox, margin_m: float) -> BBox:
    """Grow a bbox by a metric margin (longitude margin scaled at the worst-case latitude)."""
    s, w, n, e = b
    dlat = math.degrees(margin_m / EARTH_R)
    worst = min(max(abs(s), abs(n)) + dlat, 89.0)
    dlon = math.degrees(margin_m / (EARTH_R * math.cos(math.radians(worst))))
    return (max(s - dlat, -90.0), max(w - dlon, -180.0), min(n + dlat, 90.0), min(e + dlon, 180.0))


def bbox_size_m(b: BBox) -> tuple[float, float]:
    """(height, width) in metres, width measured at the mid latitude."""
    s, w, n, e = b
    lat0, _ = bbox_center(b)
    return (
        float(haversine_m(s, w, n, w)),
        float(haversine_m(lat0, w, lat0, e)),
    )


@lru_cache(maxsize=64)
def laea_crs(lat0: float, lon0: float) -> CRS:
    """Lambert Azimuthal Equal-Area CRS centred on (lat0, lon0), metres."""
    return CRS.from_proj4(
        f"+proj=laea +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


@lru_cache(maxsize=64)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def to_laea(lat, lon, lat0: float, lon0: float):
    """(lat, lon) -> (x, y) metres in the LAEA CRS centred on (lat0, lon0)."""
    t = Transformer.from_crs("EPSG:4326", laea_crs(lat0, lon0), always_xy=True)
    return t.transform(lon, lat)


def from_laea(x, y, lat0: float, lon0: float):
    """(x, y) LAEA metres -> (lat, lon)."""
    t = Transformer.from_crs(laea_crs(lat0, lon0), "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return lat, lon


def geodetic_to_ecef(lat, lon, h):
    x, y, z = _transformer("EPSG:4979", "EPSG:4978").transform(lon, lat, h)
    return np.asarray(x), np.asarray(y), np.asarray(z)


def geodetic_to_enu(lat, lon, h, lat0: float, lon0: float, h0: float = 0.0):
    """Geodetic (ellipsoidal height) -> local East/North/Up metres about (lat0, lon0, h0)."""
    x, y, z = geodetic_to_ecef(lat, lon, h)
    x0, y0, z0 = (float(v) for v in geodetic_to_ecef(lat0, lon0, h0))
    dx, dy, dz = x - x0, y - y0, z - z0
    sl, cl = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    so, co = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def enu_to_geodetic(e, n, u, lat0: float, lon0: float, h0: float = 0.0):
    sl, cl = math.sin(math.radians(lat0)), math.cos(math.radians(lat0))
    so, co = math.sin(math.radians(lon0)), math.cos(math.radians(lon0))
    e, n, u = np.asarray(e), np.asarray(n), np.asarray(u)
    dx = -so * e - sl * co * n + cl * co * u
    dy = co * e - sl * so * n + cl * so * u
    dz = cl * n + sl * u
    x0, y0, z0 = (float(v) for v in geodetic_to_ecef(lat0, lon0, h0))
    lon, lat, h = _transformer("EPSG:4978", "EPSG:4979").transform(x0 + dx, y0 + dy, z0 + dz)
    return np.asarray(lat), np.asarray(lon), np.asarray(h)
