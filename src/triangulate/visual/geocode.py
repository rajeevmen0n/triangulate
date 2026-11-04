"""Nominatim client for the visual clue layers (plan 3.6 step 5, WP16).

Usage-policy compliance (https://operations.osmfoundation.org/policies/nominatim/):
- at most one request per second (process-wide limiter), sequential only;
- identifying User-Agent from `Settings.user_agent` ("triangulate/0.1" + optional contact);
- every response, including empty ones, is cached under `data/cache/geocode/`; cache hits make no
  request, so a rerun costs nothing;
- no bulk geocoding: a hard per-client request budget (`max_requests`), and every query is clamped
  to the job bbox with `viewbox` + `bounded=1` (hits are also filtered to the bbox locally);
- HTTP 429/403 stops all further requests from that client (`blocked`), nothing is retried.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..core.geo import BBox
from ..log import get_logger
from ._http import JsonCache, RateLimiter, shared_limiter

log = get_logger("visual.geocode")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
ATTRIBUTION = "Geocoding by Nominatim, data (c) OpenStreetMap contributors (ODbL)"


class GeocodeError(RuntimeError):
    pass


@dataclass
class GeoHit:
    lat: float
    lon: float
    name: str = ""
    osm_type: str = ""
    osm_id: int | None = None
    category: str = ""
    type: str = ""
    importance: float = 0.0
    bbox: BBox | None = None  # (S, W, N, E) of the feature
    geojson: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        if self.osm_id is not None:
            return f"{self.osm_type}{self.osm_id}"
        return f"{self.lat:.5f},{self.lon:.5f}"

    def geometry(self):
        """Shapely geometry in lon/lat (Point when Nominatim gave no geometry)."""
        from shapely.geometry import Point, shape

        if self.geojson:
            try:
                g = shape(self.geojson)
                if not g.is_empty:
                    return g
            except Exception:  # noqa: BLE001
                pass
        return Point(self.lon, self.lat)


def _parse_hit(d: dict[str, Any]) -> GeoHit | None:
    try:
        lat, lon = float(d["lat"]), float(d["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    bb = None
    try:
        s, n, w, e = (float(v) for v in d["boundingbox"])
        bb = (s, w, n, e)
    except (KeyError, TypeError, ValueError):
        pass
    gj = d.get("geojson")
    return GeoHit(lat=lat, lon=lon, name=str(d.get("display_name", ""))[:200],
                  osm_type=str(d.get("osm_type", "")), osm_id=d.get("osm_id"),
                  category=str(d.get("category", d.get("class", ""))), type=str(d.get("type", "")),
                  importance=float(d.get("importance") or 0.0), bbox=bb,
                  geojson=gj if isinstance(gj, dict) else None)


def in_bbox(lat: float, lon: float, bbox: BBox, pad_deg: float = 0.0) -> bool:
    s, w, n, e = bbox
    return s - pad_deg <= lat <= n + pad_deg and w - pad_deg <= lon <= e + pad_deg


class Geocoder:
    def __init__(self, cache_dir: Path, user_agent: str, *, client: httpx.Client | None = None,
                 url: str = NOMINATIM_URL, min_interval: float = 1.1, max_requests: int = 25,
                 limiter: RateLimiter | None = None, check_cancel: Callable[[], None] | None = None):
        self.cache = JsonCache(cache_dir)
        self.user_agent = user_agent
        self.url = url
        self.max_requests = max_requests
        self.limiter = limiter or shared_limiter("nominatim", min_interval)
        self.check_cancel = check_cancel
        self._client = client
        self.n_requests = 0
        self.n_cache_hits = 0
        self.blocked = False

    @classmethod
    def from_settings(cls, settings, **kw) -> Geocoder:
        return cls(settings.cache_subdir("geocode"), settings.user_agent, **kw)

    def _fetch(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        key = self.cache.key({"url": self.url, "params": params})
        hit = self.cache.get(key)
        if hit is not None:
            self.n_cache_hits += 1
            return hit
        if self.blocked:
            raise GeocodeError("Nominatim blocked this client earlier in the job; not retrying")
        if self.n_requests >= self.max_requests:
            raise GeocodeError(f"geocoder request budget ({self.max_requests}) exhausted")
        if self.check_cancel:
            self.check_cancel()
        self.limiter.wait()
        self.n_requests += 1
        client = self._client or httpx.Client(timeout=20.0)
        try:
            r = client.get(self.url, params=params, headers={"User-Agent": self.user_agent})
        except httpx.HTTPError as e:
            raise GeocodeError(f"Nominatim request failed: {type(e).__name__}") from e
        finally:
            if self._client is None:
                client.close()
        if r.status_code in (403, 429):
            self.blocked = True
            raise GeocodeError(f"Nominatim returned HTTP {r.status_code}; stopping")
        if r.status_code != 200:
            raise GeocodeError(f"Nominatim returned HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError as e:
            raise GeocodeError("Nominatim returned non-JSON") from e
        if not isinstance(data, list):
            raise GeocodeError("unexpected Nominatim response")
        self.cache.put(key, params, data)
        return data

    def search(self, query: str, bbox: BBox, *, countrycodes: list[str] | None = None, limit: int = 5,
               polygon: bool = True) -> list[GeoHit]:
        """Free-text search clamped to `bbox` (viewbox + bounded, plus a local filter)."""
        q = " ".join(query.split())[:200]
        if not q:
            return []
        s, w, n, e = bbox
        params: dict[str, Any] = {
            "q": q, "format": "jsonv2", "limit": max(1, min(int(limit), 10)), "addressdetails": 0,
            "viewbox": f"{w:.6f},{n:.6f},{e:.6f},{s:.6f}", "bounded": 1, "accept-language": "en",
        }
        if countrycodes:
            params["countrycodes"] = ",".join(sorted({c.lower() for c in countrycodes}))
        if polygon:
            params["polygon_geojson"] = 1
            params["polygon_threshold"] = 0.001
        hits = [h for d in self._fetch(params) if (h := _parse_hit(d)) is not None]
        out = []
        for h in hits:
            b = h.bbox
            near = b is not None and not (b[2] < s or b[0] > n or b[3] < w or b[1] > e)
            if in_bbox(h.lat, h.lon, bbox) or near:
                out.append(h)
        return out

    def admin_area(self, country: str, name: str | None = None) -> GeoHit | None:
        """Boundary polygon of a country (ISO2) or of an admin-1 region `name` inside it. Not clamped
        to the bbox (the polygon is what we want); callers clip it. Simplified, cached."""
        params: dict[str, Any] = {"format": "jsonv2", "limit": 1, "addressdetails": 0,
                                  "polygon_geojson": 1, "polygon_threshold": 0.01,
                                  "accept-language": "en"}
        if name:
            params.update({"state": name[:100], "countrycodes": country.lower()})
        else:
            params.update({"country": country.upper()})
        hits = [h for d in self._fetch(params) if (h := _parse_hit(d)) is not None]
        return hits[0] if hits else None
