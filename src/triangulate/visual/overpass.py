"""Overpass QL generated from validated `OsmHint` structures (plan 3.6 step 5, WP16).

Raw QL from the model is never executed. Safety rules:
- tag keys must be in `ALLOWED_KEYS` (or be `name:<lang>` / `addr:*`); anything else rejects the hint;
- values are matched against a conservative character whitelist and then escaped; operators are
  limited to `=`, `!=`, `~` (always a *literal* case-insensitive contains-match, regex
  metacharacters are escaped) and `exists`;
- every query is restricted to the job bbox (the `near` feature set to the bbox grown by the
  distance), has `[timeout:25][maxsize:64MiB]`, and a result cap;
- responses are cached under `data/cache/overpass/`, requests are at least `min_interval` apart and
  the client has a hard per-job query budget (`max_queries`, default 3).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..core.geo import BBox
from ..log import get_logger
from ._http import JsonCache, RateLimiter, shared_limiter
from .schemas import OsmHint, OsmTag

log = get_logger("visual.overpass")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ATTRIBUTION = "Map data (c) OpenStreetMap contributors (ODbL), via Overpass API"
TIMEOUT_S = 25
MAXSIZE = 64 * 1024 * 1024
RESULT_CAP = 200

ALLOWED_KEYS = frozenset({
    "name", "alt_name", "brand", "operator", "network", "ref", "amenity", "shop", "highway", "building",
    "natural", "landuse", "leisure", "tourism", "railway", "public_transport", "man_made", "historic",
    "cuisine", "religion", "denomination", "power", "barrier", "bridge", "tunnel", "waterway", "place",
    "office", "craft", "surface", "lanes", "maxspeed", "route", "bus", "genus", "species", "emergency",
    "aeroway", "sport", "healthcare", "military", "boundary", "traffic_sign", "crossing", "oneway",
    "cycleway", "sidewalk", "lit", "street_cabinet", "pole", "utility", "wall", "roof:shape",
    "building:levels", "building:material", "addr:street", "addr:city", "addr:postcode",
})
_LANG_NAME = re.compile(r"^(name|alt_name|brand|operator):[a-z]{2,3}(-[A-Za-z]{2,4})?$")
_VALUE_OK = re.compile(r"^[\w\s\-'.,&:/()+@#]{0,80}$")


class QueryRejected(ValueError):
    pass


def check_key(key: str) -> str:
    if key in ALLOWED_KEYS or _LANG_NAME.match(key):
        return key
    raise QueryRejected(f"OSM key not allowed: {key!r}")


def _escape(value: str) -> str:
    if not _VALUE_OK.match(value) or "\n" in value:
        raise QueryRejected(f"OSM value not allowed: {value[:40]!r}")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def tag_filter(t: OsmTag) -> str:
    k = check_key(t.key)
    if t.op == "exists":
        return f'["{k}"]'
    if not t.value:
        raise QueryRejected("empty value")
    v = _escape(t.value)
    if t.op == "=":
        return f'["{k}"="{v}"]'
    if t.op == "!=":
        return f'["{k}"!="{v}"]'
    if t.op == "~":
        return f'["{k}"~"{re.escape(t.value)}",i]'
    raise QueryRejected(f"operator not allowed: {t.op!r}")


def _is_selective(tags: list[OsmTag]) -> bool:
    """At least one equality/contains constraint (not only `exists` / `!=`)."""
    return any(t.op in ("=", "~") and t.value for t in tags)


def _expand(bbox: BBox, metres: float) -> BBox:
    s, w, n, e = bbox
    dlat = metres / 111_320.0
    dlon = metres / (111_320.0 * max(math.cos(math.radians((s + n) / 2)), 0.05))
    return (max(s - dlat, -90.0), max(w - dlon, -180.0), min(n + dlat, 90.0), min(e + dlon, 180.0))


def _bb(b: BBox) -> str:
    return f"({b[0]:.6f},{b[1]:.6f},{b[2]:.6f},{b[3]:.6f})"


def build_ql(hint: OsmHint, bbox: BBox, cap: int = RESULT_CAP) -> str:
    """Overpass QL for a hint, clamped to `bbox`. Raises `QueryRejected` for anything not whitelisted."""
    if not 1 <= len(hint.tags) <= 4:
        raise QueryRejected("1 to 4 tags required")
    if not _is_selective(hint.tags):
        raise QueryRejected("hint has no selective tag")
    filt = "".join(tag_filter(t) for t in hint.tags)
    head = f"[out:json][timeout:{TIMEOUT_S}][maxsize:{MAXSIZE}];"
    tail = f"out center tags {int(cap)};"
    if hint.near is None:
        return f"{head}\nnwr{filt}{_bb(bbox)};\n{tail}"
    if not 1 <= len(hint.near.tags) <= 3 or not _is_selective(hint.near.tags):
        raise QueryRejected("near needs 1 to 3 tags with a selective one")
    d = min(2000.0, max(1.0, float(hint.near.distance_m)))
    nfilt = "".join(tag_filter(t) for t in hint.near.tags)
    return (f"{head}\n(nwr{filt}{_bb(bbox)};)->.a;\n(nwr{nfilt}{_bb(_expand(bbox, d))};)->.b;\n"
            f"nwr.a(around.b:{d:.0f});\n{tail}")


class OverpassError(RuntimeError):
    pass


@dataclass
class OsmFeature:
    lat: float
    lon: float
    osm_type: str
    osm_id: int
    tags: dict[str, str]


def parse_elements(data: dict[str, Any]) -> list[OsmFeature]:
    out = []
    for el in data.get("elements", []) if isinstance(data, dict) else []:
        try:
            c = el.get("center") or el
            out.append(OsmFeature(float(c["lat"]), float(c["lon"]), str(el["type"]), int(el["id"]),
                                  {str(k): str(v)[:100] for k, v in (el.get("tags") or {}).items()}))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


class Overpass:
    def __init__(self, cache_dir: Path, user_agent: str, *, client: httpx.Client | None = None,
                 url: str = OVERPASS_URL, min_interval: float = 2.0, max_queries: int = 3,
                 limiter: RateLimiter | None = None, check_cancel: Callable[[], None] | None = None):
        self.cache = JsonCache(cache_dir)
        self.user_agent, self.url, self.max_queries = user_agent, url, max_queries
        self.limiter = limiter or shared_limiter("overpass", min_interval)
        self.check_cancel = check_cancel
        self._client = client
        self.n_requests = 0
        self.n_cache_hits = 0
        self.blocked = False

    @classmethod
    def from_settings(cls, settings, **kw) -> Overpass:
        return cls(settings.cache_subdir("overpass"), settings.user_agent, **kw)

    def run(self, ql: str) -> tuple[list[OsmFeature], bool]:
        """Execute (or load from cache) a query. Returns (features, truncated_at_cap)."""
        key = self.cache.key({"url": self.url, "ql": ql})
        data = self.cache.get(key)
        if data is not None:
            self.n_cache_hits += 1
        else:
            if self.blocked:
                raise OverpassError("Overpass refused this client earlier in the job")
            if self.n_requests >= self.max_queries:
                raise OverpassError(f"Overpass query budget ({self.max_queries}) exhausted")
            if self.check_cancel:
                self.check_cancel()
            self.limiter.wait()
            self.n_requests += 1
            client = self._client or httpx.Client(timeout=TIMEOUT_S + 10.0)
            try:
                r = client.post(self.url, data={"data": ql}, headers={"User-Agent": self.user_agent})
            except httpx.HTTPError as e:
                raise OverpassError(f"Overpass request failed: {type(e).__name__}") from e
            finally:
                if self._client is None:
                    client.close()
            if r.status_code in (429, 504):
                self.blocked = True
            if r.status_code != 200:
                raise OverpassError(f"Overpass returned HTTP {r.status_code}")
            try:
                data = r.json()
            except ValueError as e:
                raise OverpassError("Overpass returned non-JSON") from e
            remark = str(data.get("remark", "")) if isinstance(data, dict) else ""
            if not isinstance(data, dict) or "elements" not in data or "runtime error" in remark:
                raise OverpassError(f"Overpass error: {remark[:120]}")
            self.cache.put(key, {"ql": ql}, data)
        feats = parse_elements(data)
        return feats, len(feats) >= RESULT_CAP

    def query_hint(self, hint: OsmHint, bbox: BBox) -> tuple[list[OsmFeature], bool]:
        return self.run(build_ql(hint, bbox))
