"""Clue -> likelihood layers (plan 3.6 step 5, WP16) and the clue -> `SunClue` adapter (for WP17).

Entry point: `build_visual_layers(vc, bbox, spec, out_dir, ...)` returns the visual `ModuleResult` with
`layers` (WP12 `LayerRef`s, saved under `<out_dir>/layers/`) and attribution in provenance.

Clue -> layer mapping (every layer is `log(eps + (1 - eps) K)` with eps = 1 - effective confidence):
- `geocode_queries()` (clue queries + searchable entities), Nominatim bounded to the bbox: K is a
  Gaussian of the distance to the hit geometry (point, street line or polygon), sigma 150 m (broad
  places: 10 % of their extent, <= 3 km), max over the hits. Effective confidence =
  min(conf, 0.85) / sqrt(n_hits), x0.6 for broad places.
- `osm_hints`: Overpass (whitelisted QL, bbox-bound, max 3 queries per job, clues with conf >= 0.3);
  matches are points with sigma 120 m, K = max over matches; effective confidence =
  min(conf, 0.7) * min(1, sqrt(10 / n_matches)); a result that hits the 200 cap is "too generic" and
  dropped.
- country / admin-1 implications (all clue types: language, plates, architecture, vegetation ...):
  boundary polygon from Nominatim (cached), K = sum_c p_c * inside_c with a 5 km soft edge; effective
  confidence <= 0.6 (country) / 0.7 (admin-1). Layers that are flat over the bbox are dropped.
- `region_hypotheses`: Nominatim place search, broad kernel, confidence <= 0.5.
- Koppen codes and driving side are recorded in provenance but not turned into layers (no climate or
  driving-side data set ships with v1; at region scale they are almost always uniform).
- `place_candidates` are not geocoded separately (the model's `geocode_queries` cover them).

Never dominating: every layer's depth is at most -ln(0.1) = 2.3 nats, and if the summed depth of all
layers exceeds `MAX_TOTAL_NATS` their `Layer.weight`s are scaled down to that total (the module
weight 0.5 is applied later by fusion).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ..core.geo import BBox
from ..core.types import GridSpec, ModuleResult
from ..fusion.combine import domain_polygon_xy, membership_loglik
from ..fusion.grid import (
    FINEST_LEVEL,
    Layer,
    cell_centers_xy,
    coarsest_level_at_most,
    rasterize_geom,
)
from ..log import get_logger
from . import geocode as _geo
from . import overpass as _ovp
from .geocode import GeocodeError, Geocoder, GeoHit
from .llm import to_module_result
from .overpass import Overpass, OverpassError, QueryRejected
from .schemas import Clue, VisualClues

log = get_logger("visual.layers")

MAX_TOTAL_NATS = 8.0
MAX_LAYER_CONF = 0.9
MIN_CLUE_CONF = 0.15
GEOCODE_CONF_CAP = 0.85
OSM_CONF_CAP = 0.7
COUNTRY_CONF_CAP = 0.6
ADMIN1_CONF_CAP = 0.7
REGION_CONF_CAP = 0.5
POINT_SIGMA_M = 150.0
OSM_SIGMA_M = 120.0
SOFT_EDGE_SIGMA_M = 5000.0 / 3.0  # K ~ 0.01 at 5 km
MAX_GEOCODE_QUERIES = 10
MAX_ADMIN_LOOKUPS = 8
MAX_REGION_LOOKUPS = 4
MAX_CELLS = 1_000_000
ATTRIBUTION = ["Map data (c) OpenStreetMap contributors (ODbL)"]

Item = tuple[Any, float, float]  # (shapely lon/lat geometry, weight, sigma_m)


# --------------------------------------------------------------------------------------- fields

def _distance(spec: GridSpec, gxy, lv: int, ix0: int, iy0: int, nx: int, ny: int,
              cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Distance (m) from cell centres to a LAEA-metre geometry; 0 inside polygons."""
    import shapely
    from scipy.ndimage import distance_transform_edt

    from ..core.types import GRID_LEVELS_M

    parts = list(getattr(gxy, "geoms", [gxy]))
    polys = [p for p in parts if p.geom_type == "Polygon"]
    others = [p for p in parts if p.geom_type != "Polygon"]
    d = np.full(cx.shape, np.inf)
    pts = None
    big = [p for p in polys if p.area > 4e6]
    small = [p for p in polys if p.area <= 4e6]
    if big:
        inside = rasterize_geom(spec, shapely.union_all(big), lv, ix0, iy0, nx, ny)
        if inside.any():
            d = np.minimum(d, np.where(inside, 0.0, distance_transform_edt(~inside) * GRID_LEVELS_M[lv]))
    near = others + small
    if near:
        pts = shapely.points(cx.ravel(), cy.ravel())
        g = shapely.union_all(near)
        if g.geom_type != "Point":
            g = g.simplify(5.0)
        d = np.minimum(d, shapely.distance(pts, g).reshape(cx.shape))
    return d


def region_layer(spec: GridSpec, items: Sequence[Item], confidence: float, name: str, *,
                 combine: str = "max", module: str = "visual", description: str = "",
                 meta: dict[str, Any] | None = None, **kw) -> Layer:
    """Layer from lon/lat geometries: K = combine_i(w_i exp(-d_i^2 / (2 sigma_i^2))) in [0, 1]
    ("max" or "sum" clipped to 1), ell = log(eps + (1 - eps) K). One patch, window = geometry bounds
    + 5 sigma clipped to the grid; the fill is log(eps). Resolution: coarsest level with cell <=
    sigma / 2 (>= base level), coarsened until the window has <= MAX_CELLS cells."""
    conf = min(max(confidence, 0.0), MAX_LAYER_CONF)
    lay = Layer(spec=spec, name=name, module=module, description=description,
                fill=float(membership_loglik(0.0, conf)), **kw)
    lay.meta.update({"kind": "region", "confidence": conf, "n_items": len(items), **(meta or {})})
    if not items:
        return lay
    gxy = [(domain_polygon_xy(spec, g), w, s) for g, w, s in items]
    xmin = min(g.bounds[0] - 5 * s for g, _, s in gxy)
    ymin = min(g.bounds[1] - 5 * s for g, _, s in gxy)
    xmax = max(g.bounds[2] + 5 * s for g, _, s in gxy)
    ymax = max(g.bounds[3] + 5 * s for g, _, s in gxy)
    sig = max(s for _, _, s in gxy)
    lv = max(spec.base_level, min(coarsest_level_at_most(sig / 2.0), FINEST_LEVEL))
    while True:
        s_m = spec.cell_size(lv)
        nxt, nyt = spec.shape(lv)
        ix0 = max(0, int(math.floor((xmin - spec.x0) / s_m)))
        iy0 = max(0, int(math.floor((ymin - spec.y0) / s_m)))
        ix1 = min(nxt, int(math.ceil((xmax - spec.x0) / s_m)))
        iy1 = min(nyt, int(math.ceil((ymax - spec.y0) / s_m)))
        nx, ny = ix1 - ix0, iy1 - iy0
        if nx <= 0 or ny <= 0:
            return lay
        if nx * ny <= MAX_CELLS or lv == 0:
            break
        lv -= 1
    gj, gi = np.mgrid[iy0:iy1, ix0:ix1]
    cx, cy = cell_centers_xy(spec, lv, gi, gj)
    K = np.zeros((ny, nx))
    for g, w, s in gxy:
        d = _distance(spec, g, lv, ix0, iy0, nx, ny, cx, cy)
        k = w * np.exp(-0.5 * (d / s) ** 2)
        K = np.maximum(K, k) if combine == "max" else K + k
    K = np.clip(K, 0.0, 1.0)
    lay.add_patch(lv, ix0, iy0, membership_loglik(K, conf).astype(np.float32))
    lay.meta["level"] = lv
    return lay


def _flat(lay: Layer) -> bool:
    if not lay.patches:
        return True
    v = lay.patches[0].values
    return float(np.nanmax(v) - min(np.nanmin(v), lay.fill)) < 0.01


# --------------------------------------------------------------------------------------- helpers

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:60]


def _extent_m(h: GeoHit) -> float:
    if h.bbox is None:
        return 0.0
    s, w, n, e = h.bbox
    return float(math.hypot((n - s) * 111_320.0,
                            (e - w) * 111_320.0 * max(math.cos(math.radians((s + n) / 2)), 0.05)))


def _hit_sigma(h: GeoHit) -> float:
    return float(min(3000.0, max(POINT_SIGMA_M, 0.1 * _extent_m(h)))) if h.geojson and h.geojson.get(
        "type") in ("Polygon", "MultiPolygon") else POINT_SIGMA_M


def _clip(geom, bbox: BBox, pad_deg: float = 0.3):
    import shapely
    from shapely.validation import make_valid

    s, w, n, e = bbox
    box = shapely.box(w - pad_deg, s - pad_deg, e + pad_deg, n + pad_deg)
    try:
        g = geom if geom.is_valid else make_valid(geom)
        g = g.intersection(box)
    except Exception:  # noqa: BLE001
        return None
    return None if g.is_empty else g


class _Plan:
    """One layer to add, with the depth used for the global cap."""

    def __init__(self, layer: Layer, source: str, kind: str, conf: float, n: int = 1, detail: str = ""):
        self.layer, self.source, self.kind = layer, source, kind
        self.conf, self.n, self.detail = conf, n, detail

    @property
    def depth(self) -> float:
        return -float(self.layer.fill)


# --------------------------------------------------------------------------------------- builder

def build_visual_layers(vc: VisualClues, bbox: BBox, spec: GridSpec, out_dir: Path, *, settings=None,
                        geocoder: Geocoder | None = None, overpass: Overpass | None = None,
                        client: httpx.Client | None = None,
                        check_cancel: Callable[[], None] | None = None) -> ModuleResult:
    """Turn `clues.json` content into clue layers and a `ModuleResult` (see module docstring).

    `client` (optional httpx.Client, e.g. with a MockTransport) is used for both services unless
    `geocoder` / `overpass` are given. Network failures degrade the result, never raise."""
    res = to_module_result(vc)
    if vc.report is None or vc.status in ("skipped", "failed"):
        return res
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    geocoder = geocoder or Geocoder.from_settings(settings, client=client, check_cancel=check_cancel)
    overpass = overpass or Overpass.from_settings(settings, client=client, check_cancel=check_cancel)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []
    plans: list[_Plan] = []
    clues = {c.id: c for c in vc.report.clues}

    def guarded(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except (GeocodeError, OverpassError) as e:
            warns.append(str(e))
            return None

    # 1. geocoder queries (clue queries + searchable entities)
    for q in vc.geocode_queries()[:MAX_GEOCODE_QUERIES]:
        if q.confidence < MIN_CLUE_CONF or geocoder.blocked:
            continue
        hits = guarded(geocoder.search, q.query, bbox, countrycodes=q.country_codes or None)
        if hits is None:
            continue
        uniq = list({h.key: h for h in hits}.values())[:5]
        if not uniq:
            warns.append(f"no geocoder hit inside the bbox for {q.query!r}")
            continue
        items = [(h.geometry(), 1.0, _hit_sigma(h)) for h in uniq]
        broad = max(s for _, _, s in items) >= 1000.0
        conf = min(q.confidence, GEOCODE_CONF_CAP) / math.sqrt(len(uniq)) * (0.6 if broad else 1.0)
        lay = region_layer(spec, items, conf, f"clue_{_slug(q.source)}_geo_{len(plans):02d}",
                           description=f"geocoded {q.kind}: {q.query}",
                           meta={"source": q.source, "query": q.query, "hits": [h.name for h in uniq]})
        plans.append(_Plan(lay, q.source, "geocode", conf, len(uniq), q.query))

    # 2. OSM hints through Overpass (<= max_queries per job)
    ranked = sorted((c for c in vc.report.clues if c.osm_hints and c.confidence >= 0.3),
                    key=lambda c: -c.confidence)
    for c in ranked:
        if overpass.n_requests >= overpass.max_queries or overpass.blocked:
            break
        hint = c.osm_hints[0]
        try:
            ql = _ovp.build_ql(hint, bbox)
        except QueryRejected as e:
            warns.append(f"{c.id}: OSM hint rejected ({e})")
            continue
        got = guarded(overpass.run, ql)
        if got is None:
            continue
        feats, capped = got
        if capped:
            warns.append(f"{c.id}: OSM hint matches too many features (>= {_ovp.RESULT_CAP}); ignored")
            continue
        if not feats:
            warns.append(f"{c.id}: OSM hint matched nothing in the bbox")
            continue
        from shapely.geometry import Point

        items = [(Point(f.lon, f.lat), 1.0, OSM_SIGMA_M) for f in feats]
        conf = min(c.confidence, OSM_CONF_CAP) * min(1.0, math.sqrt(10.0 / len(feats)))
        lay = region_layer(spec, items, conf, f"clue_{_slug(c.id)}_osm_{len(plans):02d}",
                           description=f"OSM match for {c.type}: {c.value}",
                           meta={"source": c.id, "n_matches": len(feats)})
        plans.append(_Plan(lay, c.id, "osm", conf, len(feats), c.value))

    # 3. country and admin-1 implications -> boundary polygons (cached), K weighted by p
    areas: dict[tuple[str, str], Any] = {}
    n_lookups = 0

    def area(country: str, name: str | None):
        nonlocal n_lookups
        key = (country, name or "")
        if key not in areas:
            if n_lookups >= MAX_ADMIN_LOOKUPS or geocoder.blocked:
                areas[key] = None
            else:
                n_lookups += 1
                h = guarded(geocoder.admin_area, country, name)
                areas[key] = _clip(h.geometry(), bbox) if h is not None else None
        return areas[key]

    def weighted(c: Clue, entries: list[tuple[str, str | None, float]], cap: float, kind: str):
        items = []
        for country, name, p in entries:
            if p < MIN_CLUE_CONF:
                continue
            g = area(country, name)
            if g is not None:
                items.append((g, p, SOFT_EDGE_SIGMA_M))
        if not items:
            return
        conf = min(c.confidence, cap)
        lay = region_layer(spec, items, conf, f"clue_{_slug(c.id)}_{kind}_{len(plans):02d}", combine="sum",
                           description=f"{kind} implied by {c.type}: {c.value}",
                           meta={"source": c.id})
        if _flat(lay):
            warns.append(f"{c.id}: {kind} layer is flat over the bbox; dropped")
            return
        plans.append(_Plan(lay, c.id, kind, conf, len(items), c.value))

    for c in sorted(vc.report.clues, key=lambda c: -c.confidence):
        if c.confidence < MIN_CLUE_CONF:
            continue
        imp = c.implication
        weighted(c, [(x.code, None, x.p) for x in imp.countries], COUNTRY_CONF_CAP, "country")
        weighted(c, [(x.country, x.name, x.p) for x in imp.admin1], ADMIN1_CONF_CAP, "admin1")

    # 4. region hypotheses
    for r in vc.report.region_hypotheses[:MAX_REGION_LOOKUPS]:
        if r.p < 0.1 or geocoder.blocked:
            continue
        cc = [r.country] if r.country else None
        hits = guarded(geocoder.search, r.name, bbox, countrycodes=cc, limit=1)
        if not hits:
            continue
        h = hits[0]
        sig = max(_hit_sigma(h), 1000.0)
        conf = min(r.p, REGION_CONF_CAP)
        lay = region_layer(spec, [(h.geometry(), 1.0, sig)], conf, f"region_{_slug(r.name)}_{len(plans):02d}",
                           description=f"region hypothesis: {r.name}", meta={"source": "region"})
        plans.append(_Plan(lay, "region", "region", conf, 1, r.name))

    # global cap: no pile of clues may dominate
    total = sum(p.depth for p in plans)
    scale = min(1.0, MAX_TOTAL_NATS / total) if total > 0 else 1.0
    refs, info = [], []
    for p in plans:
        p.layer.weight = scale
        ref = p.layer.to_ref(out_dir / "layers" / f"{p.layer.name}.npz", relative_to=out_dir)
        refs.append(ref)
        info.append({"name": p.layer.name, "source": p.source, "kind": p.kind, "confidence": round(p.conf, 3),
                     "n": p.n, "depth_nats": round(p.depth, 3), "weight": round(scale, 3),
                     "detail": p.detail})
    (out_dir / "visual_layers.json").write_text(json.dumps({"layers": info, "warnings": warns}, indent=1))

    unused = sorted({k for c in clues.values() for k in c.implication.koppen})
    attribution = list(ATTRIBUTION)
    if geocoder.n_requests or geocoder.n_cache_hits:
        attribution.append(_geo.ATTRIBUTION)
    if overpass.n_requests or overpass.n_cache_hits:
        attribution.append(_ovp.ATTRIBUTION)
    res.layers = refs
    res.artifacts["layers_index"] = "visual_layers.json"
    res.provenance.update({
        "attribution": attribution, "layers": info, "layer_warnings": warns[:50],
        "weight_scale": round(scale, 3),
        "requests": {"nominatim": geocoder.n_requests, "nominatim_cached": geocoder.n_cache_hits,
                     "overpass": overpass.n_requests, "overpass_cached": overpass.n_cache_hits},
        "not_used": {"koppen": unused,
                     "note": "Koppen codes, driving side and place_candidates are not layered"},
    })
    if warns and any("HTTP" in w or "failed" in w or "blocked" in w or "budget" in w for w in warns):
        if res.status == "ok":
            res.status = "degraded"
            res.message = "some geocoder/Overpass requests failed; layers are partial"
    return res


# --------------------------------------------------------------------------------------- sun adapter

_NUM = r"(\d+(?:\.\d+)?)"
_COMPASS = {
    "north": 0, "north-east": 45, "northeast": 45, "east": 90, "south-east": 135, "southeast": 135,
    "south": 180, "south-west": 225, "southwest": 225, "west": 270, "north-west": 315, "northwest": 315,
}
_COMPASS_RE = "|".join(sorted(_COMPASS, key=len, reverse=True))


def parse_sun_text(text: str) -> dict[str, float]:
    """Pull numbers out of a free-text `sun_shadow` clue (value + rationale). WP15 gives no numeric
    fields, so this is a best-effort parse; keys: ratio, elevation, azimuth, rel_azimuth."""
    t = text.lower().replace("°", " deg ").replace("degrees", " deg ").replace("°", " deg ")
    out: dict[str, float] = {}
    m = (re.search(rf"shadow[^.;\d]{{0,25}}{_NUM}\s*(?:x|times)\s*(?:the\s*)?(?:object|pole|post|sign|tree|"
                   rf"person|building|its)?['s ]*height", t)
         or re.search(rf"(?:shadow|length)[^.;\d]{{0,12}}(?:/|:|to|over)[^.;\d]{{0,12}}height"
              rf"[^.;\d]{{0,20}}{_NUM}", t)
         or re.search(rf"(?:shadow|s)/h[^.;\d]{{0,20}}{_NUM}", t)
         or re.search(rf"ratio[^.;\d]{{0,25}}{_NUM}", t))
    if m and 0.05 <= float(m.group(1)) <= 50:
        out["ratio"] = float(m.group(1))
    else:
        m = re.search(rf"shadow[^.;\d]{{0,20}}{_NUM}\s*m\b[^.;]{{0,60}}?height[^.;\d]{{0,15}}{_NUM}\s*m\b", t)
        if m and float(m.group(2)) > 0 and 0.05 <= float(m.group(1)) / float(m.group(2)) <= 50:
            out["ratio"] = float(m.group(1)) / float(m.group(2))
    m = (re.search(rf"(?:sun )?(?:elevation|altitude)[^.;\d]{{0,15}}{_NUM}\s*deg", t)
         or re.search(rf"{_NUM}\s*deg\s*(?:above the horizon|sun elevation|elevation|altitude)", t))
    if m and 0 < float(m.group(1)) < 90:
        out["elevation"] = float(m.group(1))
    m = re.search(rf"azimuth[^.;\d]{{0,15}}{_NUM}\s*deg", t)
    if m:
        out["azimuth"] = float(m.group(1)) % 360
    else:
        m = re.search(rf"shadows?\s+(?:points?|pointing|falls?|falling|is cast|cast|extends?|runs?)\s+"
                      rf"(?:to(?:wards?)?\s+(?:the\s+)?)?({_COMPASS_RE})\b", t)
        if m:
            out["azimuth"] = float(_COMPASS[m.group(1)])
        elif re.search(r"shadows?[^.;]{0,30}(?:toward|towards|to) the camera", t):
            out["rel_azimuth"] = 180.0
        elif re.search(r"shadows?[^.;]{0,30}away from the camera", t):
            out["rel_azimuth"] = 0.0
        elif re.search(r"shadows?[^.;]{0,30}(?:to|toward|towards) the right", t):
            out["rel_azimuth"] = 90.0
        elif re.search(r"shadows?[^.;]{0,30}(?:to|toward|towards) the left", t):
            out["rel_azimuth"] = 270.0
    return out


def sun_clue_from_clues(vc: VisualClues, *, camera_heading_deg: Sequence[float] | float | None = None,
                        time_window=None, max_conf: float = 0.8, text_conf_factor: float = 0.7):
    """Adapter: the best-confidence `sun_shadow` clue in `clues.json` -> `visual.sun.SunClue`, or None.

    Differences from the mapping guessed in sun.py's docstring (WP15 actually emits):
    - a `sun_shadow` Clue has only free text (`value`, `rationale`), a frame `evidence` list with
      normalised bboxes (no marked shadow tip / object height, no numeric fields) and `confidence`;
      so the numbers are parsed from the text (`parse_sun_text`), and a clue without parseable
      numbers returns None (the re-rank is then skipped);
    - there is no camera heading in WP15 output: relative azimuths ("towards the camera") only count
      if the caller supplies `camera_heading_deg`; compass directions in the text are used as world
      azimuths (shadow direction);
    - confidence is capped at `max_conf` and scaled by `text_conf_factor` because parsed text numbers
      are rough; `time_window` (tz-aware datetimes) must come from the caller (acoustic/metadata
      time), the clue's frame `t_s` are video-relative and carry no absolute time.
    """
    from .sun import SunClue

    if vc.report is None:
        return None
    for c in sorted((c for c in vc.report.clues if c.type == "sun_shadow"), key=lambda c: -c.confidence):
        f = parse_sun_text(f"{c.value}. {c.rationale}")
        kw: dict[str, Any] = {}
        if "ratio" in f:
            kw["shadow_length_over_height"] = f["ratio"]
        elif "elevation" in f:
            kw["sun_elevation_deg"] = f["elevation"]
        if "azimuth" in f:
            kw["shadow_azimuth_deg"] = f["azimuth"]
        elif "rel_azimuth" in f and camera_heading_deg is not None:
            kw["shadow_rel_azimuth_deg"] = f["rel_azimuth"]
            kw["camera_heading_deg"] = camera_heading_deg
        if not kw:
            continue
        conf = min(c.confidence, max_conf) * text_conf_factor
        return SunClue(confidence=conf, time_window=time_window, **kw)
    return None
