"""Fusion entry point and exports (plan 1.3 outputs, 1.4 workspace files, WP12).

`run_fusion(...)` -> `FusionResult` (posterior + candidates + HPD regions + attribution), and
`write_outputs(result, workdir)` writes:

- `posterior.tif`: GeoTIFF in the job's LAEA CRS at the base level, north-up. Band 1 = ln posterior
  density (ln probability per m^2; NaN outside the domain). Band 2 = HPD level of the cell (0..1;
  the 90 % HPD region is band2 <= 0.9).
- `posterior.npz`: the full multi-resolution posterior (`Posterior.load`), including fine tiles.
- `candidates.geojson`: FeatureCollection in lon/lat (RFC 7946). Point features
  (`properties.kind = "candidate"`) carry rank, log_density, prob_mass, contributions, flights, t_s,
  margin_to_runner_up, hpd_level, notes, evidence. (Multi)Polygon features (`kind = "hpd"`) carry
  mass, log_threshold and area_m2. Foreign member `attribution` lists data notices.
- `fusion.json`: `FusionResult.summary()` (for the job's results.json).

`web_mercator_raster` resamples the posterior onto an EPSG:3857 pixel grid for Leaflet image
overlays (WP20 renders it).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.geo import BBox
from ..core.types import GRID_LEVELS_M, GridSpec, ModuleResult
from .candidates import (
    FusionCandidate,
    HPDRegion,
    compute_hpd,
    credible_level,
    find_candidates,
    pinned_candidate,
)
from .combine import Posterior, collect_attribution, fuse, layers_from_results
from .grid import Layer, laea_wkt, north_up_transform, xy_to_latlon

__all__ = ["FusionResult", "run_fusion", "write_geotiff", "candidates_geojson", "write_candidates_geojson",
           "web_mercator_raster", "write_outputs", "geometry_to_lonlat"]


@dataclass
class FusionResult:
    posterior: Posterior
    candidates: list[FusionCandidate]
    hpd: list[HPDRegion]
    attribution: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    module_status: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """JSON-able dict for results.json."""
        post = self.posterior
        return {
            "grid": {**post.spec.model_dump(), "base_cell_m": GRID_LEVELS_M[post.base_level],
                     "fine_level": post.fine_level,
                     "fine_cell_m": GRID_LEVELS_M[post.fine_level] if post.fine_level is not None else None,
                     "n_fine_tiles": len(post.tiles), "crs_wkt": laea_wkt(post.spec)},
            "weights": post.weights,
            "layers": [{"name": lay.name, "module": lay.module, "weight": w, "fill": lay.fill,
                        "levels": lay.levels, "description": lay.description} for lay, w in post.layers],
            "module_status": self.module_status,
            "candidates": [_cand_dict(c) for c in self.candidates],
            "hpd": [{"mass": r.mass, "log_threshold": _num(r.log_threshold), "area_m2": r.area_m2,
                     "area_km2": r.area_m2 / 1e6, "actual_mass": r.actual_mass} for r in self.hpd],
            "attribution": self.attribution,
            "notes": self.notes + post.warnings,
        }


def _num(v: float) -> float | None:
    return float(v) if v is not None and math.isfinite(v) else None


def _cand_dict(c: FusionCandidate) -> dict[str, Any]:
    d = json.loads(c.model_dump_json())
    for k in ("log_density", "margin_to_runner_up", "rank_score"):
        if isinstance(d.get(k), str):  # pydantic serialises inf as a string in some modes
            d[k] = None
    return d


def run_fusion(spec: GridSpec, *, layers: Sequence[Layer] = (), module_results: Sequence[ModuleResult] = (),
               root: str | Path | None = None, domain: BBox | Any | None = None,
               weights: dict[str, float] | None = None, prior: Layer | None = None, k: int = 10,
               nms_radius_m: float = 300.0, rank_by: str = "mass", hpd_masses: Sequence[float] = (0.5, 0.9),
               pinned: Sequence[tuple[float, float, str]] = (), fine_level: int | None = None,
               ) -> FusionResult:
    """One call for the pipeline (WP18): load layers, fuse, find candidates, compute HPD.

    `layers`: in-memory layers; `module_results`: their LayerRefs are loaded (paths relative to
    `root`). `pinned`: (lat, lon, note) points shown as candidate #0 (e.g. trusted GPS).
    """
    loaded, notes = layers_from_results(module_results, root)
    all_layers = list(layers) + loaded
    post = fuse(spec, all_layers, weights=weights, domain=domain, prior=prior, fine_level=fine_level)
    cands = find_candidates(post, k=k, nms_radius_m=nms_radius_m, rank_by=rank_by)
    pins = [pinned_candidate(post, lat, lon, note, nms_radius_m) for lat, lon, note in pinned]
    hpd = compute_hpd(post, hpd_masses)
    status: dict[str, str] = {r.module: r.status for r in module_results}
    for lay in layers:
        status.setdefault(lay.module, "ok")
    for m in ("acoustic", "visual", "metadata"):
        status.setdefault(m, "absent")
    attribution = collect_attribution(module_results)
    for lay in all_layers:
        a = lay.meta.get("attribution")
        for s in [a] if isinstance(a, str) else (a or []):
            if s not in attribution:
                attribution.append(s)
    if len(cands) >= 1 and cands[0].margin_to_runner_up is not None and cands[0].margin_to_runner_up < 0.5:
        notes.append("low margin between the top two candidates")
    return FusionResult(posterior=post, candidates=pins + cands, hpd=hpd, attribution=attribution,
                        notes=notes, module_status=status)


# ------------------------------------------------------------------------------------------------
# GeoTIFF
# ------------------------------------------------------------------------------------------------


def write_geotiff(post: Posterior, path: str | Path, level: int | None = None,
                  window: tuple[int, int, int, int] | None = None, attribution: Sequence[str] = ()) -> Path:
    """Write the posterior (2 bands, see module docstring) at `level` (default base). For levels
    finer than the base, pass a `window` (ix0, iy0, nx, ny) at that level to bound the size."""
    import rasterio

    lv = post.base_level if level is None else level
    if lv <= post.base_level:
        arr = post.at_level(lv)
        ix0, iy0 = 0, 0
        if window is not None:
            ix0, iy0, nx, ny = window
            arr = arr[iy0: iy0 + ny, ix0: ix0 + nx]
    else:
        if window is None:
            nx, ny = post.spec.shape(lv)
            window = (0, 0, nx, ny)
        ix0, iy0 = window[0], window[1]
        arr = post.field.render(lv, *window, aggregate=True)
    lvl = credible_level(post, arr.ravel()).reshape(arr.shape)
    band1 = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)[::-1]
    band2 = np.where(np.isfinite(arr), lvl, np.nan).astype(np.float32)[::-1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=2, dtype="float32",
                       crs=laea_wkt(post.spec), transform=north_up_transform(post.spec, lv, ix0, iy0, ny),
                       nodata=float("nan"), compress="deflate") as dst:
        dst.write(band1, 1)
        dst.write(band2, 2)
        dst.set_band_description(1, "ln posterior density (per m^2)")
        dst.set_band_description(2, "HPD level (0..1)")
        dst.update_tags(units="ln(probability per m^2)", cell_m=str(GRID_LEVELS_M[lv]),
                        attribution="; ".join(attribution))
    return path


# ------------------------------------------------------------------------------------------------
# GeoJSON
# ------------------------------------------------------------------------------------------------


def geometry_to_lonlat(spec: GridSpec, geom_xy, simplify_m: float | None = None):
    """LAEA shapely geometry -> lon/lat shapely geometry."""
    from shapely.ops import transform

    g = geom_xy.simplify(simplify_m, preserve_topology=True) if simplify_m else geom_xy

    def inv(x, y, z=None):
        lat, lon = xy_to_latlon(spec, x, y)
        return lon, lat

    return transform(inv, g)


def candidates_geojson(result: FusionResult, simplify_m: float | None = None) -> dict[str, Any]:
    from shapely.geometry import mapping

    feats: list[dict[str, Any]] = []
    for c in result.candidates:
        props = _cand_dict(c)
        props["kind"] = "candidate"
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                      "properties": props})
    spec = result.posterior.spec
    for r in result.hpd:
        if r.geometry_xy is None or r.geometry_xy.is_empty:
            continue
        g = geometry_to_lonlat(spec, r.geometry_xy, simplify_m)
        feats.append({"type": "Feature", "geometry": mapping(g),
                      "properties": {"kind": "hpd", "mass": r.mass, "log_threshold": _num(r.log_threshold),
                                     "area_m2": r.area_m2, "actual_mass": r.actual_mass}})
    return {"type": "FeatureCollection", "features": feats, "attribution": result.attribution}


def write_candidates_geojson(result: FusionResult, path: str | Path, simplify_m: float | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidates_geojson(result, simplify_m), default=str))
    return path


# ------------------------------------------------------------------------------------------------
# Web Mercator raster (for Leaflet ImageOverlay)
# ------------------------------------------------------------------------------------------------

_R_MERC = 6378137.0


def web_mercator_raster(post: Posterior, width_px: int = 1024, bounds: BBox | None = None,
                        ) -> dict[str, Any]:
    """Sample the posterior on an EPSG:3857 pixel grid.

    Returns {"log_density": (h, w) float64, row 0 = north, -inf outside the domain;
    "bounds": (S, W, N, E) of the image edges; "crs": "EPSG:3857"}. Default bounds: the domain's
    lon/lat bounding box.
    """
    if bounds is None:
        g = geometry_to_lonlat(post.spec, post.domain_xy) if post.domain_xy is not None else None
        if g is None:
            raise ValueError("no domain; pass bounds")
        w, s, e, n = g.bounds
    else:
        s, w, n, e = bounds
    mx0, mx1 = _R_MERC * math.radians(w), _R_MERC * math.radians(e)
    my0 = _R_MERC * math.log(math.tan(math.pi / 4 + math.radians(s) / 2))
    my1 = _R_MERC * math.log(math.tan(math.pi / 4 + math.radians(n) / 2))
    h = max(1, round(width_px * (my1 - my0) / (mx1 - mx0)))
    px = mx0 + (np.arange(width_px) + 0.5) * (mx1 - mx0) / width_px
    py = my1 - (np.arange(h) + 0.5) * (my1 - my0) / h
    lon = np.degrees(px / _R_MERC)
    lat = np.degrees(2 * np.arctan(np.exp(py / _R_MERC)) - math.pi / 2)
    LON, LAT = np.meshgrid(lon, lat)
    vals = post.value_at_latlon(LAT, LON)
    return {"log_density": vals, "bounds": (s, w, n, e), "crs": "EPSG:3857"}


# ------------------------------------------------------------------------------------------------
# All outputs
# ------------------------------------------------------------------------------------------------


def write_outputs(result: FusionResult, workdir: str | Path, simplify_m: float | None = None,
                  ) -> dict[str, str]:
    """Write posterior.tif, posterior.npz, candidates.geojson, fusion.json into `workdir`.
    Returns {artifact name: path relative to workdir} (for ModuleResult/job artifacts)."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    write_geotiff(result.posterior, wd / "posterior.tif", attribution=result.attribution)
    result.posterior.save(wd / "posterior.npz")
    write_candidates_geojson(result, wd / "candidates.geojson", simplify_m)
    (wd / "fusion.json").write_text(json.dumps(result.summary(), default=str, indent=1))
    return {"posterior_tif": "posterior.tif", "posterior_npz": "posterior.npz",
            "candidates_geojson": "candidates.geojson", "fusion_json": "fusion.json"}
