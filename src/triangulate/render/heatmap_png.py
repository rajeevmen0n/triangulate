"""Posterior heatmap and per-layer band maps in Web Mercator (plan 2.2, WP20).

Both renders are RGBA PNGs on an EPSG:3857 pixel grid, for Leaflet `ImageOverlay`. Row 0 is north.
Bounds are (S, W, N, E) in degrees; the JSON sidecar also gives them in Leaflet order
[[S, W], [N, E]].

- Posterior heatmap: colour = a perceptual colormap (viridis) over the top `dynamic_range_nats`
  of ln density; alpha = 0.9 * sqrt(colour value), so low density fades out and cells outside the
  domain are transparent.
- Band map (one `fusion.grid.Layer`, e.g. the acoustic evidence for one flight tuple): values are
  the layer minus its fill (its "no information" baseline). A diverging colormap is centred on 0;
  alpha grows with |value|. Red = supports, blue = against, relative to the baseline.

Public API
    render_posterior_heatmap(post, out_png, ...) -> info dict   (PNG + out_png.json)
    write_heatmap(post, workdir) -> {"heatmap_png": ..., "heatmap_json": ...}
    render_band_map(layer, out_png, ...) -> info dict
    render_band_maps(layers, out_dir, ...) -> list of info dicts (one PNG per layer)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.geo import BBox
from ..fusion.combine import Posterior
from ..fusion.export import web_mercator_raster
from ..fusion.grid import Layer
from .common import (
    colormap_rgba,
    mercator_latlon_grid,
    safe_name,
    spec_bounds_latlon,
    to_uint8,
    write_json,
    write_png,
)

__all__ = ["HEATMAP_FORMAT", "BAND_FORMAT", "density_rgba", "render_posterior_heatmap", "write_heatmap",
           "render_band_map", "render_band_maps"]

HEATMAP_FORMAT = "triangulate.heatmap/1"
BAND_FORMAT = "triangulate.band_map/1"
DEFAULT_CMAP = "viridis"
DEFAULT_DIVERGING = "RdBu_r"


def _image_info(fmt: str, png: Path, bounds: BBox, width_px: int, height_px: int, **extra: Any) -> dict:
    s, w, n, e = bounds
    image = {"file": png.name, "width_px": int(width_px), "height_px": int(height_px)}
    return {"format": fmt, "image": image,
            "crs": "EPSG:3857", "bounds": {"south": s, "west": w, "north": n, "east": e},
            "leaflet_bounds": [[s, w], [n, e]], **extra}


def density_rgba(logv: np.ndarray, dynamic_range_nats: float = 8.0, cmap: str = DEFAULT_CMAP) -> np.ndarray:
    """ln density -> uint8 RGBA. v = 1 at the peak, 0 at `dynamic_range_nats` below it or non-finite."""
    if dynamic_range_nats <= 0:
        raise ValueError("dynamic_range_nats must be positive")
    logv = np.asarray(logv, dtype=float)
    fin = np.isfinite(logv)
    top = float(logv[fin].max()) if fin.any() else 0.0
    v = np.where(fin, 1.0 + (np.where(fin, logv, top) - top) / dynamic_range_nats, 0.0)
    v = np.clip(v, 0.0, 1.0)
    rgba = colormap_rgba(cmap, v)
    rgba[..., 3] = 0.9 * np.sqrt(v)
    return to_uint8(rgba)


def render_posterior_heatmap(post: Posterior, out_png: str | Path, *, width_px: int = 1024,
                             bounds: BBox | None = None, cmap: str = DEFAULT_CMAP,
                             dynamic_range_nats: float = 8.0, json_sidecar: bool = True) -> dict[str, Any]:
    """Posterior density as a Web Mercator RGBA PNG. `bounds` defaults to the domain's lon/lat box."""
    out_png = Path(out_png)
    grid = web_mercator_raster(post, width_px=width_px, bounds=bounds)
    logv = grid["log_density"]
    write_png(density_rgba(logv, dynamic_range_nats, cmap), out_png)
    fin = np.isfinite(logv)
    h, w = logv.shape
    info = _image_info(HEATMAP_FORMAT, out_png, grid["bounds"], w, h, crs_note="EPSG:3857 image, row 0 north",
                       units="ln(probability per m^2)", colour_scale={
                           "cmap": cmap, "dynamic_range_nats": dynamic_range_nats,
                           "top_ln_density": float(logv[fin].max()) if fin.any() else None,
                           "alpha": "0.9*sqrt(value); value = 1 + (ln p - top) / range, clipped"},
                       cover_fraction=float(fin.mean()))
    if json_sidecar:
        write_json(info, out_png.with_suffix(".json"))
    return info


def write_heatmap(post: Posterior, workdir: str | Path, *, width_px: int = 1024) -> dict[str, str]:
    """Job-workspace outputs: heatmap.png + heatmap.json. Returns artifact name -> relative path."""
    wd = Path(workdir)
    render_posterior_heatmap(post, wd / "heatmap.png", width_px=width_px)
    return {"heatmap_png": "heatmap.png", "heatmap_json": "heatmap.json"}


def render_band_map(layer: Layer, out_png: str | Path, *, bounds: BBox | None = None, width_px: int = 1024,
                    cmap: str = DEFAULT_DIVERGING, json_sidecar: bool = True) -> dict[str, Any]:
    """One layer's evidence (minus its fill baseline) as a Web Mercator RGBA PNG."""
    out_png = Path(out_png)
    bnd = tuple(bounds) if bounds is not None else spec_bounds_latlon(layer.spec)
    LAT, LON = mercator_latlon_grid(bnd, width_px)
    base = float(layer.fill) if np.isfinite(layer.fill) else 0.0
    ell = np.asarray(layer.value_at_latlon(LAT, LON), dtype=float) - base
    fin = np.isfinite(ell)
    m = float(np.abs(ell[fin]).max()) if fin.any() else 0.0
    scale = m if m > 1e-12 else 1.0
    val = np.where(fin, np.clip(ell / scale, -1.0, 1.0), 0.0)
    rgba = colormap_rgba(cmap, 0.5 + 0.5 * val)
    rgba[..., 3] = np.where(fin, 0.85 * np.sqrt(np.abs(val)), 0.0)
    h, w = ell.shape
    write_png(to_uint8(rgba), out_png)
    info = _image_info(BAND_FORMAT, out_png, bnd, w, h, layer={
        "name": layer.name, "module": layer.module, "description": layer.description,
        "weight": layer.weight, "meta": {k: v for k, v in layer.meta.items() if _jsonable(v)}},
        baseline_subtracted=base, max_abs_value=m, colour_scale={"cmap": cmap, "centre": 0.0},
        units="log likelihood ratio vs baseline")
    if json_sidecar:
        write_json(info, out_png.with_suffix(".json"))
    return info


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def render_band_maps(layers: Sequence[Layer], out_dir: str | Path, *, bounds: BBox | None = None,
                     width_px: int = 1024, prefix: str = "band") -> list[dict[str, Any]]:
    """One band map per layer, named `<prefix>_<layer name>.png` in `out_dir`.

    For a per-flight-tuple view, build one Layer per tuple (the caller decides how) and pass them here.
    """
    out = Path(out_dir)
    infos = []
    for lay in layers:
        stem = safe_name(f"{prefix}_{lay.name}")
        infos.append(render_band_map(lay, out / f"{stem}.png", bounds=bounds, width_px=width_px))
        infos[-1]["file_stem"] = stem
    return infos
