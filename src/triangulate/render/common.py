"""Shared helpers for the evidence renderers (WP20): colormaps, PNG output, Web Mercator grids.

Images are built with numpy and Pillow. matplotlib is used only for its colormap lookup tables
(perceptual maps such as viridis/magma), never for figures.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..core.geo import BBox
from ..core.types import GridSpec
from ..fusion.grid import grid_extent_xy, xy_to_latlon

__all__ = ["R_MERC", "colormap_rgba", "to_uint8", "write_png", "write_json", "safe_name",
           "mercator_latlon_grid", "spec_bounds_latlon"]

R_MERC = 6378137.0  # EPSG:3857 sphere radius (m)


def colormap_rgba(name: str, values: np.ndarray) -> np.ndarray:
    """Look up `name` (a matplotlib colormap) at `values` in [0, 1]. Returns float RGBA (..., 4)."""
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return np.asarray(cmap(np.clip(np.asarray(values, dtype=float), 0.0, 1.0)), dtype=float)


def to_uint8(rgba: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint8)


def write_png(arr: np.ndarray, path: str | Path) -> Path:
    """Write a uint8 (H, W, 3) RGB or (H, W, 4) RGBA array as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).save(path, format="PNG", optimize=True)
    return path


def write_json(obj: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, allow_nan=False))
    return path


def safe_name(name: str) -> str:
    """File-system safe version of a layer or tuple name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "unnamed"


def mercator_latlon_grid(bounds: BBox, width_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre (lat, lon) arrays of an EPSG:3857 image covering `bounds` = (S, W, N, E).

    Row 0 is the north edge. The image has `width_px` columns and a height that keeps Mercator
    pixels square. Returns (LAT, LON), each (h, w).
    """
    s, w, n, e = (float(v) for v in bounds)
    if not (s < n and w < e):
        raise ValueError("bounds must be (S, W, N, E) with S < N and W < E")
    mx0, mx1 = R_MERC * math.radians(w), R_MERC * math.radians(e)
    my0 = R_MERC * math.log(math.tan(math.pi / 4 + math.radians(s) / 2))
    my1 = R_MERC * math.log(math.tan(math.pi / 4 + math.radians(n) / 2))
    h = max(1, round(width_px * (my1 - my0) / (mx1 - mx0)))
    px = mx0 + (np.arange(width_px) + 0.5) * (mx1 - mx0) / width_px
    py = my1 - (np.arange(h) + 0.5) * (my1 - my0) / h
    lon = np.degrees(px / R_MERC)
    lat = np.degrees(2.0 * np.arctan(np.exp(py / R_MERC)) - math.pi / 2)
    LON, LAT = np.meshgrid(lon, lat)
    return LAT, LON


def spec_bounds_latlon(spec: GridSpec) -> BBox:
    """Lon/lat bounding box (S, W, N, E) of a GridSpec's full LAEA extent."""
    x0, y0, x1, y1 = grid_extent_xy(spec)
    XX, YY = np.meshgrid(np.linspace(x0, x1, 33), np.linspace(y0, y1, 33))
    lat, lon = xy_to_latlon(spec, XX, YY)
    return float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())
