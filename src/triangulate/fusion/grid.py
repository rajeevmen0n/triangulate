"""Fusion grid: cell ids, multi-resolution log-value fields, and module layers (plan 1.3, WP12).

This module has two parts.

**Part 1: cell ids and coordinates.** It is self-contained and stable; WP09 and others may rely on it.
It depends only on `core.types.GridSpec` / `GRID_LEVELS_M` and `core.geo`.

- A grid is a `GridSpec`: a Lambert Azimuthal Equal-Area (LAEA) frame centred on (lat0, lon0) with
  origin (x0, y0) in LAEA metres. Every level shares the same origin.
- Level `lv` has cell size `GRID_LEVELS_M[lv]` = 1600, 800, 400, 200, 100, 50, 25 m (lv = 0..6).
- A cell id is `(level, ix, iy)`. The cell covers `x in [x0 + ix*s, x0 + (ix+1)*s)` and
  `y in [y0 + iy*s, y0 + (iy+1)*s)`. iy grows northwards. ix/iy may be negative (halo cells).
- Levels nest exactly: the parent of (lv, ix, iy) at level lv-1 is (lv-1, ix // 2, iy // 2).
- `pack_cells` turns ids into one int64 key (for parquet/duckdb joins); `unpack_cells` reverses it.
- Tiles: `TILE` x `TILE` (256 x 256) cells of one level. Tile (tx, ty) holds ix in [tx*256, tx*256+256).
- LAEA is equal-area, so every cell of a level has area s*s m^2.

**Part 2: fields and layers.**

- `LogField`: a piecewise-constant field of natural-log values over the grid, stored as `Patch`es.
  A patch is a 2-D array at one level. `values[j, i]` is cell (ix0 + i, iy0 + j); row 0 is the
  southern row. NaN in a patch means "transparent": the next coarser patch shows through, or `fill`
  where nothing covers the point. Where patches overlap, the finer level wins (equal levels: the
  later patch wins). Values are point (intensive) quantities, so a coarse value applies unchanged to
  every child cell. Aggregating to a coarser level uses log-mean-exp, which conserves mass.
- `Layer`: a `LogField` that a module hands to fusion, plus name, module, per-layer weight, optional
  per-cell attribute table, and npz serialisation (`save`/`load`, `to_ref` -> `core.types.LayerRef`).

See `fusion/combine.py` for the layer contract (log-likelihood conventions, weights, helpers).
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pyproj import Transformer

from ..core.geo import laea_crs
from ..core.types import GRID_LEVELS_M, CellId, GridSpec, LayerRef

__all__ = [
    "GRID_LEVELS_M", "CellId", "GridSpec", "TILE", "N_LEVELS", "FINEST_LEVEL",
    "cell_size", "level_for_size", "coarsest_level_at_most", "pack_cells", "unpack_cells",
    "pack_cell", "unpack_cell", "to_level", "cell_centers_xy", "xy_to_cells", "latlon_to_xy",
    "xy_to_latlon", "latlon_to_cells", "cell_centers_latlon", "tile_of", "tile_window",
    "window_for_xy_box", "grid_extent_xy", "same_frame", "laea_wkt", "north_up_transform",
    "Patch", "LogField", "Layer", "logmeanexp_downsample", "rasterize_geom",
]

TILE = 256
N_LEVELS = len(GRID_LEVELS_M)
FINEST_LEVEL = N_LEVELS - 1


# ================================================================================================
# Part 1: cell ids and coordinates (stable API)
# ================================================================================================


def cell_size(level: int) -> int:
    """Cell edge in metres at a level."""
    return GRID_LEVELS_M[level]


def level_for_size(size_m: float) -> int:
    """Exact level index for a cell size (e.g. 200 -> 3). Raises ValueError if not a grid size."""
    for lv, s in enumerate(GRID_LEVELS_M):
        if abs(s - size_m) < 1e-9:
            return lv
    raise ValueError(f"{size_m} m is not a grid level size {GRID_LEVELS_M}")


def coarsest_level_at_most(size_m: float) -> int:
    """Coarsest level whose cell size is <= size_m (the finest level if none is that small)."""
    for lv, s in enumerate(GRID_LEVELS_M):
        if s <= size_m:
            return lv
    return FINEST_LEVEL


_LV_SHIFT = 58
_IX_SHIFT = 29
_BIAS = 1 << 28
_MASK = (1 << 29) - 1


def pack_cells(level, ix, iy) -> np.ndarray:
    """(level, ix, iy) -> int64 key. Vectorised; |ix|, |iy| < 2**28 (6700 km at 25 m)."""
    lv = np.asarray(level, dtype=np.int64)
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    if np.any(np.abs(ix) >= _BIAS) or np.any(np.abs(iy) >= _BIAS):
        raise ValueError("cell index out of packable range")
    return (lv << _LV_SHIFT) | ((ix + _BIAS) << _IX_SHIFT) | (iy + _BIAS)


def unpack_cells(key) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """int64 key -> (level, ix, iy) arrays."""
    k = np.asarray(key, dtype=np.int64)
    return k >> _LV_SHIFT, ((k >> _IX_SHIFT) & _MASK) - _BIAS, (k & _MASK) - _BIAS


def pack_cell(cell: CellId) -> int:
    return int(pack_cells(cell[0], cell[1], cell[2]))


def unpack_cell(key: int) -> CellId:
    lv, ix, iy = unpack_cells(key)
    return int(lv), int(ix), int(iy)


def to_level(level: int, ix, iy, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Map cells to another level. Coarser target: the containing (ancestor) cell. Finer target: the
    south-west child; the children span `[ix*k, ix*k + k)` with k = 2**(target - level)."""
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    if target <= level:
        sh = level - target
        return ix >> sh, iy >> sh  # arithmetic shift == floor division, also for negatives
    sh = target - level
    return ix << sh, iy << sh


def cell_centers_xy(spec: GridSpec, level: int, ix, iy) -> tuple[np.ndarray, np.ndarray]:
    s = GRID_LEVELS_M[level]
    return spec.x0 + (np.asarray(ix) + 0.5) * s, spec.y0 + (np.asarray(iy) + 0.5) * s


def xy_to_cells(spec: GridSpec, x, y, level: int) -> tuple[np.ndarray, np.ndarray]:
    s = GRID_LEVELS_M[level]
    ix = np.floor((np.asarray(x, dtype=np.float64) - spec.x0) / s).astype(np.int64)
    iy = np.floor((np.asarray(y, dtype=np.float64) - spec.y0) / s).astype(np.int64)
    return ix, iy


@lru_cache(maxsize=64)
def _fwd(lat0: float, lon0: float) -> Transformer:
    return Transformer.from_crs("EPSG:4326", laea_crs(lat0, lon0), always_xy=True)


@lru_cache(maxsize=64)
def _inv(lat0: float, lon0: float) -> Transformer:
    return Transformer.from_crs(laea_crs(lat0, lon0), "EPSG:4326", always_xy=True)


def latlon_to_xy(spec: GridSpec, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    x, y = _fwd(spec.lat0, spec.lon0).transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def xy_to_latlon(spec: GridSpec, x, y) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = _inv(spec.lat0, spec.lon0).transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lat), np.asarray(lon)


def latlon_to_cells(spec: GridSpec, lat, lon, level: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = latlon_to_xy(spec, lat, lon)
    return xy_to_cells(spec, x, y, level)


def cell_centers_latlon(spec: GridSpec, level: int, ix, iy) -> tuple[np.ndarray, np.ndarray]:
    return xy_to_latlon(spec, *cell_centers_xy(spec, level, ix, iy))


def tile_of(ix, iy) -> tuple[np.ndarray, np.ndarray]:
    """Tile coordinates (same level) of cells."""
    return np.asarray(ix, dtype=np.int64) // TILE, np.asarray(iy, dtype=np.int64) // TILE


def tile_window(tx: int, ty: int) -> tuple[int, int, int, int]:
    """(ix0, iy0, nx, ny) of a tile at its level."""
    return tx * TILE, ty * TILE, TILE, TILE


def window_for_xy_box(spec: GridSpec, level: int, xmin: float, ymin: float, xmax: float, ymax: float,
                      ) -> tuple[int, int, int, int]:
    """Smallest cell window (ix0, iy0, nx, ny) at `level` covering the LAEA box (not clipped)."""
    s = GRID_LEVELS_M[level]
    ix0 = int(np.floor((xmin - spec.x0) / s))
    iy0 = int(np.floor((ymin - spec.y0) / s))
    ix1 = int(np.ceil((xmax - spec.x0) / s))
    iy1 = int(np.ceil((ymax - spec.y0) / s))
    return ix0, iy0, max(ix1 - ix0, 1), max(iy1 - iy0, 1)


def grid_extent_xy(spec: GridSpec, level: int | None = None) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of the full cell array at `level` (default: base level)."""
    lv = spec.base_level if level is None else level
    nx, ny = spec.shape(lv)
    s = GRID_LEVELS_M[lv]
    return spec.x0, spec.y0, spec.x0 + nx * s, spec.y0 + ny * s


def same_frame(a: GridSpec, b: GridSpec) -> bool:
    """True when both specs define identical cells (same LAEA centre and origin)."""
    return all(abs(getattr(a, k) - getattr(b, k)) < 1e-6 for k in ("lat0", "lon0", "x0", "y0"))


def laea_wkt(spec: GridSpec) -> str:
    return laea_crs(spec.lat0, spec.lon0).to_wkt()


def north_up_transform(spec: GridSpec, level: int, ix0: int, iy0: int, ny: int):
    """rasterio Affine for an array of window (ix0, iy0, ny rows) after flipping it north-up
    (`values[::-1]`)."""
    from affine import Affine

    s = GRID_LEVELS_M[level]
    return Affine(s, 0.0, spec.x0 + ix0 * s, 0.0, -s, spec.y0 + (iy0 + ny) * s)


def rasterize_geom(spec: GridSpec, geom_xy, level: int, ix0: int, iy0: int, nx: int, ny: int,
                   ) -> np.ndarray:
    """Bool mask (ny, nx), row 0 = south, of cells whose centre lies inside a LAEA-metre geometry."""
    import rasterio.features as rf

    if geom_xy is None or geom_xy.is_empty:
        return np.zeros((ny, nx), dtype=bool)
    m = rf.rasterize([(geom_xy, 1)], out_shape=(ny, nx), fill=0, dtype="uint8",
                     transform=north_up_transform(spec, level, ix0, iy0, ny))
    return m[::-1].astype(bool)


# ================================================================================================
# Part 2: multi-resolution log-value fields
# ================================================================================================


def logmeanexp_downsample(a: np.ndarray, k: int) -> np.ndarray:
    """Aggregate (ny, nx) log values by k x k blocks with log-mean-exp (mass-conserving).
    ny and nx must be multiples of k. All -inf blocks stay -inf."""
    if k == 1:
        return np.asarray(a, dtype=np.float64)
    ny, nx = a.shape
    b = np.asarray(a, dtype=np.float64).reshape(ny // k, k, nx // k, k)
    m = b.max(axis=(1, 3))
    ms = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.exp(b - ms[:, None, :, None]).sum(axis=(1, 3))
        out = ms + np.log(s) - 2.0 * np.log(k)
    return np.where(np.isneginf(m), -np.inf, out)


@dataclass
class Patch:
    """A rectangular block of cells at one level. `values[j, i]` is cell (ix0 + i, iy0 + j);
    row 0 is the southern row. NaN = transparent."""

    level: int
    ix0: int
    iy0: int
    values: np.ndarray

    @property
    def ny(self) -> int:
        return int(self.values.shape[0])

    @property
    def nx(self) -> int:
        return int(self.values.shape[1])

    def window_at(self, level: int) -> tuple[int, int, int, int]:
        """Covering window (ix0, iy0, nx, ny) of this patch at another level (rounded outwards)."""
        if level >= self.level:
            k = 1 << (level - self.level)
            return self.ix0 * k, self.iy0 * k, self.nx * k, self.ny * k
        sh = self.level - level
        ix0, iy0 = self.ix0 >> sh, self.iy0 >> sh
        ix1 = -((-(self.ix0 + self.nx)) >> sh)  # ceil division
        iy1 = -((-(self.iy0 + self.ny)) >> sh)
        return ix0, iy0, ix1 - ix0, iy1 - iy0


def _intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1 - ix0, iy1 - iy0


class LogField:
    """Piecewise-constant field of natural-log values on a `GridSpec` frame (see module docstring).

    `fill` is the value wherever no patch covers a point (it may be -inf, never NaN).
    """

    def __init__(self, spec: GridSpec, fill: float = 0.0, patches: Iterable[Patch] | None = None):
        if np.isnan(fill):
            raise ValueError("fill must not be NaN")
        self.spec = spec
        self.fill = float(fill)
        self.patches: list[Patch] = []
        for p in patches or ():
            self.add_patch(p.level, p.ix0, p.iy0, p.values)

    # --- construction --------------------------------------------------------------------------
    def add_patch(self, level: int, ix0: int, iy0: int, values: np.ndarray) -> Patch:
        v = np.asarray(values)
        if v.ndim != 2 or v.size == 0:
            raise ValueError("patch values must be a non-empty 2-D array (ny, nx)")
        if not 0 <= level < N_LEVELS:
            raise ValueError(f"bad level {level}")
        if v.dtype not in (np.float32, np.float64):
            v = v.astype(np.float64)
        if np.isposinf(v).any():
            raise ValueError("patch values must not be +inf")
        p = Patch(int(level), int(ix0), int(iy0), v)
        self.patches.append(p)
        return p

    def add_dense(self, values: np.ndarray, level: int | None = None) -> Patch:
        """Add a patch covering the whole grid at `level` (default: base level); shape (ny, nx)."""
        lv = self.spec.base_level if level is None else level
        nx, ny = self.spec.shape(lv)
        if values.shape != (ny, nx):
            raise ValueError(f"dense values must have shape {(ny, nx)} at level {lv}, got {values.shape}")
        return self.add_patch(lv, 0, 0, values)

    def add_cells(self, level: int, ix, iy, values, reduce: str = "logsumexp") -> list[Patch]:
        """Add sparse cells at `level`, grouped into TILE-aligned patches (other cells NaN).
        Duplicate cells are combined with `reduce`: "logsumexp" | "max" | "sum" | "last"."""
        ix = np.asarray(ix, dtype=np.int64).ravel()
        iy = np.asarray(iy, dtype=np.int64).ravel()
        val = np.asarray(values, dtype=np.float64).ravel()
        if not (ix.size == iy.size == val.size):
            raise ValueError("ix, iy, values must have equal length")
        if ix.size == 0:
            return []
        keys = pack_cells(level, ix, iy)
        uk, inv = np.unique(keys, return_inverse=True)
        if uk.size != keys.size:
            val = _reduce_dups(val, inv, uk.size, reduce)
        else:
            order = np.argsort(inv)
            val = val[order]
        _, ux, uy = unpack_cells(uk)
        tx, ty = tile_of(ux, uy)
        tkey = (tx + _BIAS) * (1 << 30) + (ty + _BIAS)
        out = []
        for tk in np.unique(tkey):
            sel = tkey == tk
            t_x, t_y = int(tx[sel][0]), int(ty[sel][0])
            arr = np.full((TILE, TILE), np.nan, dtype=np.float64)
            arr[uy[sel] - t_y * TILE, ux[sel] - t_x * TILE] = val[sel]
            out.append(self.add_patch(level, t_x * TILE, t_y * TILE, arr))
        return out

    # --- queries -------------------------------------------------------------------------------
    @property
    def levels(self) -> list[int]:
        return sorted({p.level for p in self.patches})

    @property
    def finest_level(self) -> int | None:
        return max((p.level for p in self.patches), default=None)

    def _ordered(self) -> list[Patch]:
        return sorted(self.patches, key=lambda p: p.level)  # stable: later wins at equal level

    def render(self, level: int, ix0: int = 0, iy0: int = 0, nx: int | None = None,
               ny: int | None = None, aggregate: bool = True) -> np.ndarray:
        """Field values (ny, nx) float64 on a window at `level` (default window: the full grid).

        Coarser patches are upsampled (a coarse value applies to its children). With `aggregate`,
        content finer than `level` is rendered at its own level and log-mean-exp aggregated, so the
        result is the exact cell average of exp(field). Without it, finer patches are ignored.
        """
        if nx is None or ny is None:
            gnx, gny = self.spec.shape(level)
            nx = gnx if nx is None else nx
            ny = gny if ny is None else ny
        win = (int(ix0), int(iy0), int(nx), int(ny))
        out = np.full((win[3], win[2]), self.fill, dtype=np.float64)
        finer: list[Patch] = []
        for p in self._ordered():
            if p.level <= level:
                _paint_up(out, p, level, win)
            elif aggregate and _intersect(p.window_at(level), win) is not None:
                finer.append(p)
        if finer:
            fmax = max(p.level for p in finer)
            k = 1 << (fmax - level)
            for p in finer:
                sw = _intersect(p.window_at(level), win)
                assert sw is not None
                fine = self.render(fmax, sw[0] * k, sw[1] * k, sw[2] * k, sw[3] * k, aggregate=False)
                agg = logmeanexp_downsample(fine, k)
                out[sw[1] - win[1]: sw[1] - win[1] + sw[3], sw[0] - win[0]: sw[0] - win[0] + sw[2]] = agg
        return out

    def value_at_xy(self, x, y) -> np.ndarray:
        """Point values (finest covering patch, else fill) at LAEA coordinates. Vectorised."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        x, y = np.broadcast_to(x, shape).ravel(), np.broadcast_to(y, shape).ravel()
        out = np.full(x.shape, self.fill, dtype=np.float64)
        by_level: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for p in self._ordered():
            if p.level not in by_level:
                by_level[p.level] = xy_to_cells(self.spec, x, y, p.level)
            ix, iy = by_level[p.level]
            i, j = ix - p.ix0, iy - p.iy0
            m = (i >= 0) & (i < p.nx) & (j >= 0) & (j < p.ny)
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            v = p.values[j[idx], i[idx]]
            ok = ~np.isnan(v)
            out[idx[ok]] = v[ok]
        return out.reshape(shape)

    def value_at_latlon(self, lat, lon) -> np.ndarray:
        return self.value_at_xy(*latlon_to_xy(self.spec, lat, lon))

    def value_at_cells(self, level: int, ix, iy) -> np.ndarray:
        """Point value at the centres of cells (the field's value there, not a cell average)."""
        return self.value_at_xy(*cell_centers_xy(self.spec, level, ix, iy))

    def finer_windows(self, level: int) -> list[Patch]:
        return [p for p in self.patches if p.level > level]


def _paint_up(out: np.ndarray, p: Patch, level: int, win: tuple[int, int, int, int]) -> None:
    """Paint patch p (p.level <= level) into `out` (window `win` at `level`); NaN is transparent."""
    sh = level - p.level
    pw = p.window_at(level)
    sw = _intersect(pw, win)
    if sw is None:
        return
    cols = np.arange(sw[0], sw[0] + sw[2], dtype=np.int64)
    rows = np.arange(sw[1], sw[1] + sw[3], dtype=np.int64)
    pi = (cols >> sh) - p.ix0
    pj = (rows >> sh) - p.iy0
    if sh == 0:
        sub = p.values[pj[0]: pj[-1] + 1, pi[0]: pi[-1] + 1]
    else:
        sub = p.values[np.ix_(pj, pi)]
    r0, c0 = sw[1] - win[1], sw[0] - win[0]
    tgt = out[r0: r0 + sw[3], c0: c0 + sw[2]]
    nan = np.isnan(sub)
    if nan.any():
        np.copyto(tgt, sub, where=~nan)
    else:
        tgt[...] = sub


def _reduce_dups(val: np.ndarray, inv: np.ndarray, n: int, reduce: str) -> np.ndarray:
    if reduce == "max":
        out = np.full(n, -np.inf)
        np.maximum.at(out, inv, val)
        return out
    if reduce == "sum":
        out = np.zeros(n)
        np.add.at(out, inv, val)
        return out
    if reduce == "last":
        out = np.full(n, np.nan)
        out[inv] = val  # numpy assigns in order: last occurrence wins
        return out
    if reduce == "logsumexp":
        m = np.full(n, -np.inf)
        np.maximum.at(m, inv, val)
        ms = np.where(np.isfinite(m), m, 0.0)
        s = np.zeros(n)
        np.add.at(s, inv, np.exp(val - ms[inv]))
        with np.errstate(divide="ignore"):
            return np.where(np.isneginf(m), -np.inf, ms + np.log(s))
    raise ValueError(f"unknown reduce {reduce!r}")


# ================================================================================================
# Layers (what modules hand to fusion)
# ================================================================================================

_MODULES = ("acoustic", "visual", "metadata")


@dataclass(eq=False)
class Layer(LogField):
    """A module's log-likelihood layer. See `fusion/combine.py` for the contract.

    `attrs` (optional) is a per-cell attribute table with int columns `level, ix, iy` and any other
    columns. Recognised by fusion: `t_s` (float, UTC epoch s of audio sample 0) and `flights`
    (list[str]). All attribute columns of the cell under a candidate are copied to its evidence.
    """

    spec: GridSpec
    name: str
    module: str = "visual"
    weight: float = 1.0
    fill: float = 0.0
    description: str = ""
    attrs: pl.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    patches: list[Patch] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.module not in _MODULES:
            raise ValueError(f"module must be one of {_MODULES}")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and >= 0")
        given = list(self.patches)
        LogField.__init__(self, self.spec, self.fill, given)
        if self.attrs is not None:
            missing = {"level", "ix", "iy"} - set(self.attrs.columns)
            if missing:
                raise ValueError(f"attrs missing columns {missing}")

    # --- convenience constructors --------------------------------------------------------------
    @classmethod
    def flat(cls, spec: GridSpec, name: str, module: str, **kw) -> Layer:
        """An uninformative layer (0 everywhere): identical to leaving the module out."""
        return cls(spec=spec, name=name, module=module, fill=0.0, **kw)

    @classmethod
    def from_dense(cls, spec: GridSpec, values: np.ndarray, name: str, module: str,
                   level: int | None = None, fill: float = 0.0, **kw) -> Layer:
        lay = cls(spec=spec, name=name, module=module, fill=fill, **kw)
        lay.add_dense(values, level)
        return lay

    @classmethod
    def from_cells(cls, spec: GridSpec, level: int, ix, iy, values, name: str, module: str,
                   fill: float = 0.0, reduce: str = "logsumexp", **kw) -> Layer:
        lay = cls(spec=spec, name=name, module=module, fill=fill, **kw)
        lay.add_cells(level, ix, iy, values, reduce=reduce)
        return lay

    @classmethod
    def from_function(cls, spec: GridSpec, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      name: str, module: str, level: int | None = None,
                      window: tuple[int, int, int, int] | None = None, fill: float = 0.0,
                      chunk_rows: int = 256, **kw) -> Layer:
        """Evaluate `fn(lat, lon) -> log values` at cell centres over a window (default: full grid
        at `level`, default base level)."""
        lv = spec.base_level if level is None else level
        if window is None:
            nx, ny = spec.shape(lv)
            window = (0, 0, nx, ny)
        ix0, iy0, nx, ny = window
        vals = np.empty((ny, nx), dtype=np.float64)
        ii = np.arange(ix0, ix0 + nx)
        for r0 in range(0, ny, chunk_rows):
            jj = np.arange(iy0 + r0, iy0 + min(r0 + chunk_rows, ny))
            gi, gj = np.meshgrid(ii, jj)
            lat, lon = cell_centers_latlon(spec, lv, gi, gj)
            vals[r0: r0 + len(jj)] = np.asarray(fn(lat, lon), dtype=np.float64).reshape(gi.shape)
        lay = cls(spec=spec, name=name, module=module, fill=fill, **kw)
        lay.add_patch(lv, ix0, iy0, vals)
        return lay

    # --- attrs ---------------------------------------------------------------------------------
    def set_attrs(self, level: int, ix, iy, **columns) -> None:
        """Attach a per-cell attribute table (replaces any existing one)."""
        n = len(np.asarray(ix))
        self.attrs = pl.DataFrame({
            "level": np.full(n, level, dtype=np.int64),
            "ix": np.asarray(ix, dtype=np.int64),
            "iy": np.asarray(iy, dtype=np.int64),
            **columns,
        })

    def attrs_at_xy(self, x: float, y: float) -> dict[str, Any] | None:
        """Attribute row of the finest attrs cell containing (x, y), or None."""
        if self.attrs is None or self.attrs.height == 0:
            return None
        for lv in sorted(set(self.attrs["level"].to_list()), reverse=True):
            ix, iy = xy_to_cells(self.spec, x, y, lv)
            hit = self.attrs.filter(
                (pl.col("level") == lv) & (pl.col("ix") == int(ix)) & (pl.col("iy") == int(iy))
            )
            if hit.height:
                return hit.row(0, named=True)
        return None

    # --- serialisation -------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write an .npz (patch arrays + JSON header + Arrow IPC attrs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": "triangulate.layer/1",
            "name": self.name, "module": self.module, "weight": self.weight, "fill": self.fill,
            "description": self.description, "meta": self.meta,
            "spec": self.spec.model_dump(),
            "patches": [{"level": p.level, "ix0": p.ix0, "iy0": p.iy0} for p in self.patches],
        }
        arrays: dict[str, np.ndarray] = {
            "header": np.frombuffer(json.dumps(header, default=str).encode(), dtype=np.uint8)
        }
        for i, p in enumerate(self.patches):
            arrays[f"p{i}"] = p.values
        if self.attrs is not None:
            buf = io.BytesIO()
            self.attrs.write_ipc(buf)
            arrays["attrs"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)  # type: ignore[arg-type]
        return path

    @classmethod
    def load(cls, path: str | Path) -> Layer:
        with np.load(Path(path), allow_pickle=False) as z:
            h = json.loads(bytes(z["header"]).decode())
            if h.get("format") != "triangulate.layer/1":
                raise ValueError(f"{path}: not a triangulate layer file")
            patches = [Patch(m["level"], m["ix0"], m["iy0"], z[f"p{i}"]) for i, m in enumerate(h["patches"])]
            attrs = pl.read_ipc(io.BytesIO(bytes(z["attrs"]))) if "attrs" in z.files else None
        return cls(spec=GridSpec(**h["spec"]), name=h["name"], module=h["module"], weight=h["weight"],
                   fill=h["fill"], description=h["description"], meta=h["meta"], attrs=attrs,
                   patches=patches)

    def to_ref(self, path: str | Path, relative_to: str | Path | None = None) -> LayerRef:
        """Save and return the `LayerRef` to put in `ModuleResult.layers`."""
        p = self.save(path)
        rel = str(p.relative_to(relative_to)) if relative_to is not None else str(p)
        return LayerRef(name=self.name, module=self.module, path=rel,  # type: ignore[arg-type]
                        level=self.finest_level, weight=self.weight, description=self.description)
