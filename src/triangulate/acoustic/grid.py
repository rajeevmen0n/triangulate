"""Candidate-grid geometry for the acoustic search (plan 3.3.3, WP09).

Cell ids and LAEA coordinates come from `fusion.grid` (WP12): a cell is `(level, ix, iy)` of the job's
`GridSpec`, packed to one int64 with `fusion.grid.pack_cells`. This module adds what the CPA stage needs
on top of that:

- `CellSet`: a set of cells at one level (vectorised ix/iy).
- `AffineMap`: a fitted affine map between local ENU metres (the frame of the prepared ADS-B tracks,
  `adsb.tracks.prepare_tracks(origin=...)`) and *fractional cell coordinates* F = (u, v), where the centre
  of cell (ix, iy) is F = (ix + 0.5, iy + 0.5). Over a regional window LAEA and ENU differ by a near-affine
  map; the measured maximum residual `eps_m` is carried along so every geometric test that uses the map
  can be made conservative (the exact tests always use exact ENU cell positions).
- `ObserverCells`: per-cell observer position in the tracks' ENU frame: ground ellipsoidal height
  (DEM + geoid N, `dem.provider.ellipsoidal_height`) + h_obs, the local "up" unit vector, lat/lon.
- `cells_within`: dense boolean mask of cells within a horizontal radius of a point cloud (Euclidean
  distance transform), used for "200 m cells within R_h of the speed-gated track samples".
- `densify`: children of surviving cells (plus a halo) at a finer level, for the 100 -> 50 -> 25 m
  refinement (plan 3.3.7).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core.geo import EARTH_R, geodetic_to_enu
from ..fusion.grid import (
    GridSpec,
    cell_centers_latlon,
    cell_size,
    level_for_size,
    pack_cells,
    to_level,
    unpack_cells,
)

__all__ = [
    "SEARCH_LEVEL", "CellSet", "AffineMap", "ObserverCells", "GroundFn", "default_ground_hae",
    "region_window", "fit_lattice_map", "observer_cells", "cells_within", "densify", "max_alt_shift_m",
]

SEARCH_LEVEL = level_for_size(200)  # the coarse 200 m candidate grid (plan 3.3.3)

GroundFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (lat, lon) -> ground HAE metres


@dataclass(frozen=True, eq=False)
class CellSet:
    """Cells (level, ix[i], iy[i]) of `spec`, unique, in no particular order."""

    spec: GridSpec
    level: int
    ix: np.ndarray  # int64
    iy: np.ndarray  # int64

    def __post_init__(self) -> None:
        object.__setattr__(self, "ix", np.ascontiguousarray(self.ix, dtype=np.int64).ravel())
        object.__setattr__(self, "iy", np.ascontiguousarray(self.iy, dtype=np.int64).ravel())
        if self.ix.shape != self.iy.shape:
            raise ValueError("ix and iy must have the same length")

    def __len__(self) -> int:
        return int(self.ix.size)

    @property
    def keys(self) -> np.ndarray:
        return pack_cells(self.level, self.ix, self.iy)

    @property
    def size_m(self) -> int:
        return cell_size(self.level)

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        return cell_centers_latlon(self.spec, self.level, self.ix, self.iy)

    @classmethod
    def from_mask(cls, spec: GridSpec, level: int, mask: np.ndarray, ix0: int = 0, iy0: int = 0) -> CellSet:
        """Cells where `mask[j, i]` (row j = iy - iy0, column i = ix - ix0) is True."""
        jj, ii = np.nonzero(mask)
        return cls(spec, level, ii.astype(np.int64) + ix0, jj.astype(np.int64) + iy0)

    @classmethod
    def window(cls, spec: GridSpec, level: int, ix0: int, iy0: int, nx: int, ny: int) -> CellSet:
        gi, gj = np.meshgrid(np.arange(ix0, ix0 + nx), np.arange(iy0, iy0 + ny))
        return cls(spec, level, gi.ravel(), gj.ravel())

    def subset(self, sel) -> CellSet:
        return CellSet(self.spec, self.level, self.ix[sel], self.iy[sel])


def region_window(spec: GridSpec, level: int, pad_cells: int = 0) -> tuple[int, int, int, int]:
    """(ix0, iy0, nx, ny) of the whole region at `level`, grown by `pad_cells` on every side."""
    nx, ny = spec.shape(level)
    return -pad_cells, -pad_cells, nx + 2 * pad_cells, ny + 2 * pad_cells


@dataclass(frozen=True)
class AffineMap:
    """F = A @ [e, n, 1] (fractional cell coords) and X = B @ [u, v, 1] (ENU metres), A = B^-1.

    `eps_m` bounds |X_true - B @ [F, 1]| over the points the map was fitted on (metres).
    """

    A: np.ndarray  # (2, 3) ENU -> F
    B: np.ndarray  # (2, 3) F -> ENU
    eps_m: float

    @classmethod
    def fit(cls, F: np.ndarray, X: np.ndarray, safety_m: float = 0.5) -> AffineMap:
        F = np.asarray(F, dtype=np.float64).reshape(-1, 2)
        X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
        if F.shape[0] < 3:  # degenerate (1-2 cells): pure translation + nominal scale from the spacing
            raise ValueError("need >= 3 points to fit an affine map")
        fm = F.mean(axis=0)
        M = np.column_stack([F - fm, np.ones(F.shape[0])])
        sol, *_ = np.linalg.lstsq(M, X, rcond=None)  # (3, 2): X = (F - fm) @ L + t
        L = sol[:2].T  # (2, 2)
        t = sol[2] - L @ fm
        B = np.column_stack([L, t])
        resid = X - (F @ L.T + t)
        eps = float(np.sqrt((resid**2).sum(axis=1)).max()) if F.shape[0] else 0.0
        Ai = np.linalg.inv(L)
        A = np.column_stack([Ai, -Ai @ t])
        return cls(A=A, B=B, eps_m=eps + safety_m)

    def to_cells(self, e, n) -> tuple[np.ndarray, np.ndarray]:
        e = np.asarray(e, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        A = self.A
        return A[0, 0] * e + A[0, 1] * n + A[0, 2], A[1, 0] * e + A[1, 1] * n + A[1, 2]

    def to_enu(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        B = self.B
        return B[0, 0] * u + B[0, 1] * v + B[0, 2], B[1, 0] * u + B[1, 1] * v + B[1, 2]

    @property
    def stretch(self) -> float:
        """Largest singular value of A's linear part (cells per metre, ~1/cell size)."""
        return float(np.linalg.norm(self.A[:, :2], 2))


def fit_lattice_map(spec: GridSpec, level: int, origin: tuple[float, float],
                    window: tuple[int, int, int, int], n: int = 17, h: float = 0.0) -> AffineMap:
    """Fit the ENU<->cell map on an n x n lattice of cell centres (at ellipsoidal height h) over `window`.

    The residual between lattice points is slightly larger than at them; a 1.5x + 1 m safety is added.
    """
    ix0, iy0, nx, ny = window
    us = np.linspace(ix0, ix0 + nx, n)
    vs = np.linspace(iy0, iy0 + ny, n)
    gu, gv = np.meshgrid(us, vs)
    gu, gv = gu.ravel(), gv.ravel()
    lat, lon = cell_centers_latlon(spec, level, gu - 0.5, gv - 0.5)  # F = (u, v) is a cell-centre coord
    e, nn, _ = geodetic_to_enu(lat, lon, np.full_like(lat, h), origin[0], origin[1])
    m = AffineMap.fit(np.column_stack([gu, gv]), np.column_stack([e, nn]), safety_m=0.0)
    return AffineMap(A=m.A, B=m.B, eps_m=1.5 * m.eps_m + 1.0)


def default_ground_hae(level: int) -> GroundFn:
    """Copernicus DEM (GLO-90 for cells >= 200 m, GLO-30 finer) + EGM2008 geoid N: ground HAE metres."""
    res = 90 if cell_size(level) >= 200 else 30

    def fn(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        from ..dem.provider import ellipsoidal_height

        return np.asarray(ellipsoidal_height(lat, lon, res), dtype=np.float64)

    return fn


@dataclass(frozen=True, eq=False)
class ObserverCells:
    """Observer positions for a `CellSet` in the ENU frame of the tracks (origin lat/lon, h0 = 0)."""

    cells: CellSet
    origin: tuple[float, float]
    lat: np.ndarray
    lon: np.ndarray
    ground_hae: np.ndarray  # m, DEM + geoid N
    h_obs: float
    xyz: np.ndarray  # (m, 3) ENU of the observer (ground + h_obs)
    up: np.ndarray  # (m, 3) local vertical unit vector at the observer, in the ENU frame

    def __len__(self) -> int:
        return len(self.cells)

    def subset(self, sel) -> ObserverCells:
        return ObserverCells(self.cells.subset(sel), self.origin, self.lat[sel], self.lon[sel],
                             self.ground_hae[sel], self.h_obs, self.xyz[sel], self.up[sel])


def observer_cells(cells: CellSet, origin: tuple[float, float], ground_hae: GroundFn | None = None,
                   h_obs: float = 5.0) -> ObserverCells:
    """Observer height Z_obs = DEM(cell) + N_geoid + h_obs at each cell centre (plan 3.3.3)."""
    lat, lon = cells.latlon()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    fn = ground_hae if ground_hae is not None else default_ground_hae(cells.level)
    g = np.asarray(fn(lat, lon), dtype=np.float64).ravel() if lat.size else np.zeros(0)
    g = np.where(np.isfinite(g), g, 0.0)
    z = g + h_obs
    e, n, u = geodetic_to_enu(lat, lon, z, origin[0], origin[1])
    e1, n1, u1 = geodetic_to_enu(lat, lon, z + 1.0, origin[0], origin[1])
    xyz = np.column_stack([e, n, u]).astype(np.float64) if lat.size else np.zeros((0, 3))
    up = np.column_stack([e1 - e, n1 - n, u1 - u]) if lat.size else np.zeros((0, 3))
    if lat.size:
        up /= np.linalg.norm(up, axis=1, keepdims=True)
    return ObserverCells(cells, (float(origin[0]), float(origin[1])), lat, lon, g, float(h_obs),
                         np.ascontiguousarray(xyz), np.ascontiguousarray(up))


def cells_within(u: np.ndarray, v: np.ndarray, window: tuple[int, int, int, int],
                 radius_cells: float) -> np.ndarray:
    """Bool mask (ny, nx) over `window` of cells whose centre is within `radius_cells` (+ the
    half-diagonal of a cell, because points are snapped to the cell containing them) of any point
    (u, v) in fractional cell coordinates. Points outside the window are ignored, so pass a window
    padded by the radius and crop afterwards."""
    from scipy.ndimage import distance_transform_edt

    ix0, iy0, nx, ny = window
    occ = np.zeros((ny, nx), dtype=bool)
    if len(u):
        ci = np.floor(np.asarray(u)).astype(np.int64) - ix0
        cj = np.floor(np.asarray(v)).astype(np.int64) - iy0
        ok = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
        occ[cj[ok], ci[ok]] = True
    if not occ.any():
        return occ
    dist = distance_transform_edt(~occ)
    return dist <= radius_cells + math.sqrt(0.5)


def densify(level: int, ix, iy, target_level: int, halo: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Unique cells at `target_level` (finer) covering the given cells grown by `halo` cells at `level`
    (the plan's "surviving cells plus a 1-cell halo")."""
    if target_level < level:
        raise ValueError("target_level must be finer (larger index) than level")
    ix = np.asarray(ix, dtype=np.int64).ravel()
    iy = np.asarray(iy, dtype=np.int64).ravel()
    if halo > 0 and ix.size:
        d = np.arange(-halo, halo + 1)
        dx, dy = np.meshgrid(d, d)
        ix = (ix[:, None] + dx.ravel()[None, :]).ravel()
        iy = (iy[:, None] + dy.ravel()[None, :]).ravel()
    if ix.size:
        _, ix, iy = unpack_cells(np.unique(pack_cells(level, ix, iy)))
    cx, cy = to_level(level, ix, iy, target_level)
    f = 1 << (target_level - level)
    o = np.arange(f)
    ox, oy = np.meshgrid(o, o)
    fx = (cx[:, None] + ox.ravel()[None, :]).ravel()
    fy = (cy[:, None] + oy.ravel()[None, :]).ravel()
    return fx, fy


def max_alt_shift_m(z_max: float, horiz_extent_m: float) -> float:
    """Upper bound on the horizontal ENU offset between an aircraft at height z and the ground point
    below it (the local vertical tilts by s/R over a distance s from the ENU origin)."""
    return max(z_max, 0.0) * horiz_extent_m / EARTH_R
