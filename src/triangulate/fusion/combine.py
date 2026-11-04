"""Layer combination: the fused log posterior on the multi-resolution grid (plan 1.3, WP12).

Layer contract (how a module submits evidence)
==============================================

1. **What a layer is.** A `fusion.grid.Layer` on the job's `GridSpec` frame: a piecewise-constant
   field of natural-log *likelihood* values ell(x) over observer location x, plus `fill` (the value
   wherever the layer has no patch). Values are point (intensive) quantities: a value on a 1600 m
   cell applies to every point inside it, so layers of different levels mix without renormalising.
   NaN inside a patch is transparent. -inf is a hard exclusion (avoid it; prefer a soft floor).
   +inf is invalid.

2. **Normalisation.** Layers need no normalisation: fusion normalises the posterior. Only
   differences in ell across x matter, so any additive constant is irrelevant to the posterior.
   Convention, so that per-candidate contributions are comparable across modules: **0 means
   "no information"**, i.e. express ell as a log likelihood ratio against an uninformative
   baseline. A missing, skipped, failed or disabled module contributes exactly 0 (a flat layer).
   - acoustic: ell_A = logsumexp over flight tuples of lambda * sum_k z_k where hypotheses exist,
     `fill` = ell_floor = log(p_miss) elsewhere (`acoustic_layer`). lambda is applied by the acoustic
     module (WP11, fitted in WP22), not by fusion.
   - visual: one layer per clue, ell_c = log(eps_c + (1 - eps_c) K_c(x)), eps_c = 1 - confidence
     (`membership_loglik`, `kernel_layer`, `polygon_layer`). The module total is the sum of its layers.
   - metadata: GPS mixture as a ratio to uniform, log(alpha N(x)/U + (1 - alpha)) with U = 1/area of
     the search domain (`gps_mixture_layer`). Time hints are priors on T_s, not layers.

3. **Weights.** The fused unnormalised log posterior is
   `L(x) = log pi(x) + sum_layers w_module * w_layer * ell_layer(x)`, where w_module are the user's
   tempering weights (`DEFAULT_WEIGHTS`: acoustic 1, visual 0.5, metadata 1) and w_layer is
   `Layer.weight` / `LayerRef.weight` (default 1; use it for per-clue down-weighting).
   pi is uniform inside the domain (bbox or polygon, cell centre inside) and 0 outside. An optional
   extra prior layer (e.g. a land mask) is added with weight 1.

4. **Handing over.** In-process, pass `Layer` objects to `fuse`. Across stages, save with
   `layer.to_ref(workdir / "layers" / f"{name}.npz", relative_to=workdir)` and put the LayerRef in
   `ModuleResult.layers`; `layers_from_results` loads them. Put attribution strings in
   `ModuleResult.provenance["attribution"]` (str or list[str]).

5. **Per-cell attributes.** A layer may carry `attrs` (columns `level, ix, iy` + anything). Fusion
   copies the row under each candidate into its evidence; `t_s` (UTC epoch s of audio sample 0) and
   `flights` (list[str]) fill `Candidate.t_s` / `Candidate.flights`.

Output
------
`Posterior`: normalised log density (ln probability per m^2) as a dense base array at
`spec.base_level` plus sparse 256x256 tiles at one fine level, wherever any layer has content finer
than the base level. Base cells under a tile hold the log-mean-exp of their children, so the total
mass is identical at every level.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf, logsumexp

from ..core.geo import BBox
from ..core.types import GRID_LEVELS_M, GridSpec, ModuleResult
from .grid import (
    FINEST_LEVEL,
    TILE,
    Layer,
    LogField,
    Patch,
    cell_centers_xy,
    coarsest_level_at_most,
    latlon_to_xy,
    logmeanexp_downsample,
    rasterize_geom,
    same_frame,
    window_for_xy_box,
    xy_to_cells,
)

__all__ = [
    "DEFAULT_WEIGHTS", "Posterior", "fuse", "domain_polygon_xy", "domain_area_m2", "layers_from_results",
    "membership_loglik", "kernel_layer", "polygon_layer", "gps_mixture_layer", "acoustic_layer",
    "evidence_at", "collect_attribution",
]

DEFAULT_WEIGHTS: dict[str, float] = {"acoustic": 1.0, "visual": 0.5, "metadata": 1.0}
EPS_MIN = 1e-6  # smallest clue epsilon: keeps a clue layer soft even at confidence 1


# ------------------------------------------------------------------------------------------------
# Domain (support of the prior)
# ------------------------------------------------------------------------------------------------


def domain_polygon_xy(spec: GridSpec, domain: BBox | Any | None = None, densify: int = 64):
    """Search domain as a shapely geometry in LAEA metres.

    `domain`: a (S, W, N, E) bbox, a shapely geometry in lon/lat (x = lon, y = lat), or None for the
    whole grid extent. Bbox edges are densified so the curved LAEA outline is followed.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import transform

    if domain is None:
        nx, ny = spec.shape(spec.base_level)
        s = GRID_LEVELS_M[spec.base_level]
        return box(spec.x0, spec.y0, spec.x0 + nx * s, spec.y0 + ny * s)
    if isinstance(domain, tuple | list):
        s_, w, n, e = (float(v) for v in domain)
        t = np.linspace(0.0, 1.0, densify, endpoint=False)
        lon = np.concatenate([w + (e - w) * t, np.full(densify, e), e - (e - w) * t, np.full(densify, w)])
        lat = np.concatenate([np.full(densify, s_), s_ + (n - s_) * t, np.full(densify, n), n - (n - s_) * t])
        x, y = latlon_to_xy(spec, lat, lon)
        return Polygon(np.column_stack([x, y]))

    def fwd(lon, lat, z=None):
        return latlon_to_xy(spec, lat, lon)

    return transform(fwd, domain)


def domain_area_m2(spec: GridSpec, domain: BBox | Any | None = None) -> float:
    return float(domain_polygon_xy(spec, domain).area)


# ------------------------------------------------------------------------------------------------
# Posterior
# ------------------------------------------------------------------------------------------------


@dataclass
class Posterior:
    """Normalised log posterior density (ln p per m^2). See module docstring."""

    spec: GridSpec
    base_level: int
    base: np.ndarray  # (ny, nx) float64 at base_level; aggregated (log-mean-exp) under tiles
    fine_level: int | None = None
    tiles: dict[tuple[int, int], Patch] = field(default_factory=dict)  # (tx, ty) -> Patch at fine_level
    covered: np.ndarray | None = None  # (ny, nx) bool: base cell is refined by a tile
    log_norm: float = 0.0  # ln of the integral of exp(unnormalised L) over the domain
    weights: dict[str, float] = field(default_factory=dict)  # module weights used
    layers: list[tuple[Layer, float]] = field(default_factory=list)  # (layer, effective weight)
    prior: Layer | None = None
    domain_xy: Any = None  # shapely geometry, LAEA metres
    warnings: list[str] = field(default_factory=list)
    _field: LogField | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.covered is None:
            self.covered = np.zeros(self.base.shape, dtype=bool)

    @property
    def field(self) -> LogField:
        """The posterior as a LogField (base patch + fine tile patches); fill = -inf."""
        if self._field is None:
            f = LogField(self.spec, fill=-np.inf)
            f.add_patch(self.base_level, 0, 0, self.base)
            for p in self.tiles.values():
                f.add_patch(p.level, p.ix0, p.iy0, p.values)
            self._field = f
        return self._field

    @property
    def leaf_level_sizes(self) -> tuple[int, int | None]:
        return GRID_LEVELS_M[self.base_level], (GRID_LEVELS_M[self.fine_level] if self.fine_level is not None
                                                 else None)

    def value_at_xy(self, x, y) -> np.ndarray:
        """ln density per m^2 at LAEA points (value of the finest leaf)."""
        return self.field.value_at_xy(x, y)

    def value_at_latlon(self, lat, lon) -> np.ndarray:
        return self.field.value_at_latlon(lat, lon)

    def leaves(self) -> dict[str, np.ndarray]:
        """All leaf cells: arrays `value` (ln density), `log_area`, `level`, `ix`, `iy`."""
        assert self.covered is not None
        jj, ii = np.nonzero(~self.covered)
        parts_v = [self.base[jj, ii]]
        parts_lv = [np.full(ii.size, self.base_level, dtype=np.int64)]
        parts_ix, parts_iy = [ii.astype(np.int64)], [jj.astype(np.int64)]
        for p in self.tiles.values():
            gj, gi = np.mgrid[0: p.ny, 0: p.nx]
            parts_v.append(p.values.ravel())
            parts_lv.append(np.full(p.values.size, p.level, dtype=np.int64))
            parts_ix.append((gi + p.ix0).ravel().astype(np.int64))
            parts_iy.append((gj + p.iy0).ravel().astype(np.int64))
        lv = np.concatenate(parts_lv)
        sizes = np.asarray(GRID_LEVELS_M, dtype=np.float64)[lv]
        return {"value": np.concatenate(parts_v), "log_area": 2.0 * np.log(sizes), "level": lv,
                "ix": np.concatenate(parts_ix), "iy": np.concatenate(parts_iy)}

    def total_mass(self) -> float:
        lf = self.leaves()
        return float(np.exp(logsumexp(lf["value"] + lf["log_area"])))

    def at_level(self, level: int) -> np.ndarray:
        """Dense ln density at a level <= base_level (log-mean-exp aggregation; -inf padding)."""
        if level > self.base_level:
            raise ValueError("use field.render(level, window) for levels finer than the base")
        k = 1 << (self.base_level - level)
        ny, nx = self.base.shape
        nyp, nxp = -(-ny // k) * k, -(-nx // k) * k
        a = np.full((nyp, nxp), -np.inf)
        a[:ny, :nx] = self.base
        return logmeanexp_downsample(a, k)

    # --- persistence ---------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """npz with the raster parts and a JSON header (layers are not embedded; they have files)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hdr = {
            "format": "triangulate.posterior/1", "spec": self.spec.model_dump(),
            "base_level": self.base_level, "fine_level": self.fine_level, "log_norm": self.log_norm,
            "weights": self.weights, "warnings": self.warnings,
            "tiles": [{"tx": k[0], "ty": k[1], "ix0": p.ix0, "iy0": p.iy0} for k, p in self.tiles.items()],
            "layers": [{"name": lay.name, "module": lay.module, "weight": w} for lay, w in self.layers],
        }
        arrays = {"header": np.frombuffer(json.dumps(hdr).encode(), dtype=np.uint8), "base": self.base,
                  "covered": self.covered}
        for i, p in enumerate(self.tiles.values()):
            arrays[f"t{i}"] = p.values
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)  # type: ignore[arg-type]
        return path

    @classmethod
    def load(cls, path: str | Path) -> Posterior:
        with np.load(Path(path), allow_pickle=False) as z:
            h = json.loads(bytes(z["header"]).decode())
            if h.get("format") != "triangulate.posterior/1":
                raise ValueError(f"{path}: not a triangulate posterior file")
            fl = h["fine_level"]
            tiles = {(t["tx"], t["ty"]): Patch(fl, t["ix0"], t["iy0"], z[f"t{i}"])
                     for i, t in enumerate(h["tiles"])}
            return cls(spec=GridSpec(**h["spec"]), base_level=h["base_level"], base=z["base"], fine_level=fl,
                       tiles=tiles, covered=z["covered"], log_norm=h["log_norm"], weights=h["weights"],
                       warnings=h["warnings"])


def _effective(layers: Iterable[Layer], weights: dict[str, float]) -> list[tuple[Layer, float]]:
    out = []
    for lay in layers:
        w = float(weights.get(lay.module, 1.0)) * float(lay.weight)
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"layer {lay.name!r}: effective weight {w} must be finite and >= 0")
        out.append((lay, w))
    return out


def fuse(spec: GridSpec, layers: Iterable[Layer], *, weights: dict[str, float] | None = None,
         domain: BBox | Any | None = None, prior: Layer | None = None, fine_level: int | None = None,
         max_fine_cells: int = 64_000_000) -> Posterior:
    """Combine layers into the normalised posterior.

    `weights`: module -> tempering weight (missing modules use DEFAULT_WEIGHTS, then 1).
    `domain`: bbox / lon-lat shapely geometry / None (whole grid); pi = 0 outside.
    `prior`: optional extra log-prior layer (weight 1), e.g. a land mask (off by default).
    `fine_level`: cap for the tile level (default: finest level present in any layer).
    """
    w_mod = {**DEFAULT_WEIGHTS, **(weights or {})}
    used = _effective(layers, w_mod)
    for lay, _ in used:
        if not same_frame(lay.spec, spec):
            raise ValueError(f"layer {lay.name!r} is on a different grid frame")
    if prior is not None and not same_frame(prior.spec, spec):
        raise ValueError("prior layer is on a different grid frame")
    active = [(lay, w) for lay, w in used if w > 0]
    if prior is not None:
        active.append((prior, 1.0))

    b = spec.base_level
    nxb, nyb = spec.shape(b)
    dom = domain_polygon_xy(spec, domain)
    warnings: list[str] = []

    def combined(level: int, win: tuple[int, int, int, int], aggregate: bool) -> np.ndarray:
        acc = np.zeros((win[3], win[2]), dtype=np.float64)
        for lay, w in active:
            v = lay.render(level, *win, aggregate=aggregate)
            acc += v if w == 1.0 else w * v
        mask = rasterize_geom(spec, dom, level, *win)
        acc[~mask] = -np.inf
        return acc

    base = combined(b, (0, 0, nxb, nyb), aggregate=False)
    covered = np.zeros((nyb, nxb), dtype=bool)

    present = [p.level for lay, _ in active for p in lay.patches]
    fl = max(present, default=b)
    if fine_level is not None:
        fl = min(fl, fine_level)
    tiles: dict[tuple[int, int], Patch] = {}
    if fl > b:
        k = 1 << (fl - b)
        nxf, nyf = nxb * k, nyb * k
        keys: set[tuple[int, int]] = set()
        for lay, _ in active:
            for p in lay.patches:
                if p.level <= b:
                    continue
                ix0, iy0, nx, ny = p.window_at(fl)
                ix1, iy1 = min(ix0 + nx, nxf), min(iy0 + ny, nyf)
                ix0, iy0 = max(ix0, 0), max(iy0, 0)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                for tx in range(ix0 // TILE, (ix1 - 1) // TILE + 1):
                    for ty in range(iy0 // TILE, (iy1 - 1) // TILE + 1):
                        keys.add((tx, ty))
        if len(keys) * TILE * TILE > max_fine_cells:
            raise ValueError(f"{len(keys)} fine tiles exceed max_fine_cells={max_fine_cells}; "
                             "use coarser layers or a lower fine_level")
        for tx, ty in sorted(keys):
            win = (tx * TILE, ty * TILE, min(TILE, nxf - tx * TILE), min(TILE, nyf - ty * TILE))
            vals = combined(fl, win, aggregate=True)
            tiles[(tx, ty)] = Patch(fl, win[0], win[1], vals)
            bx, by, bnx, bny = win[0] // k, win[1] // k, win[2] // k, win[3] // k
            base[by: by + bny, bx: bx + bnx] = logmeanexp_downsample(vals, k)
            covered[by: by + bny, bx: bx + bnx] = True

    finite = np.isfinite(base)
    if not finite.any():
        raise ValueError("posterior has zero mass (empty domain or hard exclusions everywhere)")
    log_norm = float(logsumexp(base[finite]) + 2.0 * math.log(GRID_LEVELS_M[b]))
    base -= log_norm
    for p in tiles.values():
        p.values -= log_norm
    return Posterior(spec=spec, base_level=b, base=base, fine_level=fl if fl > b else None, tiles=tiles,
                     covered=covered, log_norm=log_norm, weights=w_mod, layers=used, prior=prior,
                     domain_xy=dom, warnings=warnings)


# ------------------------------------------------------------------------------------------------
# Evidence breakdown
# ------------------------------------------------------------------------------------------------


def evidence_at(post: Posterior, x: float, y: float) -> dict[str, Any]:
    """Per-layer and per-module contributions at a LAEA point.

    Returns {"layers": [{name, module, raw, weight, weighted}], "contributions": {module: sum},
    "attrs": {layer name: attrs row}}. `contributions` include "prior" when a prior layer is used.
    Weighted values satisfy: posterior(x) = sum(contributions) - post.log_norm (inside the domain).
    """
    rows: list[dict[str, Any]] = []
    contrib: dict[str, float] = {}
    attrs: dict[str, Any] = {}
    for lay, w in post.layers:
        raw = float(lay.value_at_xy(x, y))
        weighted = 0.0 if w == 0 else w * raw
        rows.append({"name": lay.name, "module": lay.module, "raw": raw, "weight": w, "weighted": weighted})
        contrib[lay.module] = contrib.get(lay.module, 0.0) + weighted
        a = lay.attrs_at_xy(x, y)
        if a is not None:
            attrs[lay.name] = a
    if post.prior is not None:
        contrib["prior"] = float(post.prior.value_at_xy(x, y))
    return {"layers": rows, "contributions": contrib, "attrs": attrs}


# ------------------------------------------------------------------------------------------------
# Module results
# ------------------------------------------------------------------------------------------------


def layers_from_results(results: Sequence[ModuleResult], root: str | Path | None = None,
                        ) -> tuple[list[Layer], list[str]]:
    """Load the layers of usable module results ("ok"/"degraded"). Skipped or failed modules
    contribute nothing (= a flat layer). `LayerRef.weight` overrides the saved layer weight.
    Returns (layers, notes)."""
    out: list[Layer] = []
    notes: list[str] = []
    for r in results:
        if r.status not in ("ok", "degraded"):
            notes.append(f"{r.module}: {r.status}" + (f" ({r.message})" if r.message else "") + " -> flat")
            continue
        for ref in r.layers:
            if ref.path is None:
                notes.append(f"{r.module}/{ref.name}: no path, ignored")
                continue
            p = Path(ref.path)
            if root is not None and not p.is_absolute():
                p = Path(root) / p
            lay = Layer.load(p)
            lay.weight = float(ref.weight)
            lay.module = ref.module
            out.append(lay)
    return out, notes


def collect_attribution(results: Sequence[ModuleResult]) -> list[str]:
    seen: list[str] = []
    for r in results:
        a = r.provenance.get("attribution")
        for s in [a] if isinstance(a, str) else (a or []):
            if s and s not in seen:
                seen.append(str(s))
    return seen


# ------------------------------------------------------------------------------------------------
# Layer builders for modules (WP11 acoustic, WP13 metadata, WP16 visual)
# ------------------------------------------------------------------------------------------------


def membership_loglik(K, confidence: float) -> np.ndarray:
    """Clue log-likelihood log(eps + (1 - eps) K), eps = 1 - confidence (>= EPS_MIN). K in [0, 1]."""
    eps = max(1.0 - float(confidence), EPS_MIN)
    k = np.clip(np.asarray(K, dtype=np.float64), 0.0, 1.0)
    return np.log(eps + (1.0 - eps) * k)


def _level_for_sigma(spec: GridSpec, sigma_m: float, level: int | None) -> int:
    if level is not None:
        return level
    return max(spec.base_level, min(coarsest_level_at_most(sigma_m), FINEST_LEVEL))


def kernel_layer(spec: GridSpec, lat: float, lon: float, sigma_m: float, confidence: float, name: str,
                 module: str = "visual", level: int | None = None, radius_sigmas: float = 5.0,
                 **kw) -> Layer:
    """Gaussian membership K = exp(-r^2 / (2 sigma^2)) around a point -> clue layer, sampled at cell
    centres. Level: the coarsest with cell <= sigma / 2 (not coarser than base). Outside the
    +-5 sigma window the value is the fill log(eps)."""
    lv = _level_for_sigma(spec, sigma_m / 2.0, level)
    x, y = latlon_to_xy(spec, lat, lon)
    r = radius_sigmas * sigma_m
    ix0, iy0, nx, ny = window_for_xy_box(spec, lv, float(x) - r, float(y) - r, float(x) + r, float(y) + r)
    gj, gi = np.mgrid[iy0: iy0 + ny, ix0: ix0 + nx]
    cx, cy = cell_centers_xy(spec, lv, gi, gj)
    K = np.exp(-((cx - x) ** 2 + (cy - y) ** 2) / (2.0 * sigma_m**2))
    lay = Layer(spec=spec, name=name, module=module, fill=float(membership_loglik(0.0, confidence)), **kw)
    lay.add_patch(lv, ix0, iy0, membership_loglik(K, confidence))
    lay.meta.update({"kind": "kernel", "lat": float(lat), "lon": float(lon), "sigma_m": sigma_m,
                     "confidence": confidence})
    return lay


def polygon_layer(spec: GridSpec, geom_lonlat, confidence: float, name: str, module: str = "visual",
                  soft_edge_m: float = 5000.0, level: int | None = None, **kw) -> Layer:
    """Polygon membership (lon/lat shapely geometry) -> clue layer, dense at `level` (default base).
    K = 1 inside; outside K = exp(-d^2 / (2 (soft_edge_m / 3)^2)), d = distance to the polygon, so K is
    about 0.01 at `soft_edge_m`."""
    from scipy.ndimage import distance_transform_edt

    lv = spec.base_level if level is None else level
    nx, ny = spec.shape(lv)
    gxy = domain_polygon_xy(spec, geom_lonlat)
    inside = rasterize_geom(spec, gxy, lv, 0, 0, nx, ny)
    s = GRID_LEVELS_M[lv]
    if inside.any():
        d = distance_transform_edt(~inside) * s
        sig = max(soft_edge_m / 3.0, 1e-9)
        K = np.where(inside, 1.0, np.exp(-0.5 * (d / sig) ** 2))
    else:
        K = np.zeros((ny, nx))
    lay = Layer(spec=spec, name=name, module=module, fill=float(membership_loglik(0.0, confidence)), **kw)
    lay.add_patch(lv, 0, 0, membership_loglik(K, confidence).astype(np.float32))
    lay.meta.update({"kind": "polygon", "soft_edge_m": soft_edge_m, "confidence": confidence})
    return lay


def gps_mixture_layer(spec: GridSpec, lat: float, lon: float, accuracy_m: float | None,
                      domain: BBox | Any | None, alpha: float = 0.6, min_sigma_m: float = 30.0,
                      name: str = "gps", level: int | None = None, radius_sigmas: float = 6.0,
                      **kw) -> Layer:
    """Untrusted GPS hint: log(alpha N(x; gps, sigma^2) / U + (1 - alpha)), U = 1 / domain area,
    sigma = max(accuracy, 30 m). N is integrated exactly over each cell (erf), so the layer's mass is
    right at any level. Fill = log(1 - alpha)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    sigma = max(float(accuracy_m or 0.0), min_sigma_m)
    area = domain_area_m2(spec, domain)
    lv = _level_for_sigma(spec, sigma, level)
    s = GRID_LEVELS_M[lv]
    xa, ya = latlon_to_xy(spec, lat, lon)
    x, y = float(xa), float(ya)
    r = radius_sigmas * sigma + s
    ix0, iy0, nx, ny = window_for_xy_box(spec, lv, x - r, y - r, x + r, y + r)
    ex = spec.x0 + np.arange(ix0, ix0 + nx + 1) * s
    ey = spec.y0 + np.arange(iy0, iy0 + ny + 1) * s
    cx = 0.5 * erf((ex - x) / (sigma * math.sqrt(2.0)))
    cy = 0.5 * erf((ey - y) / (sigma * math.sqrt(2.0)))
    p_cell = np.outer(np.diff(cy), np.diff(cx))  # (ny, nx) probability per cell
    vals = np.log(alpha * area * p_cell / (s * s) + (1.0 - alpha))
    lay = Layer(spec=spec, name=name, module="metadata", fill=math.log(1.0 - alpha), **kw)
    lay.add_patch(lv, ix0, iy0, vals)
    lay.meta.update({"kind": "gps_mixture", "lat": float(lat), "lon": float(lon), "sigma_m": sigma,
                     "alpha": alpha, "domain_area_m2": area})
    return lay


def acoustic_layer(spec: GridSpec, level: int, ix, iy, z_sum, lam: float = 0.5, p_miss: float = 0.1,
                   name: str = "acoustic", attrs: dict[str, Any] | None = None, **kw) -> Layer:
    """ell_A from hypotheses: one row per (cell, flight tuple) with z_sum = sum_k z_k. Cells get
    logsumexp_tuples(lam * z_sum); everything else gets ell_floor = log(p_miss).
    Refinement: call `layer.add_cells(finer_level, ix, iy, lam * z_sum)` again; finer cells override,
    and children without a value fall back to their coarser cell.
    `attrs`: optional per-row columns (e.g. t_s, flights); for each cell the row with the highest
    z_sum is kept."""
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    z = np.asarray(z_sum, dtype=np.float64)
    lay = Layer(spec=spec, name=name, module="acoustic", fill=math.log(p_miss), **kw)
    lay.add_cells(level, ix, iy, lam * z, reduce="logsumexp")
    lay.meta.update({"kind": "acoustic", "lambda": lam, "p_miss": p_miss})
    if attrs is not None and ix.size:
        import polars as pl

        df = pl.DataFrame({"level": np.full(ix.size, level, dtype=np.int64), "ix": ix, "iy": iy,
                           "z_sum": z, **attrs})
        df = df.sort("z_sum", descending=True).unique(subset=["level", "ix", "iy"], keep="first",
                                                      maintain_order=True)
        lay.attrs = df if lay.attrs is None else pl.concat([lay.attrs, df], how="diagonal_relaxed")
    return lay


def xy_cell(spec: GridSpec, x: float, y: float, level: int) -> tuple[int, int, int]:
    ix, iy = xy_to_cells(spec, x, y, level)
    return level, int(ix), int(iy)
