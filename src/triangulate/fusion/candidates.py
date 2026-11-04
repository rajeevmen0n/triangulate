"""Candidates (non-maximum suppression), HPD regions, and late re-ranking (plan 1.3, WP12).

- `find_candidates`: local maxima of the posterior (leaf resolution), greedy NMS with a 300 m radius,
  ranked by `prob_mass` (posterior mass within `mass_radius_m`, default = the NMS radius) or by peak
  density. Each `FusionCandidate` carries the per-module contribution breakdown, the attribute rows
  of every layer under it (acoustic: t_s, flights, per-event details), its HPD level, and the margin
  to the next candidate (difference of ln rank scores, in nats).
- `compute_hpd`: highest-posterior-density regions (default 50 % and 90 %) with threshold, area and
  polygon (LAEA metres).
- `rerank`: late evidence that depends on (x, T_s), e.g. sun/shadow (WP17), applied to the top-K.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import Field
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from ..core.types import GRID_LEVELS_M, Candidate, CellId
from .combine import Posterior, evidence_at
from .grid import TILE, cell_centers_xy, north_up_transform, window_for_xy_box, xy_to_cells, xy_to_latlon

__all__ = ["FusionCandidate", "HPDRegion", "find_candidates", "compute_hpd", "credible_level",
           "pinned_candidate", "rerank", "disc_log_mass"]


class FusionCandidate(Candidate):
    """`core.types.Candidate` plus grid position and evidence (a drop-in subclass)."""

    x_m: float = 0.0  # LAEA
    y_m: float = 0.0
    cell: CellId = (0, 0, 0)  # leaf cell of the peak
    mass_radius_m: float = 300.0
    rank_score: float = 0.0  # ln of the ranking quantity, plus late evidence
    evidence: dict[str, Any] = Field(default_factory=dict)  # {"layers": [...], "attrs": {...}}
    late: dict[str, float] = Field(default_factory=dict)  # late re-rank terms (weighted)
    pinned: bool = False  # e.g. trusted GPS (rank 0)


@dataclass
class HPDRegion:
    mass: float  # requested credible mass, e.g. 0.9
    log_threshold: float  # ln density per m^2: region = {x : posterior(x) >= threshold}
    area_m2: float  # area of the smallest set of leaves (in density order) reaching `mass`
    actual_mass: float  # mass of that discretised set (>= mass)
    geometry_xy: Any = None  # shapely (Multi)Polygon in LAEA metres ({density >= threshold}), or None


# ------------------------------------------------------------------------------------------------
# Mass bookkeeping
# ------------------------------------------------------------------------------------------------


class _Sorted:
    """Leaves sorted by density (descending) with cumulative mass."""

    def __init__(self, post: Posterior):
        lf = post.leaves()
        v = lf["value"]
        order = np.argsort(-v, kind="stable")
        self.v = v[order]
        self.area = np.exp(lf["log_area"][order])
        with np.errstate(under="ignore"):
            self.cum = np.cumsum(np.exp(self.v + lf["log_area"][order]))

    def level_of(self, value) -> np.ndarray:
        """Cumulative mass of leaves with density >= value (the HPD level of that density)."""
        idx = np.searchsorted(-self.v, -np.asarray(value, dtype=np.float64), side="right") - 1
        return np.where(idx >= 0, self.cum[np.clip(idx, 0, None)], 0.0)


def credible_level(post: Posterior, values) -> np.ndarray:
    """HPD level (0..1) of ln-density values: the mass of all leaves at least that dense."""
    return _Sorted(post).level_of(values)


def disc_log_mass(post: Posterior, x: float, y: float, radius_m: float) -> float:
    """ln posterior mass within `radius_m` of (x, y). Leaves count when their centre is inside the
    disc; a leaf larger than the radius contributes the fraction pi r^2 / s^2 of its mass if it
    contains the point."""
    spec = post.spec
    parts: list[np.ndarray] = []

    def level_part(level: int, values_fn, covered_fn) -> None:
        s = GRID_LEVELS_M[level]
        if s > radius_m:
            ix, iy = xy_to_cells(spec, x, y, level)
            v = values_fn(np.array([int(ix)]), np.array([int(iy)]))
            ok = ~covered_fn(np.array([int(ix)]), np.array([int(iy)]))
            if ok[0] and np.isfinite(v[0]):
                parts.append(np.array([v[0] + 2 * math.log(s) + math.log(math.pi * radius_m**2 / s**2)]))
            return
        ix0, iy0, nx, ny = window_for_xy_box(spec, level, x - radius_m, y - radius_m, x + radius_m,
                                             y + radius_m)
        gj, gi = np.mgrid[iy0: iy0 + ny, ix0: ix0 + nx]
        cx, cy = cell_centers_xy(spec, level, gi, gj)
        m = (cx - x) ** 2 + (cy - y) ** 2 <= radius_m**2
        gi, gj = gi[m], gj[m]
        if gi.size == 0:
            return
        v = values_fn(gi, gj)
        ok = ~covered_fn(gi, gj) & np.isfinite(v)
        if ok.any():
            parts.append(v[ok] + 2 * math.log(s))

    b = post.base_level
    nyb, nxb = post.base.shape
    assert post.covered is not None
    cov = post.covered

    def base_vals(gi, gj):
        inb = (gi >= 0) & (gi < nxb) & (gj >= 0) & (gj < nyb)
        out = np.full(gi.shape, -np.inf)
        out[inb] = post.base[gj[inb], gi[inb]]
        return out

    def base_cov(gi, gj):
        inb = (gi >= 0) & (gi < nxb) & (gj >= 0) & (gj < nyb)
        out = np.zeros(gi.shape, dtype=bool)
        out[inb] = cov[gj[inb], gi[inb]]
        return out

    level_part(b, base_vals, base_cov)
    if post.fine_level is not None and post.tiles:
        fl = post.fine_level

        def fine_vals(gi, gj):
            out = np.full(gi.shape, -np.inf)
            tx, ty = gi // TILE, gj // TILE
            for key in set(zip(tx.tolist(), ty.tolist(), strict=True)):
                p = post.tiles.get(key)
                if p is None:
                    continue
                sel = (tx == key[0]) & (ty == key[1])
                i, j = gi[sel] - p.ix0, gj[sel] - p.iy0
                inb = (i >= 0) & (i < p.nx) & (j >= 0) & (j < p.ny)
                vals = np.full(i.shape, -np.inf)
                vals[inb] = p.values[j[inb], i[inb]]
                out[sel] = vals
            return out

        level_part(fl, fine_vals, lambda gi, gj: np.zeros(gi.shape, dtype=bool))
    if not parts:
        return -math.inf
    allv = np.concatenate(parts)
    m = allv.max()
    return float(m + math.log(np.exp(allv - m).sum()))


# ------------------------------------------------------------------------------------------------
# HPD regions
# ------------------------------------------------------------------------------------------------


def _mask_polygons(post: Posterior, level: int, ix0: int, iy0: int, mask: np.ndarray) -> list:
    import rasterio.features as rf
    from shapely.geometry import shape

    if not mask.any():
        return []
    ny = mask.shape[0]
    src = mask[::-1].astype(np.uint8)
    tr = north_up_transform(post.spec, level, ix0, iy0, ny)
    return [shape(g) for g, _ in rf.shapes(src, mask=src.astype(bool), transform=tr)]


def compute_hpd(post: Posterior, masses: Sequence[float] = (0.5, 0.9), polygons: bool = True,
                ) -> list[HPDRegion]:
    """HPD regions at the given credible masses (leaf resolution)."""
    from shapely.ops import unary_union

    srt = _Sorted(post)
    total = srt.cum[-1] if srt.cum.size else 0.0
    out = []
    for a in masses:
        if not 0.0 < a <= 1.0:
            raise ValueError("HPD mass must be in (0, 1]")
        idx = min(int(np.searchsorted(srt.cum, a * total, side="left")), srt.cum.size - 1)
        t = float(srt.v[idx])
        # Area/mass of the minimal leaf set in density order (ties on a plateau are split); the
        # polygon includes every leaf with density >= threshold.
        region = HPDRegion(mass=float(a), log_threshold=t, area_m2=float(srt.area[: idx + 1].sum()),
                           actual_mass=float(srt.cum[idx]))
        if polygons:
            assert post.covered is not None
            geoms = _mask_polygons(post, post.base_level, 0, 0, (post.base >= t) & ~post.covered)
            for p in post.tiles.values():
                geoms += _mask_polygons(post, p.level, p.ix0, p.iy0, p.values >= t)
            region.geometry_xy = unary_union(geoms) if geoms else None
        out.append(region)
    return out


# ------------------------------------------------------------------------------------------------
# Candidates
# ------------------------------------------------------------------------------------------------


def _local_maxima(post: Posterior) -> tuple[np.ndarray, ...]:
    """Pool of local maxima at leaf resolution: (value, x, y, level, ix, iy)."""
    assert post.covered is not None
    b = post.base
    mf = maximum_filter(b, size=3, mode="constant", cval=-np.inf)
    jj, ii = np.nonzero((b == mf) & np.isfinite(b) & ~post.covered)
    vals = [b[jj, ii]]
    lv = [np.full(ii.size, post.base_level)]
    ixs, iys = [ii.astype(np.int64)], [jj.astype(np.int64)]
    if post.fine_level is not None:
        fl = post.fine_level
        for p in post.tiles.values():
            halo = post.field.render(fl, p.ix0 - 1, p.iy0 - 1, p.nx + 2, p.ny + 2, aggregate=False)
            m = maximum_filter(halo, size=3, mode="constant", cval=-np.inf)[1:-1, 1:-1]
            core = halo[1:-1, 1:-1]
            j, i = np.nonzero((core == m) & np.isfinite(core))
            vals.append(core[j, i])
            lv.append(np.full(i.size, fl))
            ixs.append(i.astype(np.int64) + p.ix0)
            iys.append(j.astype(np.int64) + p.iy0)
    v = np.concatenate(vals)
    level = np.concatenate(lv).astype(np.int64)
    ix, iy = np.concatenate(ixs), np.concatenate(iys)
    sizes = np.asarray(GRID_LEVELS_M, dtype=np.float64)[level]
    x = post.spec.x0 + (ix + 0.5) * sizes
    y = post.spec.y0 + (iy + 0.5) * sizes
    return v, x, y, level, ix, iy


def _build(post: Posterior, srt: _Sorted, x: float, y: float, value: float, cell: CellId,
           mass_radius_m: float, rank_by: str) -> FusionCandidate:
    lat, lon = xy_to_latlon(post.spec, x, y)
    lm = disc_log_mass(post, x, y, mass_radius_m)
    ev = evidence_at(post, x, y)
    flights: list[str] = []
    t_s = None
    # Prefer acoustic attrs for flights / T_s.
    for _, row in sorted(ev["attrs"].items(), key=lambda kv: _module_of(post, kv[0]) != "acoustic"):
        if not flights and row.get("flights") is not None:
            flights = [str(f) for f in row["flights"]]
        if t_s is None and row.get("t_s") is not None:
            ts = row["t_s"]
            t_s = ts if isinstance(ts, datetime) else datetime.fromtimestamp(float(ts), tz=UTC)
    score = lm if rank_by == "mass" else value
    return FusionCandidate(
        rank=0, lat=float(lat), lon=float(lon), log_density=float(value),
        prob_mass=float(math.exp(lm)) if math.isfinite(lm) else 0.0,
        contributions=ev["contributions"], flights=flights, t_s=t_s,
        hpd_level=float(srt.level_of(value)), x_m=float(x), y_m=float(y), cell=cell,
        mass_radius_m=mass_radius_m, rank_score=float(score),
        evidence={"layers": ev["layers"], "attrs": _jsonable(ev["attrs"])},
    )


def _module_of(post: Posterior, layer_name: str) -> str:
    for lay, _ in post.layers:
        if lay.name == layer_name:
            return lay.module
    return ""


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _finish(cands: list[FusionCandidate]) -> list[FusionCandidate]:
    """Order (pinned first, then by rank_score), assign ranks and margins."""
    pinned = [c for c in cands if c.pinned]
    rest = sorted((c for c in cands if not c.pinned), key=lambda c: -c.rank_score)
    for c in pinned:
        c.rank = 0
    for i, c in enumerate(rest):
        c.rank = i + 1
        nxt = rest[i + 1] if i + 1 < len(rest) else None
        gap = c.rank_score - nxt.rank_score if nxt is not None else math.nan
        c.margin_to_runner_up = float(gap) if math.isfinite(gap) else None
    return pinned + rest


def find_candidates(post: Posterior, k: int = 10, nms_radius_m: float = 300.0, rank_by: str = "mass",
                    mass_radius_m: float | None = None, min_prob_mass: float = 0.0,
                    pool_size: int = 20000) -> list[FusionCandidate]:
    """Top-k candidates by NMS over posterior local maxima.

    rank_by: "mass" (ln posterior mass within mass_radius_m, default the NMS radius; robust to
    very narrow spikes) or "density" (peak ln density). Candidates are selected by peak density;
    with "mass" up to 3k peaks are considered and the best k by mass are kept.
    """
    if rank_by not in ("mass", "density"):
        raise ValueError("rank_by must be 'mass' or 'density'")
    mr = nms_radius_m if mass_radius_m is None else mass_radius_m
    v, x, y, level, ix, iy = _local_maxima(post)
    if v.size == 0:
        return []
    if v.size > pool_size:
        keep = np.argpartition(-v, pool_size)[:pool_size]
        v, x, y, level, ix, iy = (a[keep] for a in (v, x, y, level, ix, iy))
    order = np.argsort(-v, kind="stable")
    tree = cKDTree(np.column_stack([x, y]))
    suppressed = np.zeros(v.size, dtype=bool)
    n_sel = k if rank_by == "density" else max(3 * k, k + 10)
    picked: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        picked.append(int(i))
        suppressed[tree.query_ball_point((x[i], y[i]), nms_radius_m)] = True
        if len(picked) >= n_sel:
            break
    srt = _Sorted(post)
    cands = [_build(post, srt, float(x[i]), float(y[i]), float(v[i]),
                    (int(level[i]), int(ix[i]), int(iy[i])), mr, rank_by) for i in picked]
    cands = [c for c in cands if (c.prob_mass or 0.0) >= min_prob_mass]
    cands.sort(key=lambda c: -c.rank_score)
    return _finish(cands[:k])


def pinned_candidate(post: Posterior, lat: float, lon: float, note: str = "pinned",
                     mass_radius_m: float = 300.0) -> FusionCandidate:
    """Candidate #0 at a fixed point (e.g. trusted GPS); it still gets the full evidence breakdown,
    so the acoustic score there is a cross-check."""
    from .grid import latlon_to_xy

    xa, ya = latlon_to_xy(post.spec, lat, lon)
    x, y = float(xa), float(ya)
    val = float(post.value_at_xy(x, y))
    lvl = post.fine_level if post.fine_level is not None else post.base_level
    cix, ciy = xy_to_cells(post.spec, x, y, lvl)
    c = _build(post, _Sorted(post), x, y, val, (lvl, int(cix), int(ciy)), mass_radius_m, "mass")
    c.lat, c.lon = float(lat), float(lon)
    c.pinned = True
    c.notes.append(note)
    return c


def rerank(cands: Sequence[FusionCandidate], name: str, logliks: Sequence[float | None],
           weight: float = 1.0) -> list[FusionCandidate]:
    """Apply late evidence (one ln-likelihood per candidate, None = not available) and re-sort.

    Adds `weight * ell` to `rank_score`, `contributions[name]` and `late[name]`. `prob_mass` is
    reweighted by exp(weight * ell) and renormalised so the candidates' total mass is unchanged.
    `log_density` stays the raster value (before late evidence).
    """
    cands = [c.model_copy(deep=True) for c in cands]
    if len(logliks) != len(cands):
        raise ValueError("need one log-likelihood per candidate")
    terms = []
    for c, ell in zip(cands, logliks, strict=True):
        if ell is None or not math.isfinite(float(ell)):
            c.notes.append(f"{name}: not available")
            terms.append(0.0)
            continue
        t = weight * float(ell)
        terms.append(t)
        c.late[name] = t
        c.contributions[name] = c.contributions.get(name, 0.0) + t
        c.rank_score += t
    masses = np.array([c.prob_mass or 0.0 for c in cands])
    total = masses.sum()
    if total > 0:
        tt = np.array(terms)
        new = masses * np.exp(tt - tt.max())
        if new.sum() > 0:
            new *= total / new.sum()
            for c, m in zip(cands, new, strict=True):
                c.prob_mass = float(m)
    return _finish(cands)
