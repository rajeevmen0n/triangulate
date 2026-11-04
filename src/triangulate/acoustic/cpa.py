"""Level 2 of the acoustic search: candidate cells, CPA / retarded time, speed + distance gates
(plan 3.3.2-3.3.3, WP09). Output: the `cpa.parquet` table (`CPA_SCHEMA`).

Pipeline (`cpa_search`)
1. **Gates** (`EventGate`), one per selected event: CPA speed in v_range +- 15 m/s (wind margin, applied
   to ADS-B ground speed), slant distance in d_range, heard CPA time t_h in the window implied by the
   start-time constraint T_s in [T_lo, T_hi] (t_h = T_s + tau_k, plus the matching slack).
2. **Candidate cells** (`candidate_cells`): cells of the job grid (default level 3 = 200 m) whose centre
   lies within R_k = d_max,k + 1 km (horizontal) of a track sample that passes event k's speed and time
   gate. Implemented as a dense Euclidean distance transform over the region window (equivalent to the
   plan's cKDTree ball query, and O(cells)). The union over events is kept.
3. **Observer height**: Z_obs = DEM(cell) + N_geoid + h_obs (`grid.observer_cells`), exact ENU position
   of every candidate cell in the tracks' frame (slant ranges are exact 3-D, earth curvature included).
4. **CPA kernel** (numba, `_cpa_segment`): for each 1-s track interval [t_i, t_i+1] the cells whose range
   r(t) has a *local minimum* inside it are exactly those with g_i < 0 <= g_i+1, where
   g = (P - X) . V = r dr/dt. That set is a thin strip (a wedge in turns) perpendicular to the track, so
   the kernel rasterises the strip row by row (conservative half-plane + disc bounds in cell coordinates)
   and tests each candidate cell exactly. Every local minimum is found, not just the global one
   (racetracks, touch-and-goes). Each minimum is refined to < 1e-6 s by Illinois regula falsi on g(t)
   along the cubic-Hermite track (`physics.nb_track_state`), then
   t_h = t_cpa + d/c (heard / retarded CPA), d = slant range, v = 3-D speed, gs = ADS-B ground speed
   (linearly interpolated; falls back to the horizontal track speed), agl = height above the cell.
   Work is ~O(samples x strip cells), independent of how long a flight lingers near a cell.
   Minima are detected at 1-s node resolution: a minimum/maximum pair inside one interval (a sub-metre
   range wiggle, e.g. at a curvature jump of a finite-difference track) is not a pass; a minimum at
   the very first interval of a segment is reported if dr/dt changes sign there.
5. **Eligibility**: bit k of `eligible` is set when the pass satisfies gate k. Only rows with at least
   one bit are kept (pass `EventGate.window(...)` to get *all* passes within a range, e.g. for the
   silence filter of WP10).

Time bases: track `t` and all output times are float epoch seconds (UTC); kernels work relative to the
segment start for precision.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

from ..core.types import Event
from ..fusion.grid import GridSpec, cell_size, pack_cells
from .grid import (
    SEARCH_LEVEL,
    AffineMap,
    CellSet,
    GroundFn,
    ObserverCells,
    cells_within,
    fit_lattice_map,
    max_alt_shift_m,
    observer_cells,
    region_window,
)
from .physics import C_DEFAULT, Track, nb_track_state

__all__ = [
    "CPA_SCHEMA", "V_MARGIN_MS", "CAND_MARGIN_M", "EventGate", "TrackSeg", "track_segments",
    "candidate_cells", "compute_cpa", "cpa_search", "CpaResult", "eligible_expr", "write_cpa", "read_cpa",
]

V_MARGIN_MS = 15.0  # wind margin on the speed gate (plan 3.3.2)
CAND_MARGIN_M = 1000.0  # R_h = d_max + 1 km (plan 3.3.3)

CPA_SCHEMA: dict[str, type[pl.DataType]] = {
    "cell": pl.Int64,  # fusion.grid.pack_cells(level, ix, iy)
    "level": pl.Int8, "ix": pl.Int32, "iy": pl.Int32,
    "icao": pl.String, "seg": pl.Int32,
    "pass": pl.Int64,  # pass key: floor(t_cpa) epoch second (unique per cell, icao)
    "t_cpa": pl.Float64,  # emission time of the CPA (epoch s)
    "t_h": pl.Float64,  # heard (retarded) CPA time t_cpa + d/c (epoch s)
    "d": pl.Float32,  # slant CPA distance (m)
    "v": pl.Float32,  # 3-D track speed at the CPA (m/s)
    "gs": pl.Float32,  # ground speed at the CPA (m/s), used by the speed gate
    "agl": pl.Float32,  # aircraft height above the cell's ground at the CPA (m)
    "eligible": pl.UInt32,  # bit k: passes gate k
}


# --- Gates ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EventGate:
    """Per-event physics gate. All bounds are *effective* (margins already applied), inclusive.

    th_lo/th_hi bound the heard CPA time t_h (epoch s); +-inf = unconstrained.
    """

    id: str
    v_lo: float
    v_hi: float
    d_lo: float
    d_hi: float
    th_lo: float = -math.inf
    th_hi: float = math.inf

    def __post_init__(self) -> None:
        if not (self.v_lo <= self.v_hi and 0 <= self.d_lo <= self.d_hi and self.th_lo <= self.th_hi):
            raise ValueError(f"bad gate {self}")
        if not math.isfinite(self.d_hi):
            raise ValueError("d_hi must be finite")

    @classmethod
    def from_ranges(cls, id: str, v: tuple[float, float], d: tuple[float, float],
                    tau: tuple[float, float] | None = None, t_s: tuple[float, float] | None = None,
                    slack_s: float = 5.0, v_margin: float = V_MARGIN_MS, d_margin: float = 0.0) -> EventGate:
        """Gate from fitted ranges. `tau` = audio-position range of the heard CPA (s), `t_s` = window
        for the absolute time of audio sample 0 (epoch s): t_h in [t_s0 + tau0 - slack, t_s1 + tau1 + slack].
        """
        th_lo, th_hi = -math.inf, math.inf
        if t_s is not None:
            if tau is None:
                raise ValueError("t_s window needs the event's tau range")
            th_lo = t_s[0] + tau[0] - slack_s
            th_hi = t_s[1] + tau[1] + slack_s
        return cls(id=id, v_lo=max(v[0] - v_margin, 0.0), v_hi=v[1] + v_margin,
                   d_lo=max(d[0] - d_margin, 0.0), d_hi=d[1] + d_margin, th_lo=th_lo, th_hi=th_hi)

    @classmethod
    def from_event(cls, ev: Event, t_s: tuple[float, float] | None = None, slack_s: float = 5.0,
                   v_margin: float = V_MARGIN_MS, d_margin: float = 0.0) -> EventGate:
        return cls.from_ranges(ev.id, (ev.v_range.lo, ev.v_range.hi), (ev.d_range.lo, ev.d_range.hi),
                               (ev.t_h_range.lo, ev.t_h_range.hi), t_s, slack_s, v_margin, d_margin)

    @classmethod
    def window(cls, id: str, r_max: float, th_lo: float = -math.inf, th_hi: float = math.inf) -> EventGate:
        """Any speed, any pass with d <= r_max heard in [th_lo, th_hi] (e.g. silence-filter passes)."""
        return cls(id=id, v_lo=0.0, v_hi=math.inf, d_lo=0.0, d_hi=r_max, th_lo=th_lo, th_hi=th_hi)


def eligible_expr(k: int) -> pl.Expr:
    """Polars predicate: row passes gate k."""
    return (pl.col("eligible") & (1 << k)) != 0


# --- Track segments -------------------------------------------------------------------------------


@dataclass(eq=False)
class TrackSeg:
    """One continuous track segment in ENU metres: t (n,) epoch s, pos/vel (n, 3), gs (n,) m/s (NaN ok)."""

    icao: str
    seg: int
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    gs: np.ndarray
    spd: np.ndarray = field(init=False)  # speed used by the gate at the nodes

    def __post_init__(self) -> None:
        self.t = np.ascontiguousarray(self.t, dtype=np.float64)
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        self.vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        self.gs = np.ascontiguousarray(self.gs, dtype=np.float64)
        self.spd = np.where(np.isfinite(self.gs), self.gs, np.hypot(self.vel[:, 0], self.vel[:, 1]))


def track_segments(tracks: pl.DataFrame | Sequence[Track] | Sequence[TrackSeg], min_samples: int = 3,
                   ) -> list[TrackSeg]:
    """Normalise the input: a prepared frame (`adsb.tracks.PREPARED_SCHEMA`: icao, seg, t, e, n, u, gs_ms),
    a list of `physics.Track` (label -> icao, gs = horizontal node speed), or TrackSegs.

    Node velocities of a prepared frame are 2nd-order finite differences (as `Track.from_positions`).
    """
    out: list[TrackSeg] = []
    if isinstance(tracks, pl.DataFrame):
        if tracks.height == 0:
            return out
        df = tracks.sort("icao", "seg", "t")
        has_gs = "gs_ms" in df.columns
        for (icao, seg), g in df.partition_by(["icao", "seg"], as_dict=True, maintain_order=True).items():
            if g.height < max(min_samples, 2):
                continue
            t = g["t"].to_numpy().astype(np.float64)
            pos = np.column_stack([g["e"].to_numpy(), g["n"].to_numpy(), g["u"].to_numpy()]).astype(float)
            ok = np.concatenate([[True], np.diff(t) > 0])
            t, pos = t[ok], pos[ok]
            if t.size < 2:
                continue
            vel = np.gradient(pos, t, axis=0, edge_order=2 if t.size > 2 else 1)
            gs = g["gs_ms"].to_numpy().astype(np.float64)[ok] if has_gs else np.full(t.size, np.nan)
            out.append(TrackSeg(str(icao), int(seg), t, pos, vel, gs))
        return out
    for i, tr in enumerate(tracks):
        if isinstance(tr, TrackSeg):
            out.append(tr)
            continue
        out.append(TrackSeg(tr.label or f"trk{i}", 0, tr.t, tr.pos, tr.vel,
                            np.hypot(tr.vel[:, 0], tr.vel[:, 1])))
    return out


def _gate_arrays(gates: Sequence[EventGate]):
    if not 1 <= len(gates) <= 32:
        raise ValueError("need 1..32 gates")
    f = lambda name: np.array([getattr(g, name) for g in gates], dtype=np.float64)  # noqa: E731
    return f("v_lo"), f("v_hi"), f("d_lo"), f("d_hi"), f("th_lo"), f("th_hi")


def _active_rmax(seg: TrackSeg, gates: Sequence[EventGate], c: float) -> np.ndarray:
    """Per node: max d_hi over the gates whose speed + emission-time window the node passes (0 = none).

    A CPA heard at t_h in [th_lo, th_hi] at range <= d_hi was emitted at t_cpa in
    [th_lo - d_hi/c, th_hi]; one extra second each side covers the interval endpoints.
    """
    r = np.zeros(seg.t.size)
    for g in gates:
        ok = ((seg.spd >= g.v_lo) & (seg.spd <= g.v_hi)
              & (seg.t >= g.th_lo - g.d_hi / c - 1.0) & (seg.t <= g.th_hi + 1.0))
        r = np.where(ok, np.maximum(r, g.d_hi), r)
    return r


# --- numba kernel ---------------------------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _grow(oc, om, of, cap):
    nc = np.empty(cap * 2, np.int64)
    nm = np.empty(cap * 2, np.int64)
    nf = np.empty((cap * 2, of.shape[1]), np.float64)
    nc[:cap] = oc
    nm[:cap] = om
    nf[:cap] = of
    return nc, nm, nf, cap * 2


@nb.njit(cache=True, nogil=True)
def _cpa_segment(tt, pos, vel, spd, rmax_node, A, B, eps, sig, idx_map, ix0, iy0, cxyz, cup,
                 zmin, zmax, c, h_obs, gvlo, gvhi, gdlo, gdhi, gtlo, gthi, tref):
    """All gated local range minima of one segment against the cells of one block.

    Returns (cell_idx, mask, F) with F columns: t_cpa, t_h (absolute), d, v, gs, agl.
    """
    n = tt.shape[0]
    ny, nx = idx_map.shape
    K = gvlo.shape[0]
    cap = 256
    oc = np.empty(cap, np.int64)
    om = np.empty(cap, np.int64)
    of = np.empty((cap, 6), np.float64)
    cnt = 0
    a00, a01, a02 = A[0, 0], A[0, 1], A[0, 2]
    a10, a11, a12 = A[1, 0], A[1, 1], A[1, 2]
    b00, b01, b02 = B[0, 0], B[0, 1], B[0, 2]
    b10, b11, b12 = B[1, 0], B[1, 1], B[1, 2]
    for i in range(n - 1):
        rmax = max(rmax_node[i], rmax_node[i + 1])
        if rmax <= 0.0:
            continue
        p0x, p0y, p0z = pos[i, 0], pos[i, 1], pos[i, 2]
        p1x, p1y, p1z = pos[i + 1, 0], pos[i + 1, 1], pos[i + 1, 2]
        v0x, v0y, v0z = vel[i, 0], vel[i, 1], vel[i, 2]
        v1x, v1y, v1z = vel[i + 1, 0], vel[i + 1, 1], vel[i + 1, 2]
        L = math.sqrt((p1x - p0x) ** 2 + (p1y - p0y) ** 2 + (p1z - p0z) ** 2)
        hz = min(p0z, p1z) - 0.6 * L - zmax  # lower bound of the height above every cell
        if hz >= rmax:
            continue
        hmin = hz if hz > 0.0 else 0.0
        rh = math.sqrt(rmax * rmax - hmin * hmin)
        mx = 0.5 * (p0x + p1x)
        my = 0.5 * (p0y + p1y)
        fu = a00 * mx + a01 * my + a02
        fv = a10 * mx + a11 * my + a12
        rho = sig * (rh + 0.6 * L + eps + 1.0)
        # g(X) = (P - X) . V with X_xy = B [u, v, 1]: linear in (u, v) plus a z term bounded by [zmin, zmax]
        ca0 = -(b00 * v0x + b10 * v0y)
        cb0 = -(b01 * v0x + b11 * v0y)
        zt0 = min((p0z - zmax) * v0z, (p0z - zmin) * v0z)
        cc0 = p0x * v0x + p0y * v0y - b02 * v0x - b12 * v0y + zt0 - math.sqrt(v0x * v0x + v0y * v0y) * eps
        ca1 = -(b00 * v1x + b10 * v1y)
        cb1 = -(b01 * v1x + b11 * v1y)
        zt1 = max((p1z - zmax) * v1z, (p1z - zmin) * v1z)
        cc1 = p1x * v1x + p1y * v1y - b02 * v1x - b12 * v1y + zt1 + math.sqrt(v1x * v1x + v1y * v1y) * eps
        jy0 = int(math.ceil(fv - rho - 0.5))
        jy1 = int(math.floor(fv + rho - 0.5))
        if jy0 < iy0:
            jy0 = iy0
        if jy1 > iy0 + ny - 1:
            jy1 = iy0 + ny - 1
        for iy in range(jy0, jy1 + 1):
            vc = iy + 0.5
            dv = vc - fv
            rem = rho * rho - dv * dv
            if rem < 0.0:
                continue
            half = math.sqrt(rem)
            lo = fu - half
            hi = fu + half
            k0 = cb0 * vc + cc0  # need ca0 u + k0 < 0
            if ca0 > 0.0:
                hi = min(hi, -k0 / ca0)
            elif ca0 < 0.0:
                lo = max(lo, -k0 / ca0)
            elif k0 >= 0.0:
                continue
            k1 = cb1 * vc + cc1  # need ca1 u + k1 >= 0
            if ca1 > 0.0:
                lo = max(lo, -k1 / ca1)
            elif ca1 < 0.0:
                hi = min(hi, -k1 / ca1)
            elif k1 < 0.0:
                continue
            if hi < lo:
                continue
            jx0 = int(math.ceil(lo - 0.5))
            jx1 = int(math.floor(hi - 0.5))
            if jx0 < ix0:
                jx0 = ix0
            if jx1 > ix0 + nx - 1:
                jx1 = ix0 + nx - 1
            row = iy - iy0
            for ix in range(jx0, jx1 + 1):
                j = idx_map[row, ix - ix0]
                if j < 0:
                    continue
                X0, X1, X2 = cxyz[j, 0], cxyz[j, 1], cxyz[j, 2]
                d0x, d0y, d0z = p0x - X0, p0y - X1, p0z - X2
                ga = d0x * v0x + d0y * v0y + d0z * v0z
                if ga >= 0.0:
                    continue
                d1x, d1y, d1z = p1x - X0, p1y - X1, p1z - X2
                gb = d1x * v1x + d1y * v1y + d1z * v1z
                if gb < 0.0:
                    continue
                r0 = math.sqrt(d0x * d0x + d0y * d0y + d0z * d0z)
                r1 = math.sqrt(d1x * d1x + d1y * d1y + d1z * d1z)
                if min(r0, r1) - 0.6 * L > rmax:
                    continue
                # Illinois regula falsi for g(t) = 0 on [tt[i], tt[i+1]]
                ta = tt[i]
                tb = tt[i + 1]
                fa = ga
                fb = gb
                ts = tb
                hint = i
                px, py, pz, vx, vy, vz = p1x, p1y, p1z, v1x, v1y, v1z
                if fb != 0.0:
                    side = 0
                    for _it in range(60):
                        ts = (ta * fb - tb * fa) / (fb - fa)
                        hint, px, py, pz, vx, vy, vz = nb_track_state(tt, pos, vel, ts, hint)
                        gg = (px - X0) * vx + (py - X1) * vy + (pz - X2) * vz
                        if gg < 0.0:
                            ta = ts
                            fa = gg
                            if side == -1:
                                fb *= 0.5
                            side = -1
                        else:
                            tb = ts
                            fb = gg
                            if side == 1:
                                fa *= 0.5
                            side = 1
                        if tb - ta < 1e-7 or abs(gg) <= 1e-9 * (vx * vx + vy * vy + vz * vz + 1.0):
                            break
                ex, ey, ez = px - X0, py - X1, pz - X2
                d = math.sqrt(ex * ex + ey * ey + ez * ez)
                sp = math.sqrt(vx * vx + vy * vy + vz * vz)
                w = (ts - tt[i]) / (tt[i + 1] - tt[i])
                if w < 0.0:
                    w = 0.0
                elif w > 1.0:
                    w = 1.0
                gsv = spd[i] + (spd[i + 1] - spd[i]) * w
                th = tref + ts + d / c
                m = 0
                for k in range(K):
                    if (gdlo[k] <= d <= gdhi[k] and gvlo[k] <= gsv <= gvhi[k]
                            and gtlo[k] <= th <= gthi[k]):
                        m |= 1 << k
                if m == 0:
                    continue
                if cnt == cap:
                    oc, om, of, cap = _grow(oc, om, of, cap)
                oc[cnt] = j
                om[cnt] = m
                of[cnt, 0] = tref + ts
                of[cnt, 1] = th
                of[cnt, 2] = d
                of[cnt, 3] = sp
                of[cnt, 4] = gsv
                of[cnt, 5] = ex * cup[j, 0] + ey * cup[j, 1] + ez * cup[j, 2] + h_obs
                cnt += 1
    return oc[:cnt], om[:cnt], of[:cnt]


# --- Candidate cells ------------------------------------------------------------------------------


def candidate_cells(segments: Sequence[TrackSeg], origin: tuple[float, float], spec: GridSpec,
                    gates: Sequence[EventGate], level: int = SEARCH_LEVEL,
                    margin_m: float = CAND_MARGIN_M, c: float = C_DEFAULT,
                    stats: dict | None = None) -> CellSet:
    """Region cells within R_k = d_hi,k + margin_m (horizontal) of a node passing gate k's speed and
    time window, unioned over the gates. Lossless for `compute_cpa` when margin_m >= 0: any pass that
    can satisfy gate k has its CPA point within d_hi,k of a gate-k node."""
    s = cell_size(level)
    r_all = max(g.d_hi for g in gates) + margin_m
    pad = int(math.ceil(r_all / s)) + 2
    win_pad = region_window(spec, level, pad)
    amap = fit_lattice_map(spec, level, origin, win_pad)
    zmax = max((float(sg.pos[:, 2].max()) for sg in segments), default=0.0)
    ext = max((float(np.abs(sg.pos[:, :2]).max()) for sg in segments), default=0.0)
    slop_m = amap.eps_m + max_alt_shift_m(zmax, ext + r_all)
    act = [np.concatenate([_node_mask(sg, g, c) for sg in segments]) if segments else np.zeros(0, bool)
           for g in gates]
    allp = np.concatenate([sg.pos[:, :2] for sg in segments]) if segments else np.zeros((0, 2))
    u, v = amap.to_cells(allp[:, 0], allp[:, 1])
    ix0, iy0, nx, ny = win_pad
    crop = (slice(pad, pad + ny - 2 * pad), slice(pad, pad + nx - 2 * pad))
    union = np.zeros((ny - 2 * pad, nx - 2 * pad), dtype=bool)
    per_event = []
    for g, a in zip(gates, act, strict=True):
        m = cells_within(u[a], v[a], win_pad, (g.d_hi + margin_m + slop_m) / s)[crop]
        per_event.append(int(m.sum()))
        union |= m
    if stats is not None:
        t_any = np.zeros(allp.shape[0], bool)
        t_all = np.concatenate([sg.t for sg in segments]) if segments else np.zeros(0)
        for g in gates:  # time window only (no speed gate): "cells near any track"
            t_any |= (t_all >= g.th_lo - g.d_hi / c - 1.0) & (t_all <= g.th_hi + 1.0)
        near = cells_within(u[t_any], v[t_any], win_pad, (r_all + slop_m) / s)[crop]
        stats.update(cells_region=int(union.size), cells_near_tracks=int(near.sum()),
                     cells_speed_gated={g.id: n for g, n in zip(gates, per_event, strict=True)},
                     cells_candidates=int(union.sum()), lattice_eps_m=amap.eps_m)
    return CellSet.from_mask(spec, level, union, 0, 0)


def _node_mask(seg: TrackSeg, g: EventGate, c: float) -> np.ndarray:
    return ((seg.spd >= g.v_lo) & (seg.spd <= g.v_hi)
            & (seg.t >= g.th_lo - g.d_hi / c - 1.0) & (seg.t <= g.th_hi + 1.0))


# --- CPA over a set of observer cells -------------------------------------------------------------


def _blocks(cells: CellSet, block: int):
    bx = cells.ix // block
    by = cells.iy // block
    key = (bx - bx.min()) * (int(by.max() - by.min()) + 1) + (by - by.min()) if len(cells) else bx
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    return np.split(order, cuts) if len(cells) else []


def compute_cpa(segments: Sequence[TrackSeg] | pl.DataFrame | Sequence[Track], obs: ObserverCells,
                gates: Sequence[EventGate], c: float = C_DEFAULT, n_threads: int | None = None,
                block: int = 1024) -> pl.DataFrame:
    """Every local range minimum of every segment at every observer cell that passes >= 1 gate.

    Rows follow `CPA_SCHEMA`. Order: by block, then segment, then time.
    """
    segs = track_segments(segments)
    gvlo, gvhi, gdlo, gdhi, gtlo, gthi = _gate_arrays(gates)
    cells = obs.cells
    if len(cells) == 0 or not segs:
        return pl.DataFrame(schema=CPA_SCHEMA)
    rmax_nodes = [_active_rmax(sg, gates, c) for sg in segs]
    seg_box = []
    for sg, rm in zip(segs, rmax_nodes, strict=True):
        a = rm > 0
        if not a.any():
            seg_box.append(None)
            continue
        a = a | np.concatenate([a[1:], [False]]) | np.concatenate([[False], a[:-1]])
        p = sg.pos[a]
        seg_box.append((p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()))
    r_all = float(gdhi.max())
    tasks = []
    for sel in _blocks(cells, block):
        bix, biy = cells.ix[sel], cells.iy[sel]
        jx0, jy0 = int(bix.min()), int(biy.min())
        nxb, nyb = int(bix.max()) - jx0 + 1, int(biy.max()) - jy0 + 1
        idx_map = np.full((nyb, nxb), -1, dtype=np.int64)
        idx_map[biy - jy0, bix - jx0] = np.arange(sel.size)
        X = obs.xyz[sel]
        amap = _block_map(bix, biy, X, cells.size_m)
        xmin, ymin = X[:, 0].min() - r_all, X[:, 1].min() - r_all
        xmax, ymax = X[:, 0].max() + r_all, X[:, 1].max() + r_all
        cx = np.ascontiguousarray(X)
        cu = np.ascontiguousarray(obs.up[sel])
        zmin, zmax = float(X[:, 2].min()), float(X[:, 2].max())
        for si, bx in enumerate(seg_box):
            if bx is None or bx[2] < xmin or bx[0] > xmax or bx[3] < ymin or bx[1] > ymax:
                continue
            tasks.append((si, sel, (amap, idx_map, jx0, jy0, cx, cu, zmin, zmax)))

    def run(task):
        si, sel, (amap, idx_map, jx0, jy0, cx, cu, zmin, zmax) = task
        sg = segs[si]
        tref = float(sg.t[0])
        oc, om, of = _cpa_segment(sg.t - tref, sg.pos, sg.vel, sg.spd, rmax_nodes[si], amap.A, amap.B,
                                  amap.eps_m, amap.stretch, idx_map, jx0, jy0, cx, cu, zmin, zmax,
                                  float(c), float(obs.h_obs), gvlo, gvhi, gdlo, gdhi, gtlo, gthi, tref)
        return si, sel[oc], om, of

    nt = n_threads or min(4, os.cpu_count() or 1)
    results = [run(t) for t in tasks[:1]]  # compile / load the kernel once before going parallel
    if nt > 1 and len(tasks) > 2:
        with ThreadPoolExecutor(nt) as ex:  # the kernel releases the GIL (nogil=True)
            results += list(ex.map(run, tasks[1:]))
    else:
        results += [run(t) for t in tasks[1:]]
    results = [r for r in results if r[1].size]
    if not results:
        return pl.DataFrame(schema=CPA_SCHEMA)
    seg_idx = np.concatenate([np.full(r[1].size, r[0], np.int64) for r in results])
    cj = np.concatenate([r[1] for r in results])
    mask = np.concatenate([r[2] for r in results])
    F = np.concatenate([r[3] for r in results])
    ix, iy = cells.ix[cj], cells.iy[cj]
    icao = pl.Series("icao", [sg.icao for sg in segs], dtype=pl.String).gather(seg_idx)
    segn = np.array([sg.seg for sg in segs], dtype=np.int64)[seg_idx]
    return pl.DataFrame({
        "cell": pack_cells(cells.level, ix, iy), "level": np.full(cj.size, cells.level, np.int8),
        "ix": ix.astype(np.int32), "iy": iy.astype(np.int32), "icao": icao, "seg": segn.astype(np.int32),
        "pass": np.floor(F[:, 0]).astype(np.int64), "t_cpa": F[:, 0], "t_h": F[:, 1],
        "d": F[:, 2].astype(np.float32), "v": F[:, 3].astype(np.float32), "gs": F[:, 4].astype(np.float32),
        "agl": F[:, 5].astype(np.float32), "eligible": mask.astype(np.uint32),
    }, schema=CPA_SCHEMA)


def _block_map(ix: np.ndarray, iy: np.ndarray, X: np.ndarray, s: float) -> AffineMap:
    F = np.column_stack([ix + 0.5, iy + 0.5]).astype(np.float64)
    if ix.size >= 3 and np.linalg.matrix_rank(np.column_stack([F - F.mean(0), np.ones(len(F))])) == 3:
        return AffineMap.fit(F, X[:, :2])
    # degenerate block (1-2 cells, or collinear): nominal north-up scale anchored on the first cell,
    # with the residual over the block as eps
    L = np.eye(2) * s
    t = X[0, :2] - L @ F[0]
    eps = float(np.sqrt(((X[:, :2] - (F @ L.T + t)) ** 2).sum(axis=1)).max()) + 0.5
    Ai = np.linalg.inv(L)
    return AffineMap(A=np.column_stack([Ai, -Ai @ t]), B=np.column_stack([L, t]), eps_m=eps)


# --- Orchestration --------------------------------------------------------------------------------


@dataclass(eq=False)
class CpaResult:
    table: pl.DataFrame  # CPA_SCHEMA rows
    observers: ObserverCells  # the candidate cells and their observer positions
    gates: list[EventGate]
    stats: dict

    def cells_eligible(self, k: int) -> int:
        return int(self.table.filter(eligible_expr(k))["cell"].n_unique())

    def cells_eligible_all(self) -> int:
        """Cells with at least one eligible pass for every gate."""
        if self.table.height == 0:
            return 0
        full = (1 << len(self.gates)) - 1
        agg = self.table.group_by("cell").agg(pl.col("eligible").bitwise_or())
        return int((agg["eligible"] == full).sum())


def cpa_search(tracks: pl.DataFrame | Sequence[Track] | Sequence[TrackSeg], origin: tuple[float, float],
               spec: GridSpec, gates: Sequence[EventGate], ground_hae: GroundFn | None = None,
               h_obs: float = 5.0, level: int = SEARCH_LEVEL, margin_m: float = CAND_MARGIN_M,
               c: float = C_DEFAULT, n_threads: int | None = None) -> CpaResult:
    """Levels 2-3 of the plan for one job: candidate cells -> DEM -> CPA rows with eligibility bits.

    `tracks`: the prepared 1-s ENU frame from `adsb.tracks.prepare_trackset(..., origin)` (jets already
    removed) or equivalents; `origin` is that ENU frame's (lat, lon). `ground_hae(lat, lon)` returns the
    ground ellipsoidal height (default: Copernicus DEM + EGM2008 N).
    """
    t0 = time.perf_counter()
    segs = track_segments(tracks)
    stats: dict = {"n_segments": len(segs), "n_samples": int(sum(sg.t.size for sg in segs))}
    t1 = time.perf_counter()
    cells = candidate_cells(segs, origin, spec, gates, level, margin_m, c, stats)
    t2 = time.perf_counter()
    obs = observer_cells(cells, origin, ground_hae, h_obs)
    t3 = time.perf_counter()
    table = compute_cpa(segs, obs, gates, c, n_threads)
    t4 = time.perf_counter()
    res = CpaResult(table, obs, list(gates), stats)
    stats.update(rows=table.height,
                 cells_eligible={g.id: res.cells_eligible(k) for k, g in enumerate(gates)},
                 cells_eligible_all=res.cells_eligible_all(),
                 timing_s=dict(segments=t1 - t0, candidates=t2 - t1, dem=t3 - t2, cpa=t4 - t3, total=t4 - t0))
    return res


def write_cpa(table: pl.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(p)
    return p


def read_cpa(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path).cast(CPA_SCHEMA)  # type: ignore[arg-type]
