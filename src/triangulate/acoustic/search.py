"""Acoustic search orchestration, Levels 1-6 (plan 3.3, WP11).

    gates (WP09) -> 200 m candidate cells + CPA -> offset matching (WP10) -> silence filter (WP10)
    -> null calibration -> locked scoring at 200 m -> refine 100 -> 50 -> 25 m -> fusion layer

Refinement (plan 3.3.7). Flight tuples (one (icao, seg) per event) are ranked by their best z_sum so
far. The top ``top_tuples[i]`` tuples (default 50, 20, 10) are refined to ``refine_levels[i]``
(100, 50, 25 m): each tuple's best cells at the previous level (z_sum within ``keep_dz`` of the
tuple's best, at most ``keep_cells``) are densified with a 1-cell halo (``grid.densify``); the CPA
rows, offset match and silence filter are recomputed at the finer cells, restricted to the tuples'
flights, and re-scored with the same calibration. The 200 -> 100 m step follows the video's
"141 m worst-case offset" argument [N 2 20:12].

Fusion layer (``fusion.combine.acoustic_layer`` contract). Per (cell, tuple) the best hypothesis'
ell = lam * z_sum; per cell logsumexp over tuples; every other cell of the searched area gets
``fill = log(p_miss)``. Finer levels are added with ``layer.add_cells`` (finer cells override, a
child without a value falls back to its parent). At a finer cell, tuples that were not refined to
that level contribute their value at the finest level they reached. ``attrs`` per cell: t_s (epoch
s of audio sample 0 = o + dt*), flights, tuple, z_sum, dt, f0_k, z_k.

Public API (WP18 pipeline stage / WP22 calibration):
    acoustic_search(spec_or_map, events, tracks, grid, origin, t_s, ...) -> SearchResult
    SearchResult.tuples / .hypotheses / .layer / .summary() / .write(dir) / .module_result(dir)
    band_maps(result, tuple_rank=0, ...) -> per-event single-event band map (evidence view)
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..core.types import Event, ModuleResult
from ..fusion.combine import acoustic_layer
from ..fusion.grid import GridSpec, Layer, cell_centers_latlon, latlon_to_cells, pack_cells, unpack_cells
from .calibrate import LAMBDA_DEFAULT, P_MISS_DEFAULT, Calibration, calibrate
from .cpa import V_MARGIN_MS, CpaResult, EventGate, compute_cpa, cpa_search, track_segments
from .detect import BAD_FLAGS
from .grid import SEARCH_LEVEL, CellSet, GroundFn, default_ground_hae, densify, observer_cells
from .matching import SLACK_S, match_offsets, taus_from_events
from .physics import C_DEFAULT
from .score import PairSet, ScoreConfig, ScoreMap, TrackBank, score_hypotheses, score_pairs
from .silence import GUARD_S, silence_filter
from .spectrogram import Spectrogram

__all__ = [
    "SearchConfig",
    "SearchResult",
    "acoustic_search",
    "band_maps",
    "exempt_events_for",
    "tuple_keys",
    "build_layer",
]


@dataclass(frozen=True)
class SearchConfig:
    slack_s: float = SLACK_S
    v_margin: float = V_MARGIN_MS
    h_obs: float = 5.0
    kappa: float = 1.0
    silence_scope: str = "between"  # or "full" (whole recording minus guards)
    silence_guard: float = GUARD_S
    exempt: str = "usable"  # unselected events exempting the silence filter: usable|aircraft|all|none
    soft_beta: float = 0.0  # > 0: silence violations become z penalties instead of rejections
    refine_levels: tuple[int, ...] = (4, 5, 6)  # 100, 50, 25 m
    top_tuples: tuple[int, ...] = (50, 20, 10)
    keep_cells: int = 32
    keep_dz: float = 10.0
    lam: float = LAMBDA_DEFAULT
    p_miss: float = P_MISS_DEFAULT
    calib_hyps: int = 400
    score: ScoreConfig = field(default_factory=ScoreConfig)
    c: float = C_DEFAULT
    n_threads: int | None = None


def exempt_events_for(events: Sequence[Event], mode: str = "usable") -> list[Event]:
    """Detected but unselected events whose times exempt silence-filter passes (plan 3.3.5).

    usable: no ``BAD_FLAGS``; aircraft: not labelled "non-aircraft"; all: every unselected event."""
    un = [e for e in events if not e.selected]
    if mode == "none":
        return []
    if mode == "all":
        return un
    if mode == "aircraft":
        return [e for e in un if "non-aircraft" not in e.quality_flags]
    if mode == "usable":
        return [e for e in un if not set(BAD_FLAGS) & set(e.quality_flags)]
    raise ValueError(f"unknown exempt mode {mode!r}")


def tuple_keys(hyps: pl.DataFrame, K: int) -> pl.Expr:
    """'icao0/seg0|icao1/seg1|...' identifying the flight tuple of a hypothesis."""
    parts = []
    for k in range(K):
        parts += [pl.col(f"icao{k}"), pl.lit("/"), pl.col(f"seg{k}").cast(pl.String)]
        if k < K - 1:
            parts.append(pl.lit("|"))
    return pl.concat_str(parts).alias("tuple")


@dataclass(eq=False)
class SearchResult:
    hypotheses: pl.DataFrame  # every scored hypothesis, all levels (+ lat, lon, tuple, flights)
    tuples: pl.DataFrame  # one row per flight tuple, ranked by best z_sum (its best hypothesis)
    layer: Layer | None
    calibration: Calibration | None
    events: list[Event]  # the selected events, in hypothesis order k
    grid: GridSpec
    stats: dict = field(default_factory=dict)
    cpa: CpaResult | None = None
    score_map: ScoreMap | None = None
    bank: TrackBank | None = None
    origin: tuple[float, float] = (0.0, 0.0)
    ground_hae: GroundFn | None = None
    config: SearchConfig = field(default_factory=SearchConfig)

    @property
    def ok(self) -> bool:
        return self.tuples.height > 0

    def best(self) -> dict[str, Any] | None:
        return self.tuples.row(0, named=True) if self.tuples.height else None

    @property
    def margin_z(self) -> float | None:
        """Best tuple's z_sum minus the runner-up tuple's (z units); None with < 2 tuples."""
        if self.tuples.height < 2:
            return None
        return float(self.tuples["z_sum"][0] - self.tuples["z_sum"][1])

    @property
    def margin_S(self) -> float | None:
        if self.tuples.height < 2:
            return None
        return float(self.tuples["S"][0] - self.tuples["S"][1])

    def top(self, n: int = 10, level: int | None = None, per_tuple: bool = False) -> pl.DataFrame:
        """Top hypotheses by z_sum (optionally at one level, optionally the best per tuple)."""
        h = self.hypotheses if level is None else self.hypotheses.filter(pl.col("level") == level)
        h = h.sort("z_sum", descending=True)
        if per_tuple:
            h = h.unique(subset=["tuple"], keep="first", maintain_order=True)
        return h.head(n)

    def summary(self, top: int = 10) -> dict[str, Any]:
        def iso(t):
            return datetime.fromtimestamp(float(t), UTC).isoformat().replace("+00:00", "Z")

        K = len(self.events)
        rows = []
        for r in self.tuples.head(top).iter_rows(named=True):
            rows.append(
                {
                    "rank": r["rank"],
                    "flights": r["flights"],
                    "tuple": r["tuple"],
                    "z_sum": r["z_sum"],
                    "S": r["S"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "lat_c": r["lat_c"],
                    "lon_c": r["lon_c"],
                    "level": r["level"],
                    "t_s": iso(r["t_s"]),
                    "dt": r["dt"],
                    "n_cells": r["n_cells"],
                    "events": [
                        {
                            "id": self.events[k].id,
                            "z": r.get(f"z_{k}"),
                            "S": r[f"S_{k}"],
                            "f0": r[f"f0_{k}"],
                            "t_h": r[f"th{k}"],
                            "d": r[f"d{k}"],
                            "v": r[f"v{k}"],
                        }
                        for k in range(K)
                    ],
                }
            )
        return {
            "n_tuples": self.tuples.height,
            "tuples": rows,
            "margin_z": self.margin_z,
            "margin_S": self.margin_S,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "events": [e.id for e in self.events],
            "stats": self.stats,
        }

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """scores.parquet (all scored hypotheses), acoustic.json (summary), layers/acoustic.npz."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        out = {"scores": d / "scores.parquet", "summary": d / "acoustic.json"}
        self.hypotheses.write_parquet(out["scores"])
        out["summary"].write_text(json.dumps(self.summary(), indent=1, default=_json_default))
        if self.layer is not None:
            out["layer"] = self.layer.save(d / "layers" / "acoustic.npz")
        return out

    def module_result(self, workdir: str | Path, attribution: str | Sequence[str] = ()) -> ModuleResult:
        wd = Path(workdir)
        paths = self.write(wd)
        layers = []
        if self.layer is not None:
            layers.append(self.layer.to_ref(wd / "layers" / "acoustic.npz", relative_to=wd))
        status = "ok" if self.ok else "degraded"
        msg = "" if self.ok else "no hypothesis survived the gates (flat acoustic layer at log p_miss)"
        attr = [attribution] if isinstance(attribution, str) else list(attribution)
        return ModuleResult(
            module="acoustic",
            status=status,
            message=msg,
            layers=layers,
            provenance={"attribution": attr, "margin_z": self.margin_z, "n_tuples": self.tuples.height},
            artifacts={k: str(p.relative_to(wd)) for k, p in paths.items()},
        )


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --- orchestration -------------------------------------------------------------------------------


def _ground(ground_hae: GroundFn | None, level: int) -> GroundFn:
    return ground_hae if ground_hae is not None else default_ground_hae(level)


def _decorate(scored: pl.DataFrame, grid: GridSpec, K: int) -> pl.DataFrame:
    if scored.height == 0:
        return scored
    lat = np.empty(scored.height)
    lon = np.empty(scored.height)
    lv = scored["level"].to_numpy()
    ix = scored["ix"].to_numpy()
    iy = scored["iy"].to_numpy()
    for L in np.unique(lv):
        m = lv == L
        a, b = cell_centers_latlon(grid, int(L), ix[m], iy[m])
        lat[m], lon[m] = a, b
    out = scored.with_columns(
        tuple_keys(scored, K),
        pl.concat_list([pl.col(f"icao{k}") for k in range(K)]).alias("flights"),
        pl.Series("lat", lat),
        pl.Series("lon", lon),
    )
    if "penalty" in out.columns:
        out = out.with_columns((pl.col("z_sum") - pl.col("penalty")).alias("z_sum"))
    return out


def _rank_tuples(h: pl.DataFrame, lam: float = LAMBDA_DEFAULT) -> pl.DataFrame:
    """One row per tuple: its best hypothesis (all columns), n_hyps, n_cells, max_level, and
    lat_c/lon_c = the exp(lam * z_sum)-weighted centroid of its cells at its finest level (the
    posterior mean of the tuple's own blob, a smoother point estimate than the argmax)."""
    if h.height == 0:
        return pl.DataFrame()
    n = h.group_by("tuple").agg(
        pl.len().alias("n_hyps"),
        pl.struct("level", "ix", "iy").n_unique().alias("n_cells"),
        pl.col("level").max().alias("max_level"),
    )
    fin = (
        h.join(n.select("tuple", "max_level"), on="tuple")
        .filter(pl.col("level") == pl.col("max_level"))
        .with_columns((lam * (pl.col("z_sum") - pl.col("z_sum").max().over("tuple"))).exp().alias("_w"))
        .group_by("tuple")
        .agg(
            ((pl.col("lat") * pl.col("_w")).sum() / pl.col("_w").sum()).alias("lat_c"),
            ((pl.col("lon") * pl.col("_w")).sum() / pl.col("_w").sum()).alias("lon_c"),
        )
    )
    best = h.sort("z_sum", descending=True).unique(subset=["tuple"], keep="first", maintain_order=True)
    best = best.join(n, on="tuple").join(fin, on="tuple").sort("z_sum", descending=True)
    return best.with_row_index("rank").with_columns(pl.col("rank").cast(pl.Int64))


def acoustic_search(
    spec: Spectrogram | ScoreMap,
    events: Sequence[Event],
    tracks: pl.DataFrame,
    grid: GridSpec,
    origin: tuple[float, float],
    t_s: tuple[float, float],
    *,
    silence_tracks: pl.DataFrame | None = None,
    ground_hae: GroundFn | None = None,
    config: SearchConfig | None = None,
    progress: Callable[[str, float, str], None] | None = None,
) -> SearchResult:
    """Run Levels 2-6 for one job.

    ``events``: all detected/edited events; the ones with ``selected`` are matched (order = k), the
    others feed the silence exemptions. ``tracks``: prepared 1-s ENU frame (``adsb.tracks``, jets
    removed) in the frame of ``origin``. ``t_s``: window for the UTC epoch of audio sample 0.
    ``silence_tracks``: flights for the silence filter (default ``tracks``). ``ground_hae``: ground
    ellipsoidal height function (default Copernicus DEM + geoid, per level).
    """
    cfg = config or SearchConfig()
    prog = progress or (lambda s, f, m="": None)
    tim: dict[str, float] = {}
    t0 = time.perf_counter()
    sel = [e for e in events if e.selected]
    if not sel:
        raise ValueError("no selected events")
    K = len(sel)
    exempt = exempt_events_for(events, cfg.exempt)
    if isinstance(spec, ScoreMap):
        smap = spec
        if [e.id for e in smap.events] != [e.id for e in sel]:
            raise ValueError("score map events differ from the selected events")
    else:
        smap = ScoreMap.from_spectrogram(spec, sel, cfg.score)
    tim["score_map"] = time.perf_counter() - t0
    stats: dict[str, Any] = {
        "events": [e.id for e in sel],
        "exempt_events": [e.id for e in exempt],
        "silence_scope": cfg.silence_scope,
    }
    prog("score", 0.02, "candidate cells and CPA")
    t1 = time.perf_counter()
    gates = [EventGate.from_event(e, t_s, cfg.slack_s, cfg.v_margin) for e in sel]
    cpa = cpa_search(
        tracks, origin, grid, gates, ground_hae, cfg.h_obs, SEARCH_LEVEL, c=cfg.c, n_threads=cfg.n_threads
    )
    tim["cpa"] = time.perf_counter() - t1
    taus = taus_from_events(sel)
    t1 = time.perf_counter()
    hyps = match_offsets(cpa.table, taus, cfg.slack_s, t_s)
    stats.update(
        cpa_rows=cpa.table.height,
        cells_candidates=cpa.stats.get("cells_candidates"),
        hyps_matched=hyps.height,
        cells_matched=hyps["cell"].n_unique() if hyps.height else 0,
    )
    sil_tracks = silence_tracks if silence_tracks is not None else tracks
    sil_kw = dict(
        kappa=cfg.kappa,
        guard=cfg.silence_guard,
        scope=cfg.silence_scope,
        duration=smap.duration,
        exempt_events=exempt,
        soft_beta=cfg.soft_beta,
        c=cfg.c,
        n_threads=cfg.n_threads,
    )
    if hyps.height:
        sr = silence_filter(hyps, sil_tracks, cpa.observers, taus, **sil_kw)  # type: ignore[arg-type]
        hyps = sr.hyps
        stats["silence"] = {k: v for k, v in sr.stats.items() if k != "spans"}
    tim["match_silence"] = time.perf_counter() - t1
    stats.update(
        hyps_after_silence=hyps.height,
        cells_after_silence=hyps["cell"].n_unique() if hyps.height else 0,
        tuples_after_silence=hyps.select([f"icao{k}" for k in range(K)] + [f"seg{k}" for k in range(K)])
        .unique()
        .height
        if hyps.height
        else 0,
    )
    empty = SearchResult(
        hypotheses=pl.DataFrame(),
        tuples=pl.DataFrame(),
        layer=None,
        calibration=None,
        events=sel,
        grid=grid,
        stats=stats,
        cpa=cpa,
        score_map=smap,
        origin=origin,
        ground_hae=ground_hae,
        config=cfg,
    )
    if hyps.height == 0:
        empty.layer = Layer(spec=grid, name="acoustic", module="acoustic", fill=math.log(cfg.p_miss))
        stats["timing_s"] = tim
        return empty
    prog("score", 0.3, f"calibrating on {min(cfg.calib_hyps, hyps.height)} hypotheses")
    t1 = time.perf_counter()
    bank = TrackBank.from_tracks(tracks)
    calib = calibrate(
        smap, bank, hyps, cpa.observers, lam=cfg.lam, p_miss=cfg.p_miss, max_hyps=cfg.calib_hyps
    )
    tim["calibrate"] = time.perf_counter() - t1
    prog("score", 0.4, f"scoring {hyps.height} hypotheses at {grid.cell_size(SEARCH_LEVEL)} m")
    t1 = time.perf_counter()
    scored = [_decorate(score_hypotheses(smap, bank, hyps, cpa.observers, calib), grid, K)]
    tim[f"score_L{SEARCH_LEVEL}"] = time.perf_counter() - t1
    refined_at: dict[str, int] = {}  # tuple -> finest level reached
    for t in scored[0]["tuple"].unique().to_list():
        refined_at[t] = SEARCH_LEVEL
    segs_all = track_segments(tracks)
    prev_level = SEARCH_LEVEL
    level_tuples: dict[int, list[str]] = {SEARCH_LEVEL: sorted(refined_at)}
    for i, (L, M) in enumerate(zip(cfg.refine_levels, cfg.top_tuples, strict=True)):
        t1 = time.perf_counter()
        allh = pl.concat(scored, how="diagonal_relaxed")
        ranked = _rank_tuples(allh, cfg.lam)
        prevh = allh.filter(pl.col("level") == prev_level)
        cand = [t for t in ranked["tuple"].to_list() if t in set(prevh["tuple"].to_list())][:M]
        if not cand:
            break
        keep = (
            prevh.filter(pl.col("tuple").is_in(cand))
            .sort("z_sum", descending=True)
            .with_columns(pl.col("z_sum").max().over("tuple").alias("_best"))
            .filter(pl.col("z_sum") >= pl.col("_best") - cfg.keep_dz)
            .group_by("tuple", maintain_order=True)
            .head(cfg.keep_cells)
        )
        fx, fy = densify(prev_level, keep["ix"].to_numpy(), keep["iy"].to_numpy(), L, halo=1)
        uk = np.unique(pack_cells(L, fx, fy))
        _, fx, fy = unpack_cells(uk)
        obs = observer_cells(CellSet(grid, L, fx, fy), origin, _ground(ground_hae, L), cfg.h_obs)
        want = {(r[f"icao{k}"], int(r[f"seg{k}"])) for r in keep.iter_rows(named=True) for k in range(K)}
        segs = [sg for sg in segs_all if (sg.icao, int(sg.seg)) in want]
        tab = compute_cpa(segs, obs, gates, cfg.c, cfg.n_threads)
        hf = match_offsets(tab, taus, cfg.slack_s, t_s)
        if hf.height:
            hf = hf.with_columns(tuple_keys(hf, K)).filter(pl.col("tuple").is_in(cand)).drop("tuple")
        n_match = hf.height
        if hf.height:
            sr = silence_filter(hf, sil_tracks, obs, taus, **sil_kw)  # type: ignore[arg-type]
            hf = sr.hyps
        prog(
            "score",
            0.4 + 0.5 * (i + 1) / len(cfg.refine_levels),
            f"refining {len(cand)} tuples at {grid.cell_size(L)} m ({hf.height} hypotheses)",
        )
        stats[f"refine_L{L}"] = {
            "tuples": len(cand),
            "cells": int(uk.size),
            "hyps_matched": n_match,
            "hyps_after_silence": hf.height,
        }
        if hf.height == 0:
            tim[f"refine_L{L}"] = time.perf_counter() - t1
            break
        sc = _decorate(score_hypotheses(smap, bank, hf, obs, calib), grid, K)
        scored.append(sc)
        level_tuples[L] = sorted(set(sc["tuple"].to_list()))
        for t in level_tuples[L]:
            refined_at[t] = L
        prev_level = L
        tim[f"refine_L{L}"] = time.perf_counter() - t1
    allh = pl.concat(scored, how="diagonal_relaxed")
    ranked = _rank_tuples(allh, cfg.lam)
    t1 = time.perf_counter()
    layer = build_layer(grid, allh, level_tuples, cfg.lam, cfg.p_miss, K)
    tim["layer"] = time.perf_counter() - t1
    tim["total"] = time.perf_counter() - t0
    stats["timing_s"] = tim
    prog("score", 1.0, f"{ranked.height} flight tuples scored")
    return SearchResult(
        hypotheses=allh,
        tuples=ranked,
        layer=layer,
        calibration=calib,
        events=sel,
        grid=grid,
        stats=stats,
        cpa=cpa,
        score_map=smap,
        bank=bank,
        origin=origin,
        ground_hae=ground_hae,
        config=cfg,
    )


# --- fusion layer --------------------------------------------------------------------------------


def _cell_tuple_best(h: pl.DataFrame) -> pl.DataFrame:
    return h.sort("z_sum", descending=True).unique(
        subset=["level", "ix", "iy", "tuple"], keep="first", maintain_order=True
    )


def build_layer(
    grid: GridSpec,
    h: pl.DataFrame,
    level_tuples: dict[int, list[str]],
    lam: float,
    p_miss: float,
    K: int,
    name: str = "acoustic",
) -> Layer:
    """Acoustic fusion layer from scored hypotheses at several levels (see module docstring)."""
    levels = sorted(level_tuples)
    best = _cell_tuple_best(h)
    acols = (
        ["t_s", "dt", "tuple", "flights"]
        + [f"f0_{k}" for k in range(K)]
        + [f"z_{k}" for k in range(K) if f"z_{k}" in best.columns]
    )
    L0 = levels[0]
    b0 = best.filter(pl.col("level") == L0)
    lay = acoustic_layer(
        grid,
        L0,
        b0["ix"].to_numpy(),
        b0["iy"].to_numpy(),
        b0["z_sum"].to_numpy(),
        lam=lam,
        p_miss=p_miss,
        name=name,
        attrs={c: b0[c].to_list() if c == "flights" else b0[c].to_numpy() for c in acols},
    )
    finest: dict[str, int] = {}
    for L in levels:
        for t in level_tuples[L]:
            finest[t] = L
    for L in levels[1:]:
        bL = best.filter(pl.col("level") == L).select("ix", "iy", "z_sum", *acols)
        if bL.height == 0:
            continue
        cells = bL.select("ix", "iy").unique()
        extra = []
        here = set(level_tuples[L])
        for Lt in levels:
            if Lt >= L:
                break
            ts = [t for t, f in finest.items() if f == Lt and t not in here]
            if not ts:
                continue
            sh = L - Lt
            anc = cells.with_columns(
                (pl.col("ix") // (1 << sh)).alias("ax"), (pl.col("iy") // (1 << sh)).alias("ay")
            )
            src = best.filter((pl.col("level") == Lt) & pl.col("tuple").is_in(ts)).select(
                pl.col("ix").alias("ax"), pl.col("iy").alias("ay"), "z_sum", *acols
            )
            j = anc.join(src, on=["ax", "ay"]).drop("ax", "ay")
            if j.height:
                extra.append(j)
        allL = pl.concat([bL, *extra], how="diagonal_relaxed") if extra else bL
        lay.add_cells(
            L,
            allL["ix"].to_numpy(),
            allL["iy"].to_numpy(),
            lam * allL["z_sum"].to_numpy(),
            reduce="logsumexp",
        )
        at = (
            bL.sort("z_sum", descending=True)
            .unique(subset=["ix", "iy"], keep="first", maintain_order=True)
            .with_columns(pl.lit(L, pl.Int64).alias("level"))
        )
        lay.attrs = (
            pl.concat([lay.attrs, at.select(lay.attrs.columns)], how="diagonal_relaxed")
            if lay.attrs is not None
            else at
        )
    lay.meta.update({"levels": levels, "n_tuples": len(finest)})
    return lay


# --- per-tuple band maps (evidence view) --------------------------------------------------------


def band_maps(
    res: SearchResult, tuple_rank: int = 0, radius_m: float = 3000.0, level: int = 4, window_s: float = 600.0
) -> pl.DataFrame:
    """Single-event score maps of one flight tuple around its best point [N 2 22:23].

    Each event k of the tuple is scored *alone*: at every cell of ``level`` within ``radius_m`` of the
    tuple's best point, the cell's own CPA of flight k (event k's speed/distance gate, heard within
    ``window_s`` of the tuple's best t_s + tau_k) gives that cell's implied start o_k = t_h - tau_k;
    the curve is then scored with dt +-5 s and f0 free. Without the second event the start time is
    free, so each z_k map is an elongated band along the track; z_sum (the sum of the bands where every
    event has a pass, else NaN) peaks where they cross. Returns level, ix, iy, lat, lon, z_0.., o_0.., z_sum.
    """
    if res.calibration is None or res.score_map is None or res.bank is None or not res.ok:
        raise ValueError("search result has no scored tuples")
    row = res.tuples.row(tuple_rank, named=True)
    grid = res.grid
    s = grid.cell_size(level)
    cx, cy = (int(np.ravel(a)[0]) for a in latlon_to_cells(grid, row["lat"], row["lon"], level))
    r = int(math.ceil(radius_m / s))
    gx, gy = np.meshgrid(np.arange(cx - r, cx + r + 1), np.arange(cy - r, cy + r + 1))
    m = (gx - cx) ** 2 + (gy - cy) ** 2 <= r * r
    cells = CellSet(grid, level, gx[m], gy[m])
    obs = observer_cells(cells, res.origin, _ground(res.ground_hae, level), res.config.h_obs)
    n = len(cells)
    keys = cells.keys
    smap, bank, cal = res.score_map, res.bank, res.calibration
    cfg = smap.cfg
    out: dict[str, Any] = {"level": np.full(n, level, np.int64), "ix": cells.ix, "iy": cells.iy}
    out["lat"], out["lon"] = cells.latlon()
    zsum = np.zeros(n)
    for k, ev in enumerate(res.events):
        sk = bank.index[(row[f"icao{k}"], int(row[f"seg{k}"]))]
        seg = [sg for sg in track_segments([bank.track(sk)])]
        center = float(row["t_s"]) + ev.t_h
        g = EventGate.from_event(ev, v_margin=res.config.v_margin)
        g = EventGate(
            id=g.id,
            v_lo=g.v_lo,
            v_hi=g.v_hi,
            d_lo=g.d_lo,
            d_hi=g.d_hi,
            th_lo=center - window_s,
            th_hi=center + window_s,
        )
        tab = compute_cpa(seg, obs, [g], res.config.c)
        o = np.full(n, np.nan)
        if tab.height:
            best = (
                tab.with_columns((pl.col("t_h") - ev.t_h).alias("o"))
                .with_columns((pl.col("o") - float(row["t_s"])).abs().alias("_dist"))
                .sort("_dist")
                .unique(subset=["cell"], keep="first")
            )
            pos = np.searchsorted(np.sort(keys), best["cell"].to_numpy())
            order = np.argsort(keys)
            o[order[pos]] = best["o"].to_numpy()
        z = np.full(n, np.nan)
        have = np.flatnonzero(np.isfinite(o))
        if have.size:
            ps = PairSet(
                seg=np.full(have.size, sk, np.int64),
                obs=np.ascontiguousarray(obs.xyz[have]),
                o=o[have],
                ev=np.full(have.size, k, np.int64),
            )
            S, _, _ = score_pairs(smap, bank, ps, cfg.n_coarse, cfg.coarse_step)
            with np.errstate(invalid="ignore"):
                bS = np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1)
            z[have] = np.where(np.isfinite(bS), cal.z(k, bS), np.nan)
        out[f"z_{k}"] = z
        out[f"o_{k}"] = o
        zsum = zsum + z
    out["z_sum"] = zsum
    return pl.DataFrame(out)
