"""WP22 synthetic end-to-end runner, lambda calibration and acceptance metrics (plan 5.2, 3.3.7).

Module-API path (no job pipeline): `synthetic_world.build_world` -> `InMemoryTrackProvider.fetch`
-> `adsb.tracks.prepare_trackset` -> `spectrogram.compute_spectrogram` -> `detect.find_events`
(unaided: its default top-2 selection) -> `search.acoustic_search` (whole-day T_s window, analytic
DEM) -> `fusion.export.run_fusion` -> candidates / HPD.

Calibration of lambda (ell_A = lam * sum_k z_k). The scored hypotheses do not depend on lambda (it
only scales the fusion layer), so each seed is searched once, its hypotheses are cached
(`save_dir`), and the layer is rebuilt and fused for every lambda on a grid. Per seed the
statistic is the HPD level of the truth, h_s(lam) = posterior mass of all leaves at least as dense
as the truth's leaf; the truth is inside the 90 % HPD region iff h_s <= 0.9. lambda* is the
largest grid value whose coverage mean_s[h_s <= 0.9] is >= the target (0.9): the sharpest
posterior that is still calibrated.

Parallelism: seeds run in a process pool (default 4 workers, 1 numba / FFT thread each).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "LAMBDA_GRID", "RunOptions", "run_seed", "run_seeds", "evaluate_lambda", "fit_lambda",
    "coverage_curve", "summarize", "run_calibration", "write_json",
]

LAMBDA_GRID: tuple[float, ...] = tuple(float(x) for x in np.round(np.geomspace(0.02, 10.0, 18), 5))


# The silence decoy "blob": hypotheses of the decoy tuple this close to X_d (farther out the silence
# flight is legitimately beyond max_k d_k and the filter must not fire).
DECOY_RADIUS_M = 400.0

# Single-event known-time worlds: one heard flight, no silence decoy (its exact copy would be a twin).
SINGLE_WORLD: dict = {"n_heard": (1,), "silence_decoy": False}


@dataclass(frozen=True)
class RunOptions:
    mp3_kbps: int | None = None
    single_event: bool = False  # known-time single-event mode (strongest selected event only)
    known_sigma_s: float = 60.0  # single-event: T_s known to +- this
    lams: tuple[float, ...] = LAMBDA_GRID
    threads: int = 1
    save_dir: str | None = None  # cache scored hypotheses per seed (parquet)
    workdir: str | None = None  # scratch for the module_result files (default data/tmp, removed)
    world: dict = field(default_factory=dict)  # WorldConfig overrides


def _set_threads(n: int) -> None:
    try:
        import numba

        numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))
    except Exception:  # pragma: no cover
        pass


def _level_tuples(h: pl.DataFrame) -> dict[int, list[str]]:
    return {int(L): sorted(set(g["tuple"].to_list()))
            for (L,), g in h.group_by("level", maintain_order=True)}


def _fusion_metrics(grid, bbox, h: pl.DataFrame, K: int, lam: float, truth, p_miss: float,
                    hpd: bool = False) -> dict[str, Any]:
    """Fuse the acoustic layer at one lambda; HPD level of the truth and candidate distances."""
    from ..core.geo import haversine_m
    from ..fusion.candidates import compute_hpd, credible_level, find_candidates
    from ..fusion.combine import fuse
    from .search import build_layer

    if h.height == 0:
        return {"lam": lam, "hpd_level": 1.0, "top1_m": None, "top3_m": None}
    lay = build_layer(grid, h, _level_tuples(h), lam, p_miss, K)
    post = fuse(grid, [lay], domain=bbox)
    v = post.value_at_latlon(np.array([truth[0]]), np.array([truth[1]]))
    lvl = float(credible_level(post, v)[0])
    cands = find_candidates(post, k=3)
    d = [float(haversine_m(c.lat, c.lon, *truth)) for c in cands]
    out: dict[str, Any] = {"lam": lam, "hpd_level": lvl, "top1_m": d[0] if d else None,
                           "top3_m": min(d) if d else None,
                           "top1_mass": cands[0].prob_mass if cands else None}
    if hpd:
        regs = compute_hpd(post, (0.5, 0.9))
        out["hpd_area_m2"] = {str(r.mass): r.area_m2 for r in regs}
        g = regs[1].geometry_xy
        if g is not None and not g.is_empty:
            rect = g.minimum_rotated_rectangle
            xs, ys = rect.exterior.coords.xy
            sides = sorted(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2))
            out["hpd90_extent_m"] = [sides[0], sides[1]]
    return out


def run_seed(seed: int, opts: RunOptions | None = None) -> dict[str, Any]:
    """One synthetic world through the whole acoustic path; JSON-able metrics (no lambda fit)."""
    opts = opts or RunOptions()
    _set_threads(opts.threads)
    from ..adsb.aircraft_types import load_types
    from ..adsb.tracks import prepare_trackset
    from ..core.geo import bbox_expand_m, haversine_m
    from ..core.types import TrackQuery
    from ..fusion.grid import GridSpec, cell_centers_latlon
    from .detect import DetectConfig, find_events
    from .matching import match_offsets, taus_from_events
    from .search import SearchConfig, acoustic_search
    from .spectrogram import compute_spectrogram
    from .synthetic_world import WorldConfig, build_world

    tim: dict[str, float] = {}
    t = time.perf_counter()
    wcfg = replace(WorldConfig(), **{**opts.world, "mp3_kbps": opts.mp3_kbps})
    w = build_world(seed, wcfg)
    tim["world"] = time.perf_counter() - t
    t = time.perf_counter()
    spec = compute_spectrogram(w.audio)
    tim["spectrogram"] = time.perf_counter() - t
    t = time.perf_counter()
    det = find_events(spec, DetectConfig(n_threads=opts.threads))
    tim["detect"] = time.perf_counter() - t
    events = list(det.events)
    heard = w.heard()

    def heard_of(ev) -> str | None:
        best = min(heard, key=lambda f: abs(f.pass_x.t_h - w.t_s - ev.t_h))
        return best.icao if abs(best.pass_x.t_h - w.t_s - ev.t_h) <= 10.0 else None

    sel = [e for e in events if e.selected]
    cfg = SearchConfig(n_threads=opts.threads)
    t_win = w.day_window
    if opts.single_event and sel:
        keep = max(sel, key=lambda e: e.snr)
        events = [e.model_copy(update={"selected": e.id == keep.id}) for e in events]
        sel = [keep]
        t_win = (w.t_s - opts.known_sigma_s, w.t_s + opts.known_sigma_s)
        cfg = replace(cfg, silence_scope="full")
    t = time.perf_counter()
    d0, d1 = w.day_window
    q = TrackQuery(bbox=bbox_expand_m(w.bbox, 15_000), t0=datetime.fromtimestamp(d0, UTC),
                   t1=datetime.fromtimestamp(d1 + wcfg.duration_s, UTC))
    ts = w.provider.fetch(q)
    grid = GridSpec.for_bbox(w.bbox)
    origin = (grid.lat0, grid.lon0)
    prep, cls = prepare_trackset(ts, origin, load_types())
    tim["tracks"] = time.perf_counter() - t
    rec: dict[str, Any] = {
        "seed": seed, "mp3_kbps": opts.mp3_kbps, "single_event": opts.single_event,
        "truth": list(w.truth_latlon), "t_s": w.t_s, "n_heard": len(heard),
        "heard": [{"icao": f.icao, "tau": f.pass_x.t_h - w.t_s, "d": f.pass_x.d, "v": f.pass_x.v,
                   "f0": f.f0, "snr_db": f.snr_db} for f in heard],
        "n_flights": ts.aircraft.height, "n_kept": int(cls["keep"].sum()),
        "events": [{"id": e.id, "t_h": e.t_h, "v": e.v, "d": e.d, "f0": e.f0, "snr": e.snr,
                    "v_range": [e.v_range.lo, e.v_range.hi], "d_range": [e.d_range.lo, e.d_range.hi],
                    "selected": e.selected, "flags": e.quality_flags, "heard": heard_of(e),
                    "truth_in_ranges": _in_ranges(e, w.by_icao(heard_of(e)), w.t_s) if heard_of(e) else None}
                   for e in events],
        "selected_heard": [heard_of(e) for e in sel],
    }
    if not sel:
        rec.update(ok=False, timing_s=tim, reason="no event selected")
        return rec
    t = time.perf_counter()
    res = acoustic_search(spec, events, prep, grid, origin, t_win, ground_hae=w.ground_hae, config=cfg)
    tim["search"] = time.perf_counter() - t
    tim.update({f"search.{k}": v for k, v in res.stats.get("timing_s", {}).items()})
    rec["stats"] = {k: v for k, v in res.stats.items() if k not in ("timing_s",)}
    rec["ok"] = res.ok
    K = len(sel)
    # silence decoy: copies of the selected events' flights at X_d, matched before / after silence
    decoy_of = {f.copy_of: f.icao for f in w.flights if f.role == "decoy"}
    want = [decoy_of.get(i) for i in rec["selected_heard"]]
    if None not in want and w.decoy_latlon is not None:
        hm = match_offsets(res.cpa.table, taus_from_events(sel), cfg.slack_s, t_win)

        def at_decoy(h: pl.DataFrame) -> int:
            """Hypotheses of the decoy tuple within DECOY_RADIUS_M of X_d at the true start (+- 10 s)."""
            cond = (pl.col("o") - w.t_s).abs() <= 10.0
            for k, icao in enumerate(want):
                cond = cond & (pl.col(f"icao{k}") == icao)
            h = h.filter(cond)
            n = 0
            for (L,), g in h.group_by("level"):
                la, lo = cell_centers_latlon(grid, int(L), g["ix"].to_numpy(), g["iy"].to_numpy())
                n += int((haversine_m(la, lo, *w.decoy_latlon) <= DECOY_RADIUS_M).sum())
            return n

        n_before = at_decoy(hm)
        n_after = at_decoy(res.hypotheses) if res.ok else 0
        rec["decoy"] = {"applicable": True, "matched": n_before, "survived": n_after,
                        "removed": n_before > 0 and n_after == 0}
    else:
        rec["decoy"] = {"applicable": False}
    if res.ok:
        b = res.best()
        rec["best"] = {
            "flights": b["flights"], "z_sum": b["z_sum"], "level": b["level"],
            "dist_m": float(haversine_m(b["lat"], b["lon"], *w.truth_latlon)),
            "centroid_dist_m": float(haversine_m(b["lat_c"], b["lon_c"], *w.truth_latlon)),
            "t_s_err": b["t_s"] - w.t_s, "correct_tuple": list(b["flights"]) == rec["selected_heard"],
        }
        rec["margin_z"] = res.margin_z
        rec["calibration"] = [asdict(c) for c in res.calibration.events]
        h = res.hypotheses
        if opts.single_event:
            rec["band"] = _band_metrics(grid, h, rec["selected_heard"][0], w.truth_latlon)
        if opts.save_dir:
            d = Path(opts.save_dir)
            d.mkdir(parents=True, exist_ok=True)
            h.write_parquet(d / f"seed_{seed:03d}.parquet")
        t = time.perf_counter()
        rec["run_fusion"] = _run_fusion_check(res, grid, w.bbox, w.truth_latlon, opts.workdir, seed)
        tim["run_fusion"] = time.perf_counter() - t
        t = time.perf_counter()
        rec["by_lam"] = [_fusion_metrics(grid, w.bbox, h, K, lam, w.truth_latlon, cfg.p_miss,
                                         hpd=opts.single_event) for lam in opts.lams]
        tim["fusion_sweep"] = time.perf_counter() - t
        rec["n_lams"] = len(opts.lams)
    rec["timing_s"] = tim
    return rec


def _run_fusion_check(res, grid, bbox, truth, workdir: str | None, seed: int) -> dict[str, Any]:
    """The pipeline hand-over at the search lambda: SearchResult.module_result (files) ->
    fusion.export.run_fusion(module_results=...) -> candidates + HPD regions."""
    import tempfile

    from ..core.geo import haversine_m
    from ..fusion.candidates import credible_level
    from ..fusion.export import run_fusion

    root = Path(workdir) if workdir else Path(__file__).resolve().parents[3] / "data" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"wp22_{seed}_", dir=root) as td:
        mr = res.module_result(td, attribution="synthetic tracks")
        fr = run_fusion(grid, module_results=[mr], root=td, domain=bbox)
    post = fr.posterior
    lvl = float(credible_level(post, post.value_at_latlon(np.array([truth[0]]), np.array([truth[1]])))[0])
    c = fr.candidates
    return {"lam": res.config.lam, "status": mr.status, "n_candidates": len(c),
            "top1_m": float(haversine_m(c[0].lat, c[0].lon, *truth)) if c else None,
            "top1_flights": c[0].flights if c else None,
            "hpd_level": lvl, "hpd_area_m2": {str(r.mass): r.area_m2 for r in fr.hpd},
            "attribution": fr.attribution, "notes": fr.notes}


def _band_metrics(grid, h: pl.DataFrame, icao: str | None, truth) -> dict[str, Any]:
    """Single-event band: the 200-m hypothesis cells of the heard flight (its support in the acoustic
    layer). contains_truth: the truth's 200-m cell is one of them. Length/width: 2 sqrt(eig) of the
    cell-centre covariance (x 2 = +-1 sigma extent), so a band has length >> width."""
    from ..fusion.grid import cell_centers_xy, latlon_to_cells
    from .grid import SEARCH_LEVEL

    if icao is None or h.height == 0:
        return {"contains_truth": False, "n_cells": 0}
    b = h.filter((pl.col("icao0") == icao) & (pl.col("level") == SEARCH_LEVEL)).unique(subset=["ix", "iy"])
    tx, ty = (int(np.ravel(a)[0]) for a in latlon_to_cells(grid, truth[0], truth[1], SEARCH_LEVEL))
    inside = b.filter((pl.col("ix") == tx) & (pl.col("iy") == ty)).height > 0
    out: dict[str, Any] = {"contains_truth": bool(inside), "n_cells": b.height}
    if b.height >= 3:
        x, y = cell_centers_xy(grid, SEARCH_LEVEL, b["ix"].to_numpy(), b["iy"].to_numpy())
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([x, y]))))
        out["length_m"] = float(4 * math.sqrt(max(ev[1], 0.0)))
        out["width_m"] = float(4 * math.sqrt(max(ev[0], 0.0)))
        out["aspect"] = out["length_m"] / max(out["width_m"], 1.0)
    return out


def _in_ranges(ev, f, t_s: float) -> bool:
    """Truth (heard CPA of flight f) inside the event's v, d, t_h and f0 ranges."""
    p = f.pass_x
    return bool(ev.v_range.contains(p.v) and ev.d_range.contains(p.d)
                and ev.t_h_range.contains(p.t_h - t_s) and ev.f0_range.contains(f.f0))


def _worker(args):
    seed, opts = args
    try:
        return run_seed(seed, opts)
    except Exception as e:  # keep the batch going; the failure is a reported seed
        import traceback

        return {"seed": seed, "ok": False, "error": repr(e), "trace": traceback.format_exc()}


def run_seeds(seeds: Sequence[int], opts: RunOptions | None = None, workers: int = 4,
              log=None) -> list[dict[str, Any]]:
    """Run seeds in a process pool (deterministic per seed; order of the output = order of seeds)."""
    opts = opts or RunOptions()
    out: dict[int, dict] = {}
    os.environ.setdefault("OMP_NUM_THREADS", str(opts.threads))
    if workers <= 1:
        for s in seeds:
            out[s] = _worker((s, opts))
            if log:
                log(out[s])
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for r in ex.map(_worker, [(s, opts) for s in seeds]):
                out[r["seed"]] = r
                if log:
                    log(r)
    return [out[s] for s in seeds]


def coverage_curve(recs: Sequence[dict], mass: float = 0.9) -> list[tuple[float, float]]:
    """(lambda, fraction of seeds whose truth lies inside the `mass` HPD region). A seed without
    hypotheses (or that failed) counts as not covered."""
    lams = next((tuple(x["lam"] for x in r["by_lam"]) for r in recs if r.get("by_lam")), ())
    out = []
    for i, lam in enumerate(lams):
        cov = [bool(r.get("by_lam")) and r["by_lam"][i]["hpd_level"] <= mass for r in recs]
        out.append((lam, float(np.mean(cov)) if cov else float("nan")))
    return out


def fit_lambda(recs: Sequence[dict], target: float = 0.9, mass: float = 0.9) -> tuple[float, float]:
    """Largest grid lambda with coverage >= target (else the largest lambda of maximum coverage)."""
    curve = coverage_curve(recs, mass)
    ok = [(lam, c) for lam, c in curve if c >= target]
    if ok:
        return max(ok)
    return max(curve, key=lambda lc: (lc[1], lc[0]))


def evaluate_lambda(recs: Sequence[dict], lam: float) -> list[dict]:
    """Per-seed fusion metrics at the grid lambda closest to `lam`."""
    out = []
    for r in recs:
        if not r.get("by_lam"):
            out.append({"seed": r["seed"], "hpd_level": 1.0, "top1_m": None, "top3_m": None})
            continue
        m = min(r["by_lam"], key=lambda x: abs(math.log(x["lam"] / lam)))
        out.append({"seed": r["seed"], **m})
    return out


def _frac(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def summarize(recs: Sequence[dict], lam: float) -> dict[str, Any]:
    """Acceptance numbers for one variant at lambda."""
    ev = evaluate_lambda(recs, lam)
    dec = [r["decoy"] for r in recs if r.get("decoy", {}).get("applicable")]
    tim = [r["timing_s"] for r in recs if "timing_s" in r]
    keys = sorted({k for t in tim for k in t})
    return {
        "n_seeds": len(recs),
        "lam": lam,
        "top1_within_100m": _frac((e["top1_m"] is not None and e["top1_m"] <= 100.0) for e in ev),
        "top3_within_250m": _frac((e["top3_m"] is not None and e["top3_m"] <= 250.0) for e in ev),
        "hpd90_coverage": _frac(e["hpd_level"] <= 0.9 for e in ev),
        "hpd50_coverage": _frac(e["hpd_level"] <= 0.5 for e in ev),
        "acoustic_best_within_100m": _frac(bool(r.get("best")) and r["best"]["dist_m"] <= 100.0
                                           for r in recs),
        "correct_tuple": _frac(bool(r.get("best")) and r["best"]["correct_tuple"] for r in recs),
        "selected_both_heard": _frac(None not in r.get("selected_heard", [None]) for r in recs),
        "decoy_applicable": len(dec),
        "decoy_removed": _frac(d["removed"] for d in dec),
        "decoy_matched": _frac(d["matched"] > 0 for d in dec),
        "median_top1_m": float(np.median([e["top1_m"] for e in ev if e["top1_m"] is not None]))
        if any(e["top1_m"] is not None for e in ev) else None,
        "selected_truth_in_ranges": _frac(e["truth_in_ranges"] for r in recs for e in r.get("events", [])
                                          if e["selected"] and e["truth_in_ranges"] is not None),
        "failed_seeds": [r["seed"] for r in recs if not r.get("ok")],
        "miss_seeds_100m": [e["seed"] for e in ev if not (e["top1_m"] is not None and e["top1_m"] <= 100)],
        "band": [{"seed": r["seed"], **r["band"], "hpd_level": e["hpd_level"], "top1_m": e["top1_m"],
                  "hpd_area_m2": e.get("hpd_area_m2"), "hpd90_extent_m": e.get("hpd90_extent_m")}
                 for r, e in zip(recs, ev, strict=True) if "band" in r],
        "timing_median_s": {k: float(np.median([t[k] for t in tim if k in t])) for k in keys},
        "timing_max_s": {k: float(np.max([t[k] for t in tim if k in t])) for k in keys},
    }


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1,
                            default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    return p


def run_calibration(n_seeds: int = 40, *, mp3_seeds: int = 40, single_seeds: int = 5, workers: int = 4,
                    out_dir: str | Path | None = None, target: float = 0.9, log=None) -> dict[str, Any]:
    """The WP22 run: `n_seeds` clean worlds (lambda fit + acceptance), `mp3_seeds` MP3-32k worlds
    and `single_seeds` single-event known-time worlds, all evaluated at the fitted lambda."""
    t0 = time.perf_counter()
    d = Path(out_dir) if out_dir else None
    sv = (lambda v: str(d / "hyps" / v)) if d else (lambda v: None)
    clean = run_seeds(range(n_seeds), RunOptions(save_dir=sv("clean")), workers, log)
    lam, cov = fit_lambda(clean, target)
    out: dict[str, Any] = {
        "lam": lam, "coverage_at_lam": cov, "target": target, "lams": list(LAMBDA_GRID),
        "coverage_curve": coverage_curve(clean), "clean": summarize(clean, lam),
        "clean_at_default_0.5": summarize(clean, 0.5),
    }
    runs: dict[str, list] = {"clean": clean}
    if mp3_seeds:
        mp3 = run_seeds(range(mp3_seeds), RunOptions(mp3_kbps=32, save_dir=sv("mp3_32k")), workers, log)
        runs["mp3_32k"] = mp3
        out["mp3_32k"] = summarize(mp3, lam)
        out["mp3_32k_coverage_curve"] = coverage_curve(mp3)
    if single_seeds:
        sopt = RunOptions(single_event=True, lams=(lam, 0.5), world=SINGLE_WORLD)
        single = run_seeds(range(single_seeds), sopt, workers, log)
        runs["single_event"] = single
        out["single_event"] = summarize(single, lam)
    out["wall_s"] = time.perf_counter() - t0
    if d:
        write_json(out, d / "lambda.json")
        write_json(runs, d / "runs.json")
    return out
