"""WP11 benchmark: locked scoring of N hypotheses x 2 events (plan 3.3.8 target: 50k x 2 in <= 60 s).

Synthetic inputs only (no network, no audio file): a 1-h random whitened map, 200 straight GA tracks,
hypotheses at random observers within 6 km of their tracks. Times the calibration and the coarse +
fine locked sweep (`score_hypotheses`).

    nix develop --command python scripts/bench_score.py [N=50000]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from triangulate.acoustic.calibrate import calibrate
from triangulate.acoustic.score import ScoreConfig, ScoreMap, TrackBank, score_hypotheses
from triangulate.acoustic.synth import straight_pass_track
from triangulate.core.types import Event, Range


def build(n_hyps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = ScoreConfig()
    U = int(np.log(1000 / 20) / cfg.du)
    T = int(3600 / cfg.frame_dt)
    Zl = np.clip(rng.gamma(1.0, 0.6, size=(T, U)), 0, 12).astype(np.float32)
    t0 = 1.789e9
    taus = (1200.0, 2100.0)
    evs = [
        Event(
            id=f"E{k}",
            t_h=tau,
            t_h_range=Range(lo=tau - 3, hi=tau + 3),
            f0=f0,
            f0_range=Range(lo=f0 * 0.97, hi=f0 * 1.03),
            v=v,
            v_range=Range(lo=v * 0.9, hi=v * 1.1),
            d=d,
            d_range=Range(lo=d * 0.7, hi=d * 1.3),
            harmonics_seen=[1, 2],
        )
        for k, (tau, f0, v, d) in enumerate(
            [(taus[0], 115.0, 135.0, 5800.0), (taus[1], 130.0, 105.0, 3200.0)]
        )
    ]
    smap = ScoreMap.from_log_map(Zl, 0.3125, cfg.frame_dt, float(np.log(20.0)), cfg.du, evs, cfg, 3600.0)
    frames, ntr = [], 200
    for i in range(ntr):
        p = rng.uniform(-30000, 30000, 3) * np.array([1, 1, 0])
        tr = straight_pass_track(
            p,
            t0 + rng.uniform(600, 3000),
            rng.uniform(50, 150),
            rng.uniform(1000, 6000),
            heading_deg=rng.uniform(0, 360),
            half_span_s=2400,
        )
        frames.append(
            pl.DataFrame(
                {
                    "icao": f"T{i:03d}",
                    "seg": 0,
                    "t": tr.t,
                    "e": tr.pos[:, 0],
                    "n": tr.pos[:, 1],
                    "u": tr.pos[:, 2],
                }
            )
        )
    bank = TrackBank.from_tracks(pl.concat(frames))
    a, b = rng.integers(0, ntr, n_hyps), rng.integers(0, ntr, n_hyps)
    obs = np.column_stack(
        [rng.uniform(-30000, 30000, n_hyps), rng.uniform(-30000, 30000, n_hyps), np.full(n_hyps, 5.0)]
    )
    hyps = pl.DataFrame(
        {
            "hid": np.arange(n_hyps),
            "cell": np.arange(n_hyps),
            "o": t0 + rng.uniform(-3, 3, n_hyps),
            "icao0": [f"T{i:03d}" for i in a],
            "seg0": 0,
            "icao1": [f"T{i:03d}" for i in b],
            "seg1": 0,
        }
    )
    return smap, bank, hyps, obs


def main(n_hyps: int = 50_000) -> dict:
    smap, bank, hyps, obs = build(n_hyps)
    small = hyps.head(64)
    score_hypotheses(smap, bank, small, obs[:64])  # JIT warm-up (cached on disk afterwards)
    t = time.perf_counter()
    cal = calibrate(smap, bank, hyps, obs, max_hyps=400)
    t_cal = time.perf_counter() - t
    t = time.perf_counter()
    out = score_hypotheses(smap, bank, hyps, obs, cal)
    t_score = time.perf_counter() - t
    res = {
        "n_hyps": n_hyps,
        "events": 2,
        "calibrate_s": round(t_cal, 2),
        "score_s": round(t_score, 2),
        "per_hyp_ms": round(1e3 * t_score / n_hyps, 3),
        "target_s": 60.0,
        "ok": bool(t_score <= 60.0),
        "covered": float((out["cov_0"] >= 0.3).mean()),
        "z_sum_median": float(out["z_sum"].median()),
    }
    print(res)
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50_000)
