"""Null calibration of the locked scores and the acoustic log-likelihood (plan 3.3.7, WP11).

Decoys. For each event k, the per-event statistic ``max_{dt, f0} S_k`` (the same sweep as the real
score: dt in +-5 s, f0 over the event's range) is evaluated for hypotheses whose model curve is
*wrong on purpose*:

* time-shifted: the hypothesis' own curve placed at audio time tau_k + Delta with
  |Delta| in [30, 300] s (shifts whose window would leave the recording are skipped), and
* time-reversed: the curve mirrored in time (a rising sweep) at the right time.

Brightness decoys mostly land on noise, hum residue or other aircraft, i.e. on what a wrong
location would see. For traced events the trace is fixed, so a shifted decoy compares the trace with
the curve displaced by Delta in [10, 120] s.

From the decoys (pooled over up to ``max_hyps`` hypotheses) the robust location and scale
(median, 1.4826 MAD, floored) give ``z_k = (S_k - mu0_k) / sigma0_k``. The acoustic log-likelihood is
``ell_A = lam * sum_k z_k`` with lam = 0.5 by default (WP22 fits lam for 90 % HPD coverage);
outside the hypotheses ell_A = log(p_miss), p_miss = 0.1 (ADS-B can miss the aircraft).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from .score import PairSet, ScoreMap, TrackBank, _hyp_pairs, observer_lookup, score_pairs

__all__ = [
    "LAMBDA_DEFAULT",
    "P_MISS_DEFAULT",
    "EventCalibration",
    "Calibration",
    "calibrate",
    "loglik",
    "BRIGHT_SHIFTS_S",
    "LINE_SHIFTS_S",
]

LAMBDA_DEFAULT = 0.5
P_MISS_DEFAULT = 0.1
BRIGHT_SHIFTS_S = tuple(float(s) for s in (90, 120, 150, 180, 210, 240, 270, 300))
LINE_SHIFTS_S = tuple(float(s) for s in (10, 15, 20, 30, 45, 60, 90, 120))


@dataclass
class EventCalibration:
    id: str
    kind: str
    mu: float
    sigma: float
    n: int
    q90: float = float("nan")  # 90th percentile of the decoy z (diagnostic)
    q99: float = float("nan")


@dataclass
class Calibration:
    events: list[EventCalibration]
    lam: float = LAMBDA_DEFAULT
    p_miss: float = P_MISS_DEFAULT
    stats: dict = field(default_factory=dict)

    @property
    def ell_floor(self) -> float:
        return math.log(self.p_miss)

    def z(self, k: int, S):
        c = self.events[k]
        return (np.asarray(S, dtype=float) - c.mu) / c.sigma

    def to_dict(self) -> dict:
        return {
            "lam": self.lam,
            "p_miss": self.p_miss,
            "ell_floor": self.ell_floor,
            "events": [asdict(e) for e in self.events],
            "stats": self.stats,
        }


def loglik(z_sum, lam: float = LAMBDA_DEFAULT):
    """ell_A of one (cell, tuple) hypothesis: lam * sum_k z_k."""
    return lam * np.asarray(z_sum, dtype=float)


def _robust(x: np.ndarray, floor: float) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, max(floor, 1e-6)
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med, max(mad, floor, 1e-6)


def calibrate(
    smap: ScoreMap,
    bank: TrackBank,
    hyps: pl.DataFrame,
    observers,
    *,
    lam: float = LAMBDA_DEFAULT,
    p_miss: float = P_MISS_DEFAULT,
    max_hyps: int = 400,
    bright_shifts_s=BRIGHT_SHIFTS_S,
    line_shifts_s=LINE_SHIFTS_S,
    reverse: bool = True,
    seed: int = 0,
    sigma_floor: float = 1e-4,
    line_sigma_floor: float = 0.02,
) -> Calibration:
    """Per-event decoy null (mu0_k, sigma0_k) from a random subset of the hypotheses."""
    K = len(smap.events)
    if hyps.height == 0:
        raise ValueError("no hypotheses to calibrate on")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(hyps.height, size=min(max_hyps, hyps.height), replace=False))
    hs = hyps[sel]
    if isinstance(observers, np.ndarray):
        xyz = observers[sel]
    else:
        xyz = observer_lookup(observers, hs["cell"].to_numpy())
    base = _hyp_pairs(bank, hs, xyz, K)
    cfg = smap.cfg
    half = (cfg.n_coarse // 2) * cfg.coarse_step
    parts: list[PairSet] = []
    shifts_used: dict[str, list[float]] = {}
    for k, plan in enumerate(smap.plans):
        m = np.flatnonzero(base.ev == k)
        shifts = line_shifts_s if plan.kind == "line" else bright_shifts_s
        used = []
        for s in shifts:
            for sgn in (-1.0, 1.0):
                sh = int(round(sgn * s / smap.fdt))
                if not smap.shift_ok(k, sh, half):
                    continue
                used.append(sgn * s)
                ps = base.subset(m)
                ps.shift0 = np.full(m.size, sh, np.int64)
                ps.rev = np.zeros(m.size, np.bool_)
                parts.append(ps)
        if reverse:
            ps = base.subset(m)
            ps.shift0 = np.zeros(m.size, np.int64)
            ps.rev = np.ones(m.size, np.bool_)
            parts.append(ps)
        shifts_used[plan.id] = used
    allp = PairSet(
        seg=np.concatenate([p.seg for p in parts]),
        obs=np.concatenate([p.obs for p in parts]),
        o=np.concatenate([p.o for p in parts]),
        ev=np.concatenate([p.ev for p in parts]),
        shift0=np.concatenate([p.shift0 for p in parts if p.shift0 is not None]),
        rev=np.concatenate([p.rev for p in parts if p.rev is not None]),
    )
    S, _, _ = score_pairs(smap, bank, allp, cfg.n_coarse, cfg.coarse_step)
    with np.errstate(invalid="ignore"):
        stat = np.nanmax(np.where(np.isfinite(S), S, -np.inf), axis=1)
    stat = np.where(np.isfinite(stat), stat, np.nan)
    out = []
    for k, plan in enumerate(smap.plans):
        x = stat[allp.ev == k]
        mu, sg = _robust(x, line_sigma_floor if plan.kind == "line" else sigma_floor)
        zz = (x[np.isfinite(x)] - mu) / sg
        out.append(
            EventCalibration(
                id=plan.id,
                kind=plan.kind,
                mu=mu,
                sigma=sg,
                n=int(np.isfinite(x).sum()),
                q90=float(np.quantile(zz, 0.9)) if zz.size else float("nan"),
                q99=float(np.quantile(zz, 0.99)) if zz.size else float("nan"),
            )
        )
    return Calibration(
        events=out,
        lam=lam,
        p_miss=p_miss,
        stats={"n_hyps": int(sel.size), "shifts_s": shifts_used, "reverse": reverse},
    )
