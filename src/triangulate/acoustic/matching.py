"""Level 3 of the acoustic search: offset matching ("CPA timeline", plan 3.3.4, WP10).

A selected event k has an audio-relative heard time tau_k. A CPA pass heard at absolute time t_h implies
the recording start `o = t_h - tau_k`. A *hypothesis* is a cell plus one eligible pass per selected event
whose implied starts all agree: every pairwise |o_j - o_k| <= slack (default 5 s, the video's value).
For N > 2 the join chains on all earlier events, so the result is exactly the set of N-cliques.

- The same aircraft may explain two events if the passes differ (distinct (icao, seg, pass) keys).
- Rows with o outside `t_s = (T_lo, T_hi)` are dropped. Single-event mode (N = 1) requires `t_s`
  (a known absolute time, plan 3.3.4); without it every pass would be a hypothesis.
- Implemented as a DuckDB join over the `cpa.parquet` table (`acoustic.cpa.CPA_SCHEMA`).

Output columns (`k` = index of the event in the `taus` list): hid, cell, level, ix, iy, o (mean implied
start, epoch s), o_spread (max - min), dmax (max_k d_k), and per event icao_k, seg_k, pass_k, th_k, d_k, v_k.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import polars as pl

from ..core.types import Event

__all__ = ["match_offsets", "taus_from_events", "hyp_icaos", "SLACK_S"]

SLACK_S = 5.0


def taus_from_events(events: Sequence[Event]) -> list[float]:
    """Audio-relative heard times of the selected events (order = gate bit order)."""
    return [float(e.t_h) for e in events]


def match_offsets(table: pl.DataFrame, taus: Sequence[float], slack: float = SLACK_S,
                  t_s: tuple[float, float] | None = None,
                  gate_bits: Sequence[int] | None = None) -> pl.DataFrame:
    """Hypotheses from a CPA table. `gate_bits[k]` = eligibility bit of event k (default k)."""
    n = len(taus)
    if n < 1:
        raise ValueError("need at least one event")
    bits = list(gate_bits) if gate_bits is not None else list(range(n))
    if len(bits) != n:
        raise ValueError("gate_bits must match taus")
    if n == 1 and t_s is None:
        raise ValueError("single-event mode needs a known start-time window t_s")
    lo, hi = (t_s if t_s is not None else (float("-inf"), float("inf")))
    con = duckdb.connect()
    con.register("cpa", table.to_arrow())
    ctes, cols = [], ["e0.cell AS cell", "e0.level AS level", "e0.ix AS ix", "e0.iy AS iy"]
    joins = []
    for k in range(n):
        win = f"AND t_h - {taus[k]!r} BETWEEN {lo!r} AND {hi!r}" if t_s is not None else ""
        ctes.append(
            f"e{k} AS (SELECT cell, level, ix, iy, icao AS icao{k}, seg AS seg{k}, pass AS pass{k}, "
            f"t_h AS th{k}, d AS d{k}, v AS v{k}, t_h - {taus[k]!r} AS o{k} FROM cpa "
            f"WHERE (eligible & {1 << bits[k]}) <> 0 {win})")
        cols += [f"icao{k}", f"seg{k}", f"pass{k}", f"th{k}", f"d{k}", f"v{k}", f"o{k}"]
        if k:
            cond = [f"e0.cell = e{k}.cell"]
            for j in range(k):
                cond.append(f"abs(o{j} - o{k}) <= {slack!r}")
                cond.append(f"NOT (icao{j} = icao{k} AND seg{j} = seg{k} AND pass{j} = pass{k})")
            joins.append(f"JOIN e{k} ON " + " AND ".join(cond))
    sql = f"WITH {', '.join(ctes)} SELECT {', '.join(cols)} FROM e0 {' '.join(joins)}"
    df = con.execute(sql).pl()
    os_ = [pl.col(f"o{k}") for k in range(n)]
    ds = [pl.col(f"d{k}") for k in range(n)]
    df = df.with_columns(o=pl.mean_horizontal(*os_),
                         o_spread=pl.max_horizontal(*os_) - pl.min_horizontal(*os_),
                         dmax=pl.max_horizontal(*ds).cast(pl.Float64)).drop([f"o{k}" for k in range(n)])
    df = df.sort("cell", "o").with_row_index("hid")
    return df.with_columns(pl.col("hid").cast(pl.Int64))


def hyp_icaos(hyps: pl.DataFrame) -> list[str]:
    """Column names holding the tuple's flights."""
    return [c for c in hyps.columns if c.startswith("icao")]
