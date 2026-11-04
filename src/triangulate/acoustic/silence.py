"""Level 4: silence filter / negative evidence (plan 3.3.5, WP10).

A hypothesis (cell, tuple of passes, implied start o) is rejected when some *other* flight has a pass at
that cell with t_h inside the silent span and `d <= kappa * max_k d_k` (kappa = 1: the video's
"no multiplier"), unless that pass lands within +-10 s of a *detected but unselected* event (that
flight was heard).

Silent span (audio-relative, shifted by o per hypothesis), guard g = 60 s around every selected event:
- `scope="between"` (plan default, N >= 2): [tau_first + g, tau_last - g], minus the guard of any event
  in between (so for N = 3 the two gaps are tested separately).
- `scope="full"` (needs `duration`): the whole covered interval [0, duration] minus +-g around every
  event. This is the only meaningful span for a single-event recording.
- Single event with `scope="between"`: the span is empty, so the filter is a no-op (flagged in stats).

"Other flight" = an icao not in the hypothesis tuple (`own_exempt=True`); with False, every flight
counts, including another pass of the tuple's own aircraft.

Implementation: one second CPA computation (`compute_cpa` with `EventGate.window`, any speed, all
passes within kappa * dmax) only at the surviving cells, then a DuckDB anti-join. `soft_beta > 0`
keeps violators and reports `penalty = beta * sum_over_violating_flights(1 - d / R)` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb
import numpy as np
import polars as pl

from ..core.types import Event
from ..fusion.grid import pack_cells
from .cpa import EventGate, TrackSeg, compute_cpa, track_segments
from .grid import ObserverCells
from .matching import hyp_icaos
from .physics import C_DEFAULT, Track

__all__ = ["silence_filter", "silent_spans", "SilenceResult", "GUARD_S", "EXEMPT_S"]

GUARD_S = 60.0
EXEMPT_S = 10.0


def silent_spans(taus: Sequence[float], guard: float = GUARD_S, scope: str = "between",
                 duration: float | None = None) -> list[tuple[float, float]]:
    """Audio-relative silent intervals (possibly empty)."""
    ts = sorted(taus)
    if scope == "between":
        edges = list(zip(ts[:-1], ts[1:], strict=True))
        pts = [(a + guard, b - guard) for a, b in edges]
    elif scope == "full":
        if duration is None:
            raise ValueError("scope='full' needs the recording duration")
        bounds = [0.0, *[x for t in ts for x in (t - guard, t + guard)], duration]
        pts = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)]
        pts = [(max(a, 0.0), min(b, duration)) for a, b in pts]
    else:
        raise ValueError(f"unknown scope {scope!r}")
    return [(a, b) for a, b in pts if b > a]


@dataclass
class SilenceResult:
    hyps: pl.DataFrame  # surviving hypotheses (+ `penalty` in soft mode)
    rejected: pl.DataFrame  # hid, icao (first violator), t_h, d of the nearest violating pass
    passes: pl.DataFrame  # the silent-span CPA passes computed
    stats: dict = field(default_factory=dict)


def _exempt_windows(events: Sequence[Event], exempt_s: float) -> list[tuple[float, float]]:
    return [(min(e.t_h, e.t_h_range.lo) - exempt_s, max(e.t_h, e.t_h_range.hi) + exempt_s) for e in events]


def silence_filter(hyps: pl.DataFrame, tracks: pl.DataFrame | Sequence[Track] | Sequence[TrackSeg],
                   observers: ObserverCells, taus: Sequence[float], *, kappa: float = 1.0,
                   guard: float = GUARD_S, scope: str = "between", duration: float | None = None,
                   exempt_events: Sequence[Event] = (), exempt_s: float = EXEMPT_S,
                   own_exempt: bool = True, soft_beta: float = 0.0, c: float = C_DEFAULT,
                   n_threads: int | None = None) -> SilenceResult:
    """Drop (or penalise) hypotheses that would have heard another flight. `tracks` = all flights to
    consider (jets excluded by the caller by default); `observers` must contain the hypothesis cells
    (e.g. `CpaResult.observers`); `exempt_events` = detected-but-unselected `Event`s."""
    spans = silent_spans(taus, guard, scope, duration)
    stats: dict = {"n_in": hyps.height, "cells_in": hyps["cell"].n_unique() if hyps.height else 0,
                   "spans": spans, "kappa": kappa}
    if not spans or hyps.height == 0:
        stats.update(n_out=hyps.height, cells_out=stats["cells_in"], noop=not spans)
        empty = pl.DataFrame(schema={"hid": pl.Int64, "icao": pl.String, "t_h": pl.Float64, "d": pl.Float64})
        return SilenceResult(hyps, empty, pl.DataFrame(), stats)
    cells = np.unique(hyps["cell"].to_numpy())
    keys = pack_cells(observers.cells.level, observers.cells.ix, observers.cells.iy)
    sel = np.flatnonzero(np.isin(keys, cells))
    if sel.size != cells.size:
        raise ValueError("observers do not cover all hypothesis cells")
    obs = observers.subset(sel)
    r_max = kappa * float(hyps["dmax"].max())
    lo = float(hyps["o"].min()) + min(a for a, _ in spans) - 1.0
    hi = float(hyps["o"].max()) + max(b for _, b in spans) + 1.0
    segs = track_segments(tracks)
    passes = compute_cpa(segs, obs, [EventGate.window("silence", r_max, lo, hi)], c, n_threads)
    stats["silence_passes"] = passes.height

    con = duckdb.connect()
    con.register("h", hyps.select("hid", "cell", "o", "dmax", *hyp_icaos(hyps)).to_arrow())
    con.register("p", passes.select("cell", "icao", "t_h", pl.col("d").cast(pl.Float64)).to_arrow())
    sp = pl.DataFrame({"a": [a for a, _ in spans], "b": [b for _, b in spans]})
    ex = pl.DataFrame({"a": [a for a, _ in _exempt_windows(exempt_events, exempt_s)],
                       "b": [b for _, b in _exempt_windows(exempt_events, exempt_s)]},
                      schema={"a": pl.Float64, "b": pl.Float64})
    con.register("sp", sp.to_arrow())
    con.register("ex", ex.to_arrow())
    own = ""
    if own_exempt:
        own = " AND " + " AND ".join(f"p.icao <> h.{c_}" for c_ in hyp_icaos(hyps))
    viol = con.execute(f"""
        SELECT h.hid, p.icao, p.t_h, p.d, {kappa!r} * h.dmax AS r
        FROM h JOIN p ON h.cell = p.cell
        WHERE p.d <= {kappa!r} * h.dmax {own}
          AND EXISTS (SELECT 1 FROM sp WHERE p.t_h BETWEEN h.o + sp.a AND h.o + sp.b)
          AND NOT EXISTS (SELECT 1 FROM ex WHERE p.t_h BETWEEN h.o + ex.a AND h.o + ex.b)
    """).pl()
    if soft_beta > 0:
        pen = (viol.group_by("hid", "icao").agg(pl.col("d").min(), pl.col("r").first())
               .group_by("hid").agg((soft_beta * (1.0 - pl.col("d") / pl.col("r"))).sum().alias("penalty")))
        out = hyps.join(pen, on="hid", how="left").with_columns(pl.col("penalty").fill_null(0.0))
    else:
        bad = viol["hid"].unique()
        out = hyps.filter(~pl.col("hid").is_in(bad.implode()))
    nearest = viol.sort("d").group_by("hid", maintain_order=True).first().select("hid", "icao", "t_h", "d")
    stats.update(n_out=out.height, cells_out=out["cell"].n_unique() if out.height else 0,
                 n_violating=int(viol["hid"].n_unique()), noop=False)
    return SilenceResult(out, nearest, passes, stats)
