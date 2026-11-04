"""Progress reporting: stages report (stage, fraction, message)."""

from __future__ import annotations

from typing import Protocol

# Stage weights for overall job progress (plan 1.4). The visual module runs in parallel and has none.
STAGE_WEIGHTS: dict[str, float] = {
    "ingest": 0.02, "metadata": 0.01, "spectrogram": 0.03, "events": 0.08, "adsb": 0.20,
    "dem": 0.03, "cpa": 0.20, "match": 0.05, "score": 0.25, "fusion": 0.03, "render": 0.02,
}


class Progress(Protocol):
    def __call__(self, stage: str, fraction: float, message: str = "") -> None: ...


def null_progress(stage: str, fraction: float, message: str = "") -> None:
    """A Progress that discards everything."""


def overall_fraction(stage: str, fraction: float, weights: dict[str, float] = STAGE_WEIGHTS) -> float:
    """Overall job progress in [0, 1] given the current stage and its own fraction.

    Stages are ordered as in `weights`; unknown stages (e.g. 'visual') return 0 contribution
    beyond the stages before them, so callers should not feed them here.
    """
    done = 0.0
    total = sum(weights.values())
    for name, w in weights.items():
        if name == stage:
            done += w * min(max(fraction, 0.0), 1.0)
            break
        done += w
    return min(done / total, 1.0)
