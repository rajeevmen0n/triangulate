"""Pipeline skeleton: pluggable stages, a registry, and a cached, cancellable, progress-reporting run loop.

Stage contract (how a module WP plugs in):

    class MyStage:
        name = "spectrogram"                 # one of PIPELINE_ORDER (see core.progress.STAGE_WEIGHTS)
        depends_on = ("ingest",)             # upstream stages whose cache keys feed this stage's key
        def params(self, ctx) -> dict: ...   # JSON-able params/constraints this stage depends on (cache key)
        def run(self, ctx, out_dir, progress) -> dict: ...

    register_stage(MyStage())                # replaces the stub of the same name

* `ctx` (StageContext): job_id, input_path, input_sha256, constraints (core.types.Constraints), options
  (JobParams), settings, `results[stage]` (JSON dict returned by every upstream stage), `out_dirs[stage]`
  (the directory holding its files), `check_cancel()`.
* `run` writes any files into `out_dir` (a cache directory, content-addressed) and returns a JSON-able dict
  of small results. Both are cached under data/cache/stages/ by sha256(stage, input hash, params, upstream
  keys, code version); on a cache hit `run` is not called. Files are also copied to the job workspace
  data/jobs/<id>/ (flat) so artifacts live in one place.
* `progress(stage, fraction, message="")` reports 0..1 within the stage; calling it also checks for
  cancellation (raises JobCancelled). Stages doing long work without reporting should call ctx.check_cancel().
* A stage may return {"skipped": "reason"}; downstream stages then see that dict.
* A stage (normally "render") may write results.json into out_dir; if none exists the runner writes a
  summary of all stage results.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..config import Settings
from ..core.cache import StageCache, stage_key
from ..core.progress import STAGE_WEIGHTS, overall_fraction
from ..core.types import Constraints

# Execution order. "visual" is optional and has no progress weight (plan 1.4).
PIPELINE_ORDER: list[str] = [
    "ingest", "metadata", "spectrogram", "events", "adsb", "dem", "cpa", "match", "score",
    "visual", "fusion", "render",
]


class JobCancelled(Exception):
    pass


class JobParams(BaseModel):
    """Everything a job needs besides the file; stored as params_json."""

    constraints: Constraints
    visual: bool = False  # opt-in, off by default (Decision 3)
    visual_consent: bool = False  # explicit consent that frames go to the Claude API
    events: list[dict[str, Any]] | None = None  # user-edited/provided events replacing auto-detection
    weights: dict[str, float] = Field(
        default_factory=lambda: {"acoustic": 1.0, "visual": 0.5, "metadata": 1.0}
    )
    llm_budget_usd: float | None = None  # None -> settings.llm_budget_usd (default $2 hard cap)
    use_metadata: bool = True  # read container metadata (untrusted hints); False for fixtures like audio.mp3
    trust_gps: bool = False  # user trusts the file's GPS: it becomes pinned candidate #0 (plan 3.5)
    input_name: str = "input"


@dataclass
class StageContext:
    job_id: str
    input_path: Path
    input_sha256: str
    params: JobParams
    settings: Settings
    workdir: Path
    cancel_check: Callable[[], bool] = lambda: False
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    out_dirs: dict[str, Path] = field(default_factory=dict)

    @property
    def constraints(self) -> Constraints:
        return self.params.constraints

    @property
    def llm_budget_usd(self) -> float:
        b = self.params.llm_budget_usd
        cap = self.settings.llm_budget_usd
        return cap if b is None else min(b, cap)

    def check_cancel(self) -> None:
        if self.cancel_check():
            raise JobCancelled()


class Stage(Protocol):
    name: str
    depends_on: tuple[str, ...]

    def params(self, ctx: StageContext) -> Any: ...
    def run(self, ctx: StageContext, out_dir: Path, progress: Callable[..., None]) -> dict[str, Any]: ...


class StubStage:
    """Placeholder until the real module registers itself. Writes `<name>.json` and reports progress."""

    def __init__(self, name: str, depends_on: tuple[str, ...] = (), steps: int = 3):
        self.name, self.depends_on, self.steps = name, depends_on, steps

    def params(self, ctx: StageContext) -> Any:
        return {"stub": True}

    def run(self, ctx: StageContext, out_dir: Path, progress: Callable[..., None]) -> dict[str, Any]:
        for i in range(self.steps):
            ctx.check_cancel()
            progress(self.name, (i + 1) / self.steps, f"{self.name} (stub) {i + 1}/{self.steps}")
        out = {"stub": True, "stage": self.name}
        (out_dir / f"{self.name}.json").write_text(json.dumps(out))
        return out


class _VisualStub(StubStage):
    def params(self, ctx: StageContext) -> Any:
        return {"stub": True, "enabled": self.enabled(ctx), "budget": ctx.llm_budget_usd}

    @staticmethod
    def enabled(ctx: StageContext) -> bool:
        return ctx.params.visual and ctx.params.visual_consent and ctx.settings.llm_enabled

    def run(self, ctx, out_dir, progress):
        if not self.enabled(ctx):
            why = ("not requested" if not ctx.params.visual else
                   "no consent" if not ctx.params.visual_consent else "ANTHROPIC_API_KEY not set")
            return {"skipped": why}
        return super().run(ctx, out_dir, progress)


_DEPS = {
    "metadata": ("ingest",), "spectrogram": ("ingest",), "events": ("spectrogram",),
    "adsb": ("events",), "dem": (), "cpa": ("events", "adsb", "dem"), "match": ("cpa",),
    "score": ("match",), "visual": ("ingest",), "fusion": ("score", "visual", "metadata"),
    "render": ("fusion",),
}

_REGISTRY: dict[str, Stage] = {}


def register_stage(stage: Stage) -> None:
    """Register (or replace) the implementation of a pipeline stage."""
    if stage.name not in PIPELINE_ORDER:
        raise ValueError(f"unknown stage {stage.name!r}; expected one of {PIPELINE_ORDER}")
    _REGISTRY[stage.name] = stage


def reset_registry() -> None:
    """Restore all stubs (tests)."""
    _REGISTRY.clear()
    for n in PIPELINE_ORDER:
        _REGISTRY[n] = _VisualStub(n, _DEPS.get(n, ())) if n == "visual" else StubStage(n, _DEPS.get(n, ()))


reset_registry()


def get_stages(names: list[str] | None = None) -> list[Stage]:
    return [_REGISTRY[n] for n in (names or PIPELINE_ORDER)]


def workspace(settings: Settings, job_id: str) -> Path:
    return settings.jobs_dir / job_id


def run_pipeline(
    ctx: StageContext,
    stages: list[Stage],
    on_progress: Callable[[str, float, str], None],
    cache: StageCache | None = None,
) -> dict[str, Any]:
    """Run stages in order. `on_progress(stage, overall_fraction, message)` is monotonic in the fraction.
    Raises JobCancelled between/inside stages. Returns {results_path, cached, keys}."""
    cache = cache or StageCache(ctx.settings.cache_subdir("stages"))
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    weights = {s.name: STAGE_WEIGHTS.get(s.name, 0.0) for s in stages}
    if sum(weights.values()) == 0:
        weights = {s.name: 1.0 for s in stages}
    last = [0.0]
    keys: dict[str, str] = {}
    cached: dict[str, bool] = {}

    def emit(stage: str, frac: float, msg: str = "") -> None:
        ctx.check_cancel()
        o = max(last[0], overall_fraction(stage, frac, weights)) if weights.get(stage) else last[0]
        last[0] = o
        on_progress(stage, o, msg)

    for st in stages:
        ctx.check_cancel()
        key = stage_key(
            st.name,
            {"input": ctx.input_sha256, "deps": {d: keys[d] for d in st.depends_on if d in keys}},
            {"params": st.params(ctx), "constraints": ctx.constraints.model_dump(mode="json")
             if st.name != "ingest" else None, "events": ctx.params.events if st.name == "events" else None},
        )
        keys[st.name] = key
        hit = cache.get_dir(key)
        if hit is not None:
            out = json.loads((hit / "stage_result.json").read_text())
            cached[st.name] = True
            emit(st.name, 1.0, f"{st.name}: cached")
        else:
            d = cache.begin(key)
            emit(st.name, 0.0, f"{st.name}: start")
            out = st.run(ctx, d, emit) or {}
            (d / "stage_result.json").write_text(json.dumps(out, default=str))
            cache.commit(key)
            cached[st.name] = False
            emit(st.name, 1.0, f"{st.name}: done")
        d = cache.dir_for(key)
        ctx.results[st.name] = out
        ctx.out_dirs[st.name] = d
        for f in d.iterdir():
            if f.is_file() and not f.name.startswith(".") and f.name != "stage_result.json":
                shutil.copy2(f, ctx.workdir / f.name)

    res_path = ctx.workdir / "results.json"
    if not res_path.exists():
        summary = {"job_id": ctx.job_id, "stages": ctx.results, "cached": cached, "stub": True}
        res_path.write_text(json.dumps(summary, default=str, indent=2))
    return {"results_path": str(res_path), "cached": cached, "keys": keys}
