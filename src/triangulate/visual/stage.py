"""The "visual" pipeline stage (plan 3.6 steps 1-4): local frames + OCR + timestamp OCR, then the
Claude steps when the job has visual consent and a key.

Not wired automatically: call `register()` (e.g. from the CLI/API start-up) to replace the WP18 stub.
WP16 adds the likelihood layers from `clues.json` (see `VisualClues.geocode_queries()` and the clue
implications/osm_hints).

Returned dict (stage contract): `ModuleResult.model_dump()` plus `clues_path`, `n_frames`, `ocr_path`,
`timestamp_hint`; also `skipped` (the reason) whenever the Claude steps did not run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..log import get_logger
from .llm import LlmOptions, run_visual_llm, to_module_result
from .schemas import SCHEMA_VERSION

log = get_logger("visual.stage")


class VisualStage:
    name = "visual"
    depends_on: tuple[str, ...] = ("ingest",)

    def __init__(self, client_factory: Callable[[Any], Any] | None = None,
                 options: LlmOptions | None = None, run_ocr: bool = True):
        self.client_factory = client_factory  # test seam: settings -> client
        self.options = options
        self.run_ocr = run_ocr

    def _opt(self, ctx) -> LlmOptions:
        return self.options or LlmOptions.from_settings(ctx.settings)

    def params(self, ctx) -> dict:
        o = self._opt(ctx)
        return {"requested": ctx.params.visual, "consent": ctx.params.visual_consent,
                "llm": bool(ctx.settings.llm_enabled or self.client_factory), "budget": ctx.llm_budget_usd,
                "triage_model": o.triage_model, "clue_model": o.clue_model, "schema": SCHEMA_VERSION,
                "notes": getattr(ctx.params, "notes", None)}

    def run(self, ctx, out_dir: Path, progress: Callable[..., None]) -> dict:
        from .frames import extract_frames
        from .ocr import ocr_frames, video_timestamp_hint

        if not ctx.params.visual:
            return {"skipped": "visual module not requested for this job"}
        progress(self.name, 0.02, "visual: extracting frames")
        try:
            frames = extract_frames(ctx.input_path, out_dir / "frames")
        except Exception as e:  # noqa: BLE001 - audio-only uploads have no video stream
            return {"skipped": f"no usable video stream ({type(e).__name__})"}
        if not frames:
            return {"skipped": "no video frames"}
        ctx.check_cancel()
        ocr, hint = [], None
        if self.run_ocr:
            progress(self.name, 0.1, f"visual: OCR on {len(frames)} frames")
            try:
                ocr = ocr_frames(frames)
            except Exception as e:  # noqa: BLE001
                log.warning("OCR failed: %s", e)
            try:
                hint = video_timestamp_hint(ctx.input_path)
            except Exception as e:  # noqa: BLE001
                log.warning("timestamp OCR failed: %s", e)
        (out_dir / "ocr.json").write_text(json.dumps([o.model_dump() for o in ocr], indent=1))
        ctx.check_cancel()

        client = None
        if ctx.params.visual_consent and self.client_factory is not None:
            client = self.client_factory(ctx.settings)
        c = ctx.constraints
        vc = run_visual_llm(
            frames, ocr, out_dir=out_dir, budget_usd=ctx.llm_budget_usd, consent=ctx.params.visual_consent,
            settings=ctx.settings, client=client, options=self._opt(ctx), video_path=ctx.input_path,
            bbox=c.bbox, tz=c.tz, date=c.date_start.isoformat(), notes=getattr(ctx.params, "notes", None),
            check_cancel=ctx.check_cancel,
            progress=lambda f, m: progress(self.name, 0.2 + 0.8 * f, f"visual: {m}"),
        )
        res = to_module_result(vc).model_dump(mode="json")
        res.update({"clues_path": "clues.json", "ocr_path": "ocr.json", "n_frames": len(frames),
                    "timestamp_hint": hint.model_dump(mode="json") if hint else None})
        if vc.status == "skipped":
            res["skipped"] = vc.message
        return res


def register(**kw: Any) -> VisualStage:
    from ..jobs.pipeline import register_stage

    st = VisualStage(**kw)
    register_stage(st)
    return st
