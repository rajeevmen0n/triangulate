"""Claude triage + clue extraction (plan 3.6 steps 3-4): client, budget guard, refusal handling,
server-side fallbacks, usage/cost logging.

Public API:
    run_visual_llm(frames, ocr, *, out_dir, budget_usd, consent, ...) -> VisualClues   (writes clues.json)
    to_module_result(vc) -> core.types.ModuleResult
    LlmOptions, BudgetGuard, cost_usd, estimate_cost, image_tokens, PRICES, PRICING_SOURCE

Gate: nothing is sent unless the job has visual consent AND an API key (or an explicitly injected
client, the test seam). Otherwise the result is status "skipped" with the reason.

Budget: before every call the input is counted with `messages.count_tokens` (fallback: a conservative
local estimate), and the worst case (input x 1.05 at cache-write rates if cached, plus `max_tokens` of
output, plus a second full attempt on the most expensive fallback model when server-side fallbacks are
on) must fit in what is left of the cap. Actual cost is taken from `usage` (per attempt, from
`usage.iterations`, when present). When the next call does not fit, the module stops and returns what
it has (status "degraded").

API surface used (anthropic 1.8.0): `messages.create` / `beta.messages.create` with
`output_config={"effort": ..., "format": {"type": "json_schema", "schema": ...}}`, `messages.count_tokens`,
base64 image blocks, `cache_control` on the system block, `fallbacks="default"` with beta
`server-side-fallback-2026-07-01` (Opus only; Haiku 5.5 has no server-side fallback). The JSON is
validated locally rather than with `messages.parse`, so a malformed or truncated reply still has its
usage recorded and charged.
"""

from __future__ import annotations

import base64
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, Field

from ..core.types import ModuleResult
from ..log import get_logger
from . import prompts
from .frames import Frame, hamming
from .ocr import FrameOcr, ocr_text_block
from .schemas import (
    ClueReport,
    FrameTriage,
    FrameTriageItem,
    SentFrame,
    UsageRecord,
    VisualClues,
    api_schema,
    validate_clue_report,
    validate_triage,
)

log = get_logger("visual.llm")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
STRUCTURED_BETA = "structured-outputs-2025-12-15"  # what the SDK's beta parse() adds for output_config.format

# --------------------------------------------------------------------------- pricing

PRICING_SOURCE = (
    "claude-api skill (bundled 2.1.318): model table 'Current Models (cached: 2026-10-06)' and "
    "shared/model-migration.md (Haiku 5.5 two rate cards split at a 100K-token prompt; cache reads 0.1x, "
    "5-min cache writes 1.25x; Opus 5.5 cache reads $0.20/MTok)"
)


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    cache_read_mult: float = 0.1
    cache_write_mult: float = 1.25  # 5-minute TTL
    long_threshold: int | None = None  # prompt tokens above which the long-context card applies
    long_input: float | None = None
    long_output: float | None = None

    def rates(self, prompt_tokens: int) -> tuple[float, float]:
        if self.long_threshold is not None and prompt_tokens > self.long_threshold:
            return self.long_input or self.input, self.long_output or self.output
        return self.input, self.output


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, long_threshold=100_000, long_input=0.50, long_output=2.50),
    "claude-opus-5-5": ModelPrice(4.00, 20.00, cache_read_mult=0.05),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "claude-fable-5-1": ModelPrice(10.00, 50.00, cache_read_mult=0.025),
    "claude-fable-5": ModelPrice(10.00, 50.00),
}
# Unknown model ids (e.g. a configured newer model) are priced at the most expensive known card.
UNKNOWN_PRICE = ModelPrice(10.00, 50.00, cache_read_mult=0.1)
# Worst-case price of whatever `fallbacks="default"` routes to (Opus 5 / Opus 4.8 for Opus 5.5).
FALLBACK_WORST = "claude-opus-5"


def price_for(model: str) -> ModelPrice:
    return PRICES.get(model, UNKNOWN_PRICE)


def cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0, cache_read: int = 0,
             cache_write: int = 0) -> float:
    p = price_for(model)
    pin, pout = p.rates(input_tokens + cache_read + cache_write)
    return (input_tokens * pin + cache_write * pin * p.cache_write_mult
            + cache_read * pin * p.cache_read_mult + output_tokens * pout) / 1e6


def estimate_cost(model: str, input_tokens: int, max_tokens: int, *, cached: bool = False,
                  fallbacks: bool = False, margin: float = 1.05) -> float:
    """Upper bound for one call: input (x margin, at cache-write rate if cached) + max_tokens output, plus
    a full second attempt on the worst fallback model when server-side fallbacks are enabled."""
    n = int(math.ceil(input_tokens * margin)) + 16
    est = cost_usd(model, 0 if cached else n, max_tokens, cache_write=n if cached else 0)
    if fallbacks:
        est += cost_usd(FALLBACK_WORST, n, max_tokens)
    return est


def image_tokens(w: int, h: int) -> int:
    """Vision cost of one image (plan 0: ceil(w/28) * ceil(h/28) on 4.7+ models)."""
    return math.ceil(w / 28) * math.ceil(h / 28)


# --------------------------------------------------------------------------- options / budget

class LlmOptions(BaseModel):
    triage_model: str = "claude-haiku-5-5"
    clue_model: str = "claude-opus-5-5"
    triage_effort: str = "low"
    clue_effort: str = "high"
    triage_batch: int = Field(20, ge=1, le=20)  # <= 20 images per request (plan 0)
    triage_long_edge: int = Field(1280, ge=256, le=2576)
    clue_long_edge: int = Field(1920, ge=256, le=2576)  # ~2.7k tokens per 16:9 frame
    top_k: int = Field(10, ge=1, le=20)
    min_clue_frames: int = Field(3, ge=1, le=20)
    triage_max_tokens: int = 4096  # includes adaptive thinking at effort low
    clue_max_tokens: int = 16000  # non-streaming ceiling; thinking counts toward it
    clue_min_max_tokens: int = 8000  # lowest max_tokens tried when the budget is tight
    clue_fallbacks: bool = True  # server-side fallbacks="default" for the Opus call
    triage_budget_fraction: float = Field(0.25, gt=0, le=1)  # share of the cap triage may use
    max_retries: int = 3  # SDK retries (429, 5xx, connection errors)
    timeout_s: float = 600.0
    ocr_chars: int = 6000

    @classmethod
    def from_settings(cls, settings: Any, **kw: Any) -> LlmOptions:
        return cls(triage_model=getattr(settings, "triage_model", cls.model_fields["triage_model"].default),
                   clue_model=getattr(settings, "clue_model", cls.model_fields["clue_model"].default), **kw)


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetGuard:
    cap_usd: float
    spent_usd: float = 0.0
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def allows(self, estimate_usd: float) -> bool:
        return self.spent_usd + estimate_usd <= self.cap_usd + 1e-12

    def charge(self, rec: UsageRecord) -> None:
        self.spent_usd += rec.cost_usd
        self.records.append(rec)
        if rec.cost_usd > rec.estimate_usd + 1e-9:
            log.warning("%s cost $%.4f exceeded its estimate $%.4f", rec.call, rec.cost_usd, rec.estimate_usd)
        log.info("Claude %s (%s): in=%d out=%d cache_r=%d cache_w=%d cost=$%.4f (est $%.4f); job total $%.4f "
                 "of $%.2f", rec.call, ",".join(rec.served_by) or rec.model, rec.input_tokens,
                 rec.output_tokens, rec.cache_read_input_tokens, rec.cache_creation_input_tokens,
                 rec.cost_usd, rec.estimate_usd, self.spent_usd, self.cap_usd)


class LlmError(Exception):
    pass


# --------------------------------------------------------------------------- images

def encode_image(path: str | Path, long_edge: int) -> tuple[str, int, int]:
    """JPEG base64 of an image scaled so its long edge is <= long_edge (never upscaled)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise LlmError(f"cannot read frame image {path}")
    h, w = img.shape[:2]
    s = min(1.0, long_edge / max(w, h))
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise LlmError(f"cannot encode frame image {path}")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), w, h


def hires_frame(video: str | Path, t_s: float, out_path: Path, long_edge: int) -> Path | None:
    """Re-extract one frame from the video at a larger size (local ffmpeg). None on failure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({long_edge},iw),-2)':'if(gt(iw,ih),-2,min({long_edge},ih))'"
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", f"{max(0.0, t_s):.3f}", "-i", str(video),
         "-frames:v", "1", "-an", "-sn", "-vf", scale, "-q:v", "2", str(out_path)],
        capture_output=True, text=True, check=False,
    )
    return out_path if r.returncode == 0 and out_path.exists() else None


def _image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


# --------------------------------------------------------------------------- response helpers

def _get(o: Any, k: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)


def response_text(resp: Any) -> str:
    """Concatenated text blocks after the last `fallback` block (a declined attempt's partial text is
    dropped); thinking blocks are ignored."""
    blocks = list(_get(resp, "content", None) or [])
    last = max((i for i, b in enumerate(blocks) if _get(b, "type") == "fallback"), default=-1)
    return "".join(_get(b, "text", "") or "" for b in blocks[last + 1:] if _get(b, "type") == "text")


def usage_record(resp: Any, call: str, model: str, estimate: float) -> UsageRecord:
    u = _get(resp, "usage", None)
    iters = [it for it in (_get(u, "iterations", None) or [])
             if _get(it, "type") in ("message", "fallback_message")]
    rows = iters or ([u] if u is not None else [])
    served = [(_get(it, "model", None) or _get(resp, "model", None) or model) for it in rows] if iters else \
        [_get(resp, "model", None) or model]
    tot = dict(i=0, o=0, r=0, w=0)
    cost = 0.0
    for it, m in zip(rows, served, strict=False):
        i, o = int(_get(it, "input_tokens", 0) or 0), int(_get(it, "output_tokens", 0) or 0)
        r = int(_get(it, "cache_read_input_tokens", 0) or 0)
        w = int(_get(it, "cache_creation_input_tokens", 0) or 0)
        tot["i"] += i
        tot["o"] += o
        tot["r"] += r
        tot["w"] += w
        cost += cost_usd(m, i, o, r, w)
    return UsageRecord(call=call, model=model, served_by=served, input_tokens=tot["i"],
                       output_tokens=tot["o"],
                       cache_read_input_tokens=tot["r"], cache_creation_input_tokens=tot["w"],
                       estimate_usd=round(estimate, 6), cost_usd=round(cost, 6),
                       stop_reason=_get(resp, "stop_reason", None),
                       request_id=getattr(resp, "_request_id", None) or _get(resp, "id", None))


def _refusal_category(resp: Any) -> str:
    sd = _get(resp, "stop_details", None)
    return str(_get(sd, "category", None) or "unspecified")


# --------------------------------------------------------------------------- Claude wrapper

def make_client(settings: Any, options: LlmOptions) -> Any:
    import anthropic

    key = settings.anthropic_api_key.get_secret_value()
    return anthropic.Anthropic(api_key=key, max_retries=options.max_retries, timeout=options.timeout_s)


class ClaudeVision:
    def __init__(self, client: Any, options: LlmOptions, budget: BudgetGuard,
                 check_cancel: Callable[[], None] | None = None):
        self.client, self.opt, self.budget = client, options, budget
        self.check_cancel = check_cancel or (lambda: None)
        self.warnings: list[str] = []
        self._schemas = {FrameTriage: api_schema(FrameTriage), ClueReport: api_schema(ClueReport)}

    # ---- request plumbing
    def _request(self, model: str, system: str, content: list[dict], schema_model: type[BaseModel],
                 effort: str, max_tokens: int, cache_system: bool) -> dict:
        sys_block: dict = {"type": "text", "text": system}
        if cache_system:
            sys_block["cache_control"] = {"type": "ephemeral"}
        return dict(model=model, max_tokens=max_tokens, system=[sys_block],
                    messages=[{"role": "user", "content": content}],
                    output_config={"effort": effort,
                                   "format": {"type": "json_schema", "schema": self._schemas[schema_model]}})

    def count_input(self, req: dict, image_toks: int) -> tuple[int, bool]:
        """(input tokens, exact?) via count_tokens; on any failure a conservative local estimate."""
        try:
            r = self.client.messages.count_tokens(
                model=req["model"], system=req["system"], messages=req["messages"],
                output_config=req["output_config"])
            n = int(_get(r, "input_tokens"))
            if n > 0:
                return n, True
        except Exception as e:  # noqa: BLE001 - counting is best effort; the estimate below is safe
            log.info("count_tokens failed (%s); using a local estimate", type(e).__name__)
        chars = len(req["system"][0]["text"]) + len(json.dumps(req["output_config"]["format"]["schema"]))
        chars += sum(len(b.get("text", "")) for b in req["messages"][0]["content"] if b.get("type") == "text")
        return int((image_toks + chars / 2.0 + 600) * 1.25), False

    def _send(self, req: dict, fallbacks: bool) -> Any:
        import anthropic

        try:
            if fallbacks:
                return self.client.beta.messages.create(**req, betas=[FALLBACK_BETA, STRUCTURED_BETA],
                                                        fallbacks="default")
            return self.client.messages.create(**req)
        except anthropic.AuthenticationError as e:
            raise LlmError("Claude API authentication failed (check ANTHROPIC_API_KEY)") from e
        except anthropic.PermissionDeniedError as e:
            raise LlmError("Claude API permission denied for this key/model") from e
        except anthropic.NotFoundError as e:
            raise LlmError(f"model {req['model']!r} not found") from e
        except anthropic.BadRequestError as e:
            raise LlmError(f"Claude API rejected the request: {str(e)[:200]}") from e
        except anthropic.RateLimitError as e:
            raise LlmError("Claude API rate limit (after SDK retries)") from e
        except anthropic.APIStatusError as e:
            raise LlmError(f"Claude API error {e.status_code} (after SDK retries)") from e
        except anthropic.APIConnectionError as e:
            raise LlmError("cannot reach the Claude API (after SDK retries)") from e

    def call(self, name: str, req: dict, *, estimate: float, fallbacks: bool) -> tuple[Any, UsageRecord]:
        if not self.budget.allows(estimate):
            raise BudgetExceeded(f"{name}: estimate ${estimate:.4f} > remaining ${self.budget.remaining:.4f}")
        self.check_cancel()
        resp = self._send(req, fallbacks)
        rec = usage_record(resp, name, req["model"], estimate)
        self.budget.charge(rec)
        return resp, rec

    # ---- step 3: triage
    def triage(self, frames: Sequence[Frame], ocr_by_id: dict[str, str]) -> dict[str, FrameTriageItem]:
        opt, out = self.opt, {}
        cap = self.budget.cap_usd * opt.triage_budget_fraction
        spent0 = self.budget.spent_usd
        batches = [frames[i:i + opt.triage_batch] for i in range(0, len(frames), opt.triage_batch)]
        errors = 0
        for bi, batch in enumerate(batches):
            content: list[dict] = []
            itoks = 0
            for f in batch:
                b64, w, h = encode_image(f.path, opt.triage_long_edge)
                itoks += image_tokens(w, h)
                content += [{"type": "text", "text": prompts.frame_label(f.id, f.t_s, ocr_by_id.get(f.id))},
                            _image_block(b64)]
            content.append({"type": "text", "text": prompts.TRIAGE_INSTRUCTION})
            req = self._request(opt.triage_model, prompts.TRIAGE_SYSTEM, content, FrameTriage,
                                opt.triage_effort, opt.triage_max_tokens, cache_system=True)
            n_in, _ = self.count_input(req, itoks)
            est = estimate_cost(opt.triage_model, n_in, opt.triage_max_tokens, cached=True)
            if (self.budget.spent_usd - spent0) + est > cap or not self.budget.allows(est):
                self.warnings.append(f"triage stopped after {bi}/{len(batches)} batches: budget "
                                     f"(${self.budget.spent_usd:.4f} spent of ${self.budget.cap_usd:.2f})")
                break
            try:
                resp, rec = self.call(f"triage[{bi}]", req, estimate=est, fallbacks=False)
            except LlmError as e:
                errors += 1
                self.warnings.append(f"triage[{bi}]: {e}")
                if errors >= 2:
                    self.warnings.append("triage stopped after repeated API errors")
                    break
                continue
            if rec.stop_reason == "refusal":
                self.warnings.append(f"triage[{bi}]: declined by the model ({_refusal_category(resp)}); "
                                     "batch left untriaged")
                continue
            try:
                items, vwarn = validate_triage(json.loads(response_text(resp)), {f.id for f in batch})
            except (ValueError, json.JSONDecodeError) as e:
                self.warnings.append(f"triage[{bi}]: invalid JSON ({type(e).__name__}); batch left untriaged")
                continue
            self.warnings += vwarn
            out.update({it.frame_id: it for it in items})
        return out

    # ---- step 4: clues
    def extract_clues(self, sel: Sequence[tuple[Frame, Path]], ocr_text: str, context: str,
                      known_frames: dict[str, float]) -> tuple[ClueReport | None, str, str, list[SentFrame]]:
        """Returns (report, status, message, frames actually sent)."""
        opt = self.opt
        imgs = []
        for f, p in sel:
            b64, w, h = encode_image(p, opt.clue_long_edge)
            imgs.append((f, b64, w, h))

        def content_for(n: int) -> list[dict]:
            c: list[dict] = []
            for f, b64, _, _ in imgs[:n]:
                c += [{"type": "text", "text": prompts.frame_label(f.id, f.t_s)}, _image_block(b64)]
            c += [{"type": "text", "text": prompts.wrap_ocr(ocr_text, opt.ocr_chars)},
                  {"type": "text", "text": context},
                  {"type": "text", "text": prompts.CLUE_INSTRUCTION}]
            return c

        full = self._request(opt.clue_model, prompts.CLUE_SYSTEM, content_for(len(imgs)), ClueReport,
                             opt.clue_effort, opt.clue_max_tokens, cache_system=False)
        itoks = [image_tokens(w, h) for _, _, w, h in imgs]
        n_full, _ = self.count_input(full, sum(itoks))
        choice = None
        lo = min(opt.min_clue_frames, len(imgs))
        for n in range(len(imgs), lo - 1, -1):
            n_in = n_full - sum(itoks[n:])
            for mt in dict.fromkeys((opt.clue_max_tokens, min(opt.clue_max_tokens, opt.clue_min_max_tokens))):
                for fb in dict.fromkeys((opt.clue_fallbacks, False)):
                    est = estimate_cost(opt.clue_model, n_in, mt, fallbacks=fb)
                    if self.budget.allows(est):
                        choice = (n, mt, fb, est)
                        break
                if choice:
                    break
            if choice:
                break
        if choice is None:
            need = estimate_cost(opt.clue_model, n_full - sum(itoks[lo:]),
                                 min(opt.clue_max_tokens, opt.clue_min_max_tokens))
            return None, "degraded", (f"clue extraction skipped: budget (needs about ${need:.2f}, "
                                      f"${self.budget.remaining:.4f} left of ${self.budget.cap_usd:.2f})"), []
        n, mt, fb, est = choice
        if (n, mt, fb) != (len(imgs), opt.clue_max_tokens, opt.clue_fallbacks):
            self.warnings.append(f"clue request reduced to fit the budget: {n}/{len(imgs)} frames, "
                                 f"max_tokens {mt}, fallbacks {'on' if fb else 'off'}")
        req = full if n == len(imgs) else {**full, "messages": [{"role": "user", "content": content_for(n)}]}
        req["max_tokens"] = mt
        sent = [SentFrame(frame_id=f.id, t_s=f.t_s, width=w, height=h) for f, _, w, h in imgs[:n]]
        try:
            resp, rec = self.call("clues", req, estimate=est, fallbacks=fb)
        except LlmError as e:
            return None, "degraded", f"clue extraction failed: {e}", sent
        if rec.stop_reason == "refusal":
            return None, "degraded", ("clue extraction declined by Claude (refusal, category "
                                      f"{_refusal_category(resp)}); no visual clues"), sent
        text = response_text(resp)
        try:
            report, vwarn = validate_clue_report(json.loads(text), known_frames)
        except (ValueError, json.JSONDecodeError) as e:
            why = "output truncated at max_tokens" if rec.stop_reason == "max_tokens" else "invalid JSON"
            return None, "degraded", f"clue extraction failed: {why} ({type(e).__name__})", sent
        self.warnings += vwarn
        if rec.stop_reason == "max_tokens":
            self.warnings.append("clue output hit max_tokens but parsed")
        return report, "ok", f"{len(report.clues)} clues", sent


# --------------------------------------------------------------------------- selection

def ocr_prior(text: str | None) -> float:
    """Fallback informativeness for untriaged frames: frames with readable text first."""
    n = len((text or "").strip())
    return 0.15 + 0.35 * min(1.0, n / 40.0)


def select_frames(frames: Sequence[Frame], triage: dict[str, FrameTriageItem], ocr_by_id: dict[str, str],
                  k: int, min_hamming: int = 10) -> list[Frame]:
    """Top-k by triage score (OCR prior when untriaged), greedy for diversity: skip near-duplicates
    (dHash Hamming <= min_hamming) and sky-only frames, small bonus for flags not yet covered."""
    def score(f: Frame) -> float:
        t = triage.get(f.id)
        return t.score if t is not None else ocr_prior(ocr_by_id.get(f.id))

    pool = [f for f in frames if not (f.id in triage and "sky_only" in triage[f.id].flags)] or list(frames)
    chosen: list[Frame] = []
    covered: set[str] = set()
    remaining = sorted(pool, key=lambda f: -score(f))
    while remaining and len(chosen) < k:
        def gain(f: Frame, cov: frozenset[str] = frozenset(covered)) -> float:
            fl = set(triage[f.id].flags) if f.id in triage else set()
            return score(f) + 0.05 * len(fl - cov)

        best = max(remaining, key=gain)
        remaining.remove(best)
        h = int(best.dhash, 16)
        if any(hamming(h, int(c.dhash, 16)) <= min_hamming for c in chosen):
            continue
        chosen.append(best)
        if best.id in triage:
            covered |= set(triage[best.id].flags)
    if len(chosen) < k:  # fill with the best of the rest (near-duplicates allowed, exact ones not)
        hashes = {c.dhash for c in chosen}
        for f in sorted(pool, key=lambda f: -score(f)):
            if len(chosen) >= k:
                break
            if f not in chosen and f.dhash not in hashes:
                chosen.append(f)
                hashes.add(f.dhash)
    return sorted(chosen, key=lambda f: f.t_s)


# --------------------------------------------------------------------------- entry point

def run_visual_llm(
    frames: Sequence[Frame],
    ocr: Sequence[FrameOcr],
    *,
    out_dir: str | Path,
    budget_usd: float,
    consent: bool,
    settings: Any = None,
    client: Any = None,
    options: LlmOptions | None = None,
    video_path: str | Path | None = None,
    bbox: Sequence[float] | None = None,
    tz: str | None = None,
    date: str | None = None,
    notes: str | None = None,
    check_cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> VisualClues:
    """Triage (Haiku) -> diverse top-k -> clue extraction (Opus). Writes `out_dir/clues.json`.

    Never sends anything unless `consent` is true and either `settings` has an API key or a `client`
    is injected (tests). Never raises for API problems: they become status "degraded" with a message.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    opt = options or LlmOptions.from_settings(settings)
    vc = VisualClues(status="skipped", triage_model=opt.triage_model, clue_model=opt.clue_model,
                     budget_usd=budget_usd)
    prog = progress or (lambda f, m: None)

    def finish(v: VisualClues) -> VisualClues:
        (out / "clues.json").write_text(v.model_dump_json(indent=1))
        log.info("visual LLM %s: %s; cost $%.4f of $%.2f over %d calls", v.status, v.message, v.cost_usd,
                 v.budget_usd, len(v.usage))
        return v

    if not consent:
        vc.message = "Claude steps skipped: no visual consent for this job (no frames were sent)"
        return finish(vc)
    if client is None and not getattr(settings, "llm_enabled", False):
        vc.message = "Claude steps skipped: ANTHROPIC_API_KEY not set (no frames were sent)"
        return finish(vc)
    if not frames:
        vc.message = "Claude steps skipped: no frames"
        return finish(vc)
    if budget_usd <= 0:
        vc.message = "Claude steps skipped: LLM budget is $0"
        return finish(vc)

    if client is None:
        client = make_client(settings, opt)
    budget = BudgetGuard(budget_usd)
    cv = ClaudeVision(client, opt, budget, check_cancel)
    ocr_by_id = {o.frame_id: o.text for o in ocr}
    for o in ocr:
        if found := prompts.injection_phrases(o.text):
            cv.warnings.append(f"{o.frame_id}: instruction-like on-screen text treated as data: {found[:3]}")

    prog(0.1, f"Claude triage of {len(frames)} frames")
    triage = cv.triage(frames, ocr_by_id)
    top = select_frames(frames, triage, ocr_by_id, opt.top_k)
    prog(0.6, f"Claude clue extraction on {len(top)} frames")

    sel: list[tuple[Frame, Path]] = []
    for f in top:
        p = Path(f.path)
        if video_path is not None and opt.clue_long_edge > 1280:
            p = hires_frame(video_path, f.t_s, out / "frames_hires" / f"{f.id}.jpg", opt.clue_long_edge) or p
        sel.append((f, p))
    known = {f.id: f.t_s for f in frames}
    report, status, msg, sent = cv.extract_clues(sel, ocr_text_block(ocr, opt.ocr_chars),
                                                 prompts.region_text(bbox, tz, date, notes), known)
    for s in sent:
        t = triage.get(s.frame_id)
        s.triaged = t is not None
        s.score = t.score if t else round(ocr_prior(ocr_by_id.get(s.frame_id)), 3)
        s.flags = list(t.flags) if t else []
    if status == "ok" and any(w.startswith("triage stopped") for w in cv.warnings):
        status, msg = "degraded", msg + " (triage partial)"
    vc = vc.model_copy(update=dict(
        status=status, message=msg, frames_sent=sent, report=report, cost_usd=round(budget.spent_usd, 6),
        usage=budget.records, warnings=cv.warnings,
    ))
    prog(1.0, msg)
    return finish(vc)


def to_module_result(vc: VisualClues, clues_name: str = "clues.json") -> ModuleResult:
    """ModuleResult for the visual module. Layers are added by WP16 from clues.json."""
    return ModuleResult(
        module="visual", status=vc.status, message=vc.message, cost_usd=vc.cost_usd,
        artifacts={"clues": clues_name},
        provenance={
            "triage_model": vc.triage_model, "clue_model": vc.clue_model, "budget_usd": vc.budget_usd,
            "n_calls": len(vc.usage), "n_frames_sent": len(vc.frames_sent),
            "n_clues": len(vc.report.clues) if vc.report else 0,
            "top_countries": dict(list(vc.country_scores().items())[:5]),
            "warnings": vc.warnings[:50], "pricing_source": PRICING_SOURCE,
            "disclosure": "Selected frames and OCR text were sent to the Claude API (Anthropic)."
            if vc.usage else "Nothing was sent to the Claude API.",
        },
    )
