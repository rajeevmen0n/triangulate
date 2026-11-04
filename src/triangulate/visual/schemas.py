"""Pydantic contracts for the Claude visual steps (plan 3.6 steps 3-4).

* `FrameTriage` is what the triage model (Haiku) returns per batch of frames.
* `ClueReport` is what the clue model (Opus) returns.
* `VisualClues` is what this module writes to `clues.json` (the WP16 input): the validated report,
  the frames that were sent (with timestamps), cost, usage and warnings.

Model output is untrusted. Every string goes through `_clean` (NFKC, control and bidi characters
removed, whitespace collapsed, truncated), every probability is clamped to [0, 1], and list fields
are validated item by item by `validate_clue_report` / `validate_triage`, so one bad item is dropped
instead of failing the whole response. Nothing in these objects is ever executed; OSM hints are
structured filters that WP16 turns into Overpass QL through its own whitelist.

The JSON schema the API sees is `api_schema(Model)`; fields marked `SkipJsonSchema` (ids, timestamps)
are filled in locally after validation and never requested from the model.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from pydantic.json_schema import SkipJsonSchema

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- sanitising primitives

_BIDI = dict.fromkeys(map(ord, "​‌‍‎‏‪‫‬‭‮"
                              "⁦⁧⁨⁩﻿"), None)


def clean_text(v: Any, n: int) -> str:
    """NFKC, drop control/format/bidi characters, collapse whitespace, truncate to n characters."""
    s = unicodedata.normalize("NFKC", "" if v is None else str(v)).translate(_BIDI)
    s = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]


def _clean(n: int):
    return BeforeValidator(lambda v: clean_text(v, n))


def _clamp01(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x != x:  # NaN
        return 0.0
    return min(1.0, max(0.0, x))


Str40 = Annotated[str, _clean(40)]
Str80 = Annotated[str, _clean(80)]
Str120 = Annotated[str, _clean(120)]
Str160 = Annotated[str, _clean(160)]
Str200 = Annotated[str, _clean(200)]
Str500 = Annotated[str, _clean(500)]
Str800 = Annotated[str, _clean(800)]
Prob = Annotated[float, BeforeValidator(_clamp01), Field(description="0 to 1")]
FrameId = Annotated[str, _clean(16), Field(description="frame id exactly as given, e.g. f0007")]
Iso2 = Annotated[str, BeforeValidator(lambda v: clean_text(v, 8).upper()),
                 Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 code")]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unknown keys from the model are ignored, not trusted


# --------------------------------------------------------------------------- triage (Haiku)

TriageFlag = Literal[
    "text", "signage", "plate", "architecture", "vegetation", "road_markings", "landmark",
    "sun_shadow", "terrain", "sky_only",
]


class FrameTriageItem(_M):
    frame_id: FrameId
    score: Prob = Field(description="how much this frame could help locate the place, 0 to 1")
    flags: list[TriageFlag] = Field(default_factory=list)
    note: Str200 = ""


class FrameTriage(_M):
    frames: list[FrameTriageItem]


# --------------------------------------------------------------------------- clues (Opus)

ClueType = Literal[
    "language_script",       # language or writing system of visible text
    "signage_text",          # text on a sign (street sign text goes to street_name)
    "business_brand",        # shop, chain or brand name
    "street_name",
    "phone_area_code",       # phone number format or area code (never a full personal number)
    "plate_format",          # licence plate format / jurisdiction only, never the number
    "architecture",
    "vegetation_biome",
    "road_markings_driving_side",
    "utility_poles",
    "landmark",
    "terrain",
    "sun_shadow",
    "other",
]

DrivingSide = Literal["left", "right", "unknown"]
OsmOp = Literal["=", "!=", "~", "exists"]
EntityKind = Literal["street", "business", "landmark", "place", "transit", "natural", "other"]


class NormBox(_M):
    """Region of the frame, normalised to [0, 1] (x to the right, y down)."""

    x0: Prob
    y0: Prob
    x1: Prob
    y1: Prob


class Evidence(_M):
    frame_id: FrameId
    bbox: NormBox | None = None
    t_s: SkipJsonSchema[float | None] = None  # filled locally from the frame index


class CountryP(_M):
    code: Iso2
    p: Prob


class Admin1P(_M):
    country: Iso2
    name: Str80 = Field(description="admin-1 region (state/province) name")
    p: Prob


class Implication(_M):
    countries: list[CountryP] = Field(default_factory=list, description="at most 10")
    admin1: list[Admin1P] = Field(default_factory=list, description="at most 10")
    koppen: list[Str40] = Field(default_factory=list, description="Koppen climate codes, e.g. Csa")
    driving_side: DrivingSide = "unknown"


class OsmTag(_M):
    key: Annotated[str, _clean(48), Field(description="OSM tag key, e.g. amenity, highway, name")]
    op: OsmOp = "="
    value: Str120 = ""


class OsmNear(_M):
    tags: list[OsmTag] = Field(description="1 to 3 tags of the second feature")
    distance_m: float = Field(description="max distance in metres, 1 to 2000")


class OsmHint(_M):
    """A structured OSM filter: all `tags` on one feature, optionally within `near.distance_m` of a
    feature matching `near.tags`. WP16 builds Overpass QL from this through a key whitelist."""

    tags: list[OsmTag] = Field(description="1 to 4 tags that must all match one feature")
    near: OsmNear | None = None
    note: Str160 = ""


class Clue(_M):
    id: SkipJsonSchema[str] = ""  # "c01", assigned locally
    type: ClueType
    value: Str200 = Field(description="what is observed, e.g. the sign text or 'palm trees'")
    evidence: list[Evidence] = Field(default_factory=list)
    implication: Implication = Field(default_factory=Implication)
    osm_hints: list[OsmHint] = Field(default_factory=list, description="at most 5")
    place_candidates: list[Str120] = Field(default_factory=list,
                                           description="candidate place/region names, at most 8")
    geocode_queries: list[Str160] = Field(default_factory=list,
                                          description="free-text geocoder queries, at most 5")
    confidence: Prob
    rationale: Str500 = ""


class RegionHypothesis(_M):
    name: Str120
    country: Annotated[str, BeforeValidator(lambda v: clean_text(v, 8).upper()),
                       Field(description="ISO 3166-1 alpha-2 code, or empty")] = ""
    admin1: Str80 = ""
    p: Prob


class SearchableEntity(_M):
    name: Str120
    kind: EntityKind = "other"
    query: Str160 = Field(description="geocoder query, e.g. 'Rue de la Paix, Paris'")
    frame_ids: list[FrameId] = Field(default_factory=list)
    confidence: Prob


class ClueReport(_M):
    clues: list[Clue] = Field(default_factory=list)
    region_hypotheses: list[RegionHypothesis] = Field(default_factory=list)
    searchable_entities: list[SearchableEntity] = Field(default_factory=list)
    summary: Str800 = ""


# --------------------------------------------------------------------------- module output

class SentFrame(BaseModel):
    frame_id: str
    t_s: float
    score: float | None = None
    flags: list[str] = Field(default_factory=list)
    triaged: bool = False
    width: int | None = None
    height: int | None = None


class UsageRecord(BaseModel):
    call: str  # "triage[0]", "clues"
    model: str  # requested model
    served_by: list[str] = Field(default_factory=list)  # models that actually ran (fallbacks)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    estimate_usd: float = 0.0
    cost_usd: float = 0.0
    stop_reason: str | None = None
    request_id: str | None = None


class GeoQuery(BaseModel):
    """One geocodable query for WP16 (Nominatim), with provenance."""

    query: str
    kind: str
    source: str  # clue id or "entity"
    confidence: float
    frame_ids: list[str] = Field(default_factory=list)
    t_s: list[float] = Field(default_factory=list)
    country_codes: list[str] = Field(default_factory=list)  # restrict the geocoder when non-empty


class VisualClues(BaseModel):
    """Contents of clues.json."""

    version: int = SCHEMA_VERSION
    status: Literal["ok", "skipped", "degraded", "failed"] = "ok"
    message: str = ""
    triage_model: str = ""
    clue_model: str = ""
    frames_sent: list[SentFrame] = Field(default_factory=list)
    report: ClueReport | None = None
    budget_usd: float = 0.0
    cost_usd: float = 0.0
    usage: list[UsageRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def country_scores(self) -> dict[str, float]:
        """Noisy-OR of clue country implications weighted by clue confidence, plus region hypotheses.
        A quick summary for the UI; WP16 builds layers per clue."""
        acc: dict[str, float] = {}
        if self.report is None:
            return acc
        for c in self.report.clues:
            for cp in c.implication.countries:
                q = c.confidence * cp.p
                acc[cp.code] = 1 - (1 - acc.get(cp.code, 0.0)) * (1 - q)
        for r in self.report.region_hypotheses:
            if r.country:
                acc[r.country] = 1 - (1 - acc.get(r.country, 0.0)) * (1 - r.p)
        return dict(sorted(acc.items(), key=lambda kv: -kv[1]))

    def geocode_queries(self) -> list[GeoQuery]:
        """All geocodable queries (clue queries + searchable entities), de-duplicated by text."""
        out: dict[str, GeoQuery] = {}
        if self.report is None:
            return []
        t_of = {f.frame_id: f.t_s for f in self.frames_sent}
        for c in self.report.clues:
            fids = [e.frame_id for e in c.evidence]
            cc = [cp.code for cp in sorted(c.implication.countries, key=lambda x: -x.p) if cp.p >= 0.2]
            for q in c.geocode_queries:
                if q and q.lower() not in out:
                    out[q.lower()] = GeoQuery(query=q, kind=c.type, source=c.id, confidence=c.confidence,
                                              frame_ids=fids, t_s=[t_of[f] for f in fids if f in t_of],
                                              country_codes=cc)
        for e in self.report.searchable_entities:
            if e.query and e.query.lower() not in out:
                out[e.query.lower()] = GeoQuery(query=e.query, kind=e.kind, source="entity",
                                                confidence=e.confidence, frame_ids=e.frame_ids,
                                                t_s=[t_of[f] for f in e.frame_ids if f in t_of])
        return sorted(out.values(), key=lambda g: -g.confidence)


# --------------------------------------------------------------------------- schema + validation

def api_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema in the strict form structured outputs expect (additionalProperties false; numeric
    and length constraints moved into descriptions by the SDK's own transformer)."""
    from anthropic import transform_schema

    return transform_schema(model)


_OSM_KEY = re.compile(r"^[a-z][a-z0-9_]*(:[a-z0-9_]+)*$")
_PLATE_LIKE = re.compile(r"\d")
_LIMITS = {"clues": 40, "region_hypotheses": 10, "searchable_entities": 20}


def _items(raw: Any, model: type[BaseModel], limit: int, dropped: list[str], where: str) -> list:
    out = []
    for i, it in enumerate(raw if isinstance(raw, list) else []):
        try:
            out.append(model.model_validate(it))
        except ValidationError as e:
            dropped.append(f"{where}[{i}]: {e.errors()[0].get('msg', 'invalid')}")
    if len(out) > limit:
        dropped.append(f"{where}: truncated {len(out)} -> {limit}")
    return out[:limit]


def _fix_osm(h: OsmHint) -> OsmHint | None:
    tags = [t for t in h.tags if _OSM_KEY.match(t.key)][:4]
    if not tags:
        return None
    near = h.near
    if near is not None:
        ntags = [t for t in near.tags if _OSM_KEY.match(t.key)][:3]
        dist = min(2000.0, max(1.0, float(near.distance_m)))
        near = OsmNear(tags=ntags, distance_m=dist) if ntags else None
    return OsmHint(tags=tags, near=near, note=h.note)


def _normalise(ps: list, attr: str = "p") -> None:
    s = sum(getattr(x, attr) for x in ps)
    if s > 1.0:
        for x in ps:
            setattr(x, attr, getattr(x, attr) / s)


def validate_clue_report(raw: Any, known_frames: dict[str, float]) -> tuple[ClueReport, list[str]]:
    """Validate an untrusted parsed-JSON report. Returns (report, warnings).

    Drops invalid items individually, unknown frame ids, malformed OSM keys; redacts plate numbers;
    renormalises country probabilities; assigns clue ids and frame timestamps."""
    warns: list[str] = []
    if not isinstance(raw, dict):
        raise ValueError("clue report is not a JSON object")
    clues = _items(raw.get("clues"), Clue, _LIMITS["clues"], warns, "clues")
    regions = _items(raw.get("region_hypotheses"), RegionHypothesis, _LIMITS["region_hypotheses"], warns,
                     "region_hypotheses")
    ents = _items(raw.get("searchable_entities"), SearchableEntity, _LIMITS["searchable_entities"], warns,
                  "searchable_entities")
    for k, c in enumerate(clues, 1):
        c.id = f"c{k:02d}"
        ev = [e for e in c.evidence if e.frame_id in known_frames][:10]
        if len(ev) < len(c.evidence):
            warns.append(f"{c.id}: dropped {len(c.evidence) - len(ev)} evidence refs to unknown frames")
            if not ev:
                c.confidence *= 0.5
        for e in ev:
            e.t_s = known_frames[e.frame_id]
            if e.bbox is not None and (e.bbox.x1 <= e.bbox.x0 or e.bbox.y1 <= e.bbox.y0):
                e.bbox = None
        c.evidence = ev
        imp = c.implication
        imp.countries = imp.countries[:10]
        imp.admin1 = imp.admin1[:10]
        imp.koppen = [x for x in imp.koppen if re.fullmatch(r"[A-E][A-Za-z]{0,2}", x)][:5]
        _normalise(imp.countries)
        fixed = [h2 for h in c.osm_hints if (h2 := _fix_osm(h)) is not None][:5]
        if len(fixed) < len(c.osm_hints):
            warns.append(f"{c.id}: dropped {len(c.osm_hints) - len(fixed)} malformed OSM hints")
        c.osm_hints = fixed
        c.place_candidates = [p for p in c.place_candidates if p][:8]
        c.geocode_queries = [q for q in c.geocode_queries if q][:5]
        if c.type == "plate_format" and _PLATE_LIKE.search(c.value):
            c.value = "[plate number redacted]"
            warns.append(f"{c.id}: plate number redacted")
        if c.type == "phone_area_code":
            c.value = re.sub(r"(\d{3})[\d\s().-]{4,}", r"\1 [redacted]", c.value)
    for e in ents:
        e.frame_ids = [f for f in e.frame_ids if f in known_frames][:10]
    regions = [r for r in regions if r.name]
    for r in regions:
        if not re.fullmatch(r"[A-Z]{2}", r.country or ""):
            r.country = ""
    _normalise(regions)
    report = ClueReport(clues=clues, region_hypotheses=regions,
                        searchable_entities=[e for e in ents if e.query],
                        summary=clean_text(raw.get("summary", ""), 800))
    return report, warns


def validate_triage(raw: Any, known_ids: set[str]) -> tuple[list[FrameTriageItem], list[str]]:
    warns: list[str] = []
    if not isinstance(raw, dict):
        raise ValueError("triage result is not a JSON object")
    items = _items(raw.get("frames"), FrameTriageItem, 1000, warns, "frames")
    seen: set[str] = set()
    out = []
    for it in items:
        if it.frame_id not in known_ids or it.frame_id in seen:
            warns.append(f"triage: ignored unknown or repeated frame id {it.frame_id!r}")
            continue
        seen.add(it.frame_id)
        it.flags = list(dict.fromkeys(it.flags))
        out.append(it)
    return out, warns
