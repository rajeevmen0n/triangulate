"""Real pipeline stages (WP25): the glue between the module packages and the WP18 job runner.

`register_real_stages()` replaces the WP18 stubs (CLI `run`/`serve` call it); tests that need stubs call
`pipeline.reset_registry()`. Test seam: `SERVICES` (track-provider factory, ground-height function).

Data flow (each stage writes files into its cache dir; later stages read them via `ctx.out_dirs`)::

    ingest      decode 4 kHz mono PCM (never container tags)          -> audio_4k.f32
    metadata    WP13 hints (skipped if disabled)                       -> metadata.json
    spectrogram WP05 whitened SNR map                                  -> spec.npz
    events      WP08 detection, or JobParams.events when supplied      -> events.json, spectrogram.png/.json
    adsb        provider fetch + 1-s ENU tracks (jets removed)         -> tracks.parquet, flights.json
    dem         pre-fetch Copernicus tiles for the bbox (best effort)
    cpa, match  no-ops: performed inside `score` (acoustic_search is one call)
    score       WP09-WP11 gates, offset match, silence, scoring        -> layers/acoustic.npz, acoustic.json
    visual      WP15 (frames/OCR local; Claude only with consent+key)  -> clues.json, ocr.json
    fusion      acoustic + visual + GPS layers -> posterior, candidates; late re-rank (time prior, sun)
                                                                       -> posterior.*, candidates.geojson, fusion.json
    render      heatmap + results.json                                 -> heatmap.png/.json, results.json

Metadata / OCR time hints never restrict the search window. They become a mixture prior on T_s
(WP13 `time_components`) applied as a late soft re-rank of the candidates (alpha = GPS_ALPHA).

results.json schema ("triangulate.results/1")::

    {
     "schema": "triangulate.results/1",
     "status": "ok" | "degraded" | "no_result",
     "candidates": [ {rank, lat, lon, prob_mass, mass_radius_m, hpd_level, log_density,
                      margin_to_runner_up (nats), t_s (ISO UTC, recording start), flights [icao],
                      contributions {module: weighted ln-lik}, notes [str], pinned} ... ],
     "best": {lat, lon, t_s, flights, margin_nats} | null,
     "margin": {nats: float|null, acoustic_z: float|null},
     "T_s": {best: ISO|null, window: [ISO, ISO], prior_components: int},
     "flights": {icao: {reg, typecode, category}},         # info for the icao ids in candidates
     "modules": {acoustic|visual|metadata: {status, message, weight, ...}},
     "module_status": {module: status},
     "events": {source: "detected"|"user", n_total, n_used, ids_used, duration_s},
     "hpd": [{mass, area_km2, ...}],
     "warnings": [str],  # single-event, low margin, degraded/skipped modules, ...
     "attribution": [str], "cost_usd": float, "notes": [str]
    }
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..core.geo import bbox_expand_m, haversine_m
from ..core.timeutil import infer_tz, local_window_to_utc
from ..core.types import Event, GridSpec, ModuleResult
from ..log import get_logger
from .pipeline import StageContext, register_stage

log = get_logger("jobs.stages")

RESULTS_SCHEMA = "triangulate.results/1"
LOW_MARGIN_NATS = 0.5
ADSB_ATTRIBUTION = "Flight data © adsb.lol contributors (ODbL 1.0)"
DEM_ATTRIBUTION = (
    "Contains Copernicus DEM (GLO-30/GLO-90) © DLR e.V. and © Airbus Defence and Space GmbH, "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved"
)
PIPELINE_VERSION = "wp25-1"


@dataclass
class Services:
    """Injection points for tests (no network)."""

    track_provider: Callable[[], Any] | None = None  # -> TrackProvider
    ground_hae: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None  # (lat, lon) -> HAE m


SERVICES = Services()


# ------------------------------------------------------------------------------------ helpers

def _iso(t: float | datetime | None) -> str | None:
    if t is None:
        return None
    if not isinstance(t, datetime):
        t = datetime.fromtimestamp(float(t), UTC)
    return t.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _tz(ctx: StageContext) -> str:
    c = ctx.constraints
    if c.tz:
        return c.tz
    s, w, n, e = c.bbox
    return infer_tz((s + n) / 2, (w + e) / 2)


def ts_window(ctx: StageContext) -> tuple[datetime, datetime]:
    """Interval for T_s (UTC of audio sample 0) from the USER's constraints only."""
    c = ctx.constraints
    if c.known_audio_time is not None:
        k = c.known_audio_time
        centre = k.utc - timedelta(seconds=k.audio_pos_s)
        return centre - timedelta(seconds=3 * k.sigma_s), centre + timedelta(seconds=3 * k.sigma_s)
    tz = _tz(ctx)
    t0, _ = local_window_to_utc(c.date_start, tz, c.window_start, c.window_end)
    _, t1 = local_window_to_utc(c.date_end or c.date_start, tz, c.window_start, c.window_end)
    return t0, t1


def _grid(ctx: StageContext) -> GridSpec:
    return GridSpec.for_bbox(ctx.constraints.bbox)


def _load_spec(ctx: StageContext):
    from ..acoustic.spectrogram import Spectrogram

    with np.load(ctx.out_dirs["spectrogram"] / "spec.npz") as z:
        return Spectrogram(t=z["t"], f=z["f"], Z=z["Z"], db=z["db"])


def _load_events(ctx: StageContext) -> list[Event]:
    raw = json.loads((ctx.out_dirs["events"] / "events.json").read_text())
    return [Event.model_validate(e) for e in raw]


def _metadata_enabled(ctx: StageContext) -> bool:
    return ctx.params.use_metadata and os.environ.get("TRIANGULATE_SKIP_METADATA", "") not in ("1", "true")


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------------ stages

class _Base:
    name = ""
    depends_on: tuple[str, ...] = ()

    def params(self, ctx: StageContext) -> Any:
        return {"v": PIPELINE_VERSION}


class IngestStage(_Base):
    name = "ingest"

    def run(self, ctx, out_dir, progress):
        from ..ingest.media import SAMPLE_RATE, decode_audio

        progress(self.name, 0.1, "decoding audio")
        x = decode_audio(ctx.input_path, SAMPLE_RATE)
        x.astype("<f4").tofile(out_dir / "audio_4k.f32")
        return {"sample_rate": SAMPLE_RATE, "n_samples": int(len(x)), "duration_s": len(x) / SAMPLE_RATE}


class MetadataStage(_Base):
    name = "metadata"
    depends_on = ("ingest",)

    def params(self, ctx):
        return {"v": PIPELINE_VERSION, "enabled": _metadata_enabled(ctx)}

    def run(self, ctx, out_dir, progress):
        if not _metadata_enabled(ctx):
            return {"skipped": "metadata reading disabled for this job"}
        from ..ingest.metadata import extract_metadata

        rep = extract_metadata(ctx.input_path)
        (out_dir / "metadata.json").write_text(rep.model_dump_json(indent=1))
        return {"has_hints": rep.has_hints, "n_gps": len(rep.gps), "n_times": len(rep.times),
                "reencode_markers": rep.reencode_markers, "errors": rep.errors}


class SpectrogramStage(_Base):
    name = "spectrogram"
    depends_on = ("ingest",)

    def run(self, ctx, out_dir, progress):
        from ..acoustic.spectrogram import SR, compute_spectrogram

        progress(self.name, 0.1, "STFT and whitening")
        x = np.fromfile(ctx.out_dirs["ingest"] / "audio_4k.f32", dtype="<f4")
        spec = compute_spectrogram(x, SR)
        np.savez_compressed(out_dir / "spec.npz", t=spec.t, f=spec.f, Z=spec.Z, db=spec.db)
        return {"n_frames": int(len(spec.t)), "duration_s": float(len(x) / SR)}


class EventsStage(_Base):
    name = "events"
    depends_on = ("spectrogram",)

    def params(self, ctx):
        return {"v": PIPELINE_VERSION, "user_events": ctx.params.events is not None}

    def run(self, ctx, out_dir, progress):
        from ..acoustic.detect import find_events
        from ..render.spectrogram_png import render_spectrogram_window

        spec = _load_spec(ctx)
        if ctx.params.events is not None:
            events = [Event.model_validate(e) for e in ctx.params.events]
            source = "user"
        else:
            events = find_events(spec, progress=lambda f, m="": progress(self.name, 0.05 + 0.85 * f, m)).events
            source = "detected"
        ctx.check_cancel()
        (out_dir / "events.json").write_text(json.dumps([e.model_dump(mode="json") for e in events], indent=1))
        render_spectrogram_window(spec, out_dir / "spectrogram.png", events=events, width_px=1600,
                                  height_px=520)
        used = [e for e in events if e.selected]
        return {"source": source, "n_total": len(events), "n_used": len(used), "ids_used": [e.id for e in used],
                "duration_s": float(spec.t[-1]) if len(spec.t) else 0.0}


class AdsbStage(_Base):
    name = "adsb"
    depends_on = ("events",)

    def params(self, ctx):
        c = ctx.constraints
        return {"v": PIPELINE_VERSION, "provider": "injected" if SERVICES.track_provider else "adsblol",
                "excl_heli": ctx.settings.exclude_helicopters, "tz": c.tz}

    def run(self, ctx, out_dir, progress):
        from ..adsb.aircraft_types import load_types
        from ..adsb.provider import get_provider
        from ..adsb.tracks import prepare_trackset
        from ..core.types import TrackQuery

        if ctx.results["events"]["n_used"] == 0:
            return {"skipped": "no selected events"}
        t_lo, t_hi = ts_window(ctx)
        dur = ctx.results["events"]["duration_s"]
        t0, t1 = t_lo - timedelta(seconds=300), t_hi + timedelta(seconds=dur + 300)
        prov = SERVICES.track_provider() if SERVICES.track_provider else get_provider("adsblol")
        q = TrackQuery(bbox=bbox_expand_m(ctx.constraints.bbox, 15_000), t0=t0, t1=t1)
        ts = prov.fetch(q, lambda s, f, m="": progress(self.name, 0.05 + 0.85 * f, m))
        ctx.check_cancel()
        grid = _grid(ctx)
        prep, cls = prepare_trackset(ts, (grid.lat0, grid.lon0), load_types(), ctx.settings.exclude_helicopters)
        prep.write_parquet(out_dir / "tracks.parquet")
        info: dict[str, dict[str, Any]] = {}
        for r in ts.aircraft.iter_rows(named=True):
            info[r["icao"]] = {"reg": r["reg"], "typecode": r["typecode"], "category": r["category"]}
        for r in cls.iter_rows(named=True):
            d = info.setdefault(r["icao"], {})
            d.setdefault("typecode", r["typecode"])
            d.setdefault("category", r["category"])
        (out_dir / "flights.json").write_text(json.dumps(info))
        return {"provider": ts.provider or prov.name, "licence": ts.licence,
                "n_aircraft_raw": int(ts.tracks["icao"].n_unique()) if ts.tracks.height else 0,
                "n_aircraft_kept": int(prep["icao"].n_unique()) if prep.height else 0,
                "fetch_window": [_iso(t0), _iso(t1)], "t_s_window": [_iso(t_lo), _iso(t_hi)]}


class DemStage(_Base):
    name = "dem"

    def params(self, ctx):
        return {"v": PIPELINE_VERSION, "injected": SERVICES.ground_hae is not None,
                "bbox": list(ctx.constraints.bbox)}

    def run(self, ctx, out_dir, progress):
        if SERVICES.ground_hae is not None:
            return {"skipped": "ground height function injected"}
        from ..dem.provider import elevation

        s, w, n, e = bbox_expand_m(ctx.constraints.bbox, 15_000)
        la, lo = np.meshgrid(np.linspace(s, n, 9), np.linspace(w, e, 9))
        try:
            for i, res in enumerate((90, 30)):
                z = elevation(la, lo, res)
                progress(self.name, (i + 1) / 2, f"DEM {res} m tiles ready")
        except Exception as ex:  # noqa: BLE001 - the search fetches lazily and reports its own failure
            return {"status": "degraded", "message": f"DEM prefetch failed: {ex}"}
        return {"status": "ok", "elev_min_m": float(np.min(z)), "elev_max_m": float(np.max(z)),
                "attribution": DEM_ATTRIBUTION}


class _Merged(_Base):
    def __init__(self, name: str, deps: tuple[str, ...]):
        self.name, self.depends_on = name, deps

    def run(self, ctx, out_dir, progress):
        return {"note": "performed inside the score stage (acoustic_search)"}


class ScoreStage(_Base):
    name = "score"
    depends_on = ("match",)

    def params(self, ctx):
        c = ctx.constraints
        return {"v": PIPELINE_VERSION, "h_obs": c.h_obs_m, "temp": c.temperature_c,
                "injected_dem": SERVICES.ground_hae is not None}

    def run(self, ctx, out_dir, progress):
        from ..acoustic.physics import C_DEFAULT
        from ..acoustic.search import SearchConfig, acoustic_search

        grid = _grid(ctx)
        events = _load_events(ctx)
        sel = [e for e in events if e.selected]
        tr_path = ctx.out_dirs["adsb"] / "tracks.parquet"
        if not sel or not tr_path.exists():
            return ModuleResult(module="acoustic", status="skipped",
                                message="no selected events" if not sel else "no tracks").model_dump(mode="json")
        tracks = pl.read_parquet(tr_path)
        t_lo, t_hi = ts_window(ctx)
        c = ctx.constraints
        kw: dict[str, Any] = {"h_obs": c.h_obs_m}
        if c.temperature_c is not None:
            kw["c"] = 331.3 * math.sqrt(1 + c.temperature_c / 273.15)
        else:
            kw["c"] = C_DEFAULT
        if len(sel) == 1:  # WP10: single-event recordings need the full silence scope
            kw.update(silence_scope="full", exempt="aircraft")
        res = acoustic_search(
            _load_spec(ctx), events, tracks, grid, (grid.lat0, grid.lon0), (t_lo.timestamp(), t_hi.timestamp()),
            ground_hae=SERVICES.ground_hae, config=SearchConfig(**kw),
            progress=lambda s, f, m="": progress(self.name, f, m))
        if res.ok:
            mr = res.module_result(out_dir, [ADSB_ATTRIBUTION, DEM_ATTRIBUTION])
        else:
            mr = ModuleResult(module="acoustic", status="degraded",
                              message="no flight/position hypothesis survived the gates",
                              provenance={"attribution": [ADSB_ATTRIBUTION]})
            if res.layer is not None:
                mr.layers = [res.layer.to_ref(out_dir / "layers" / "acoustic.npz", relative_to=out_dir)]
        out = mr.model_dump(mode="json")
        out["n_events"] = len(sel)
        out["margin_z"] = res.margin_z
        out["margin_S"] = res.margin_S
        out["top"] = res.summary(top=5)["tuples"] if res.ok else []
        return out


class FusionStage(_Base):
    name = "fusion"
    depends_on = ("score", "visual", "metadata")

    def params(self, ctx):
        return {"v": PIPELINE_VERSION, "weights": ctx.params.weights, "trust_gps": ctx.params.trust_gps}

    # -- evidence pieces
    @staticmethod
    def _metadata(ctx: StageContext):
        p = ctx.out_dirs["metadata"] / "metadata.json"
        if not p.exists() or "skipped" in ctx.results.get("metadata", {}):
            return None
        from ..ingest.metadata import MetadataReport

        return MetadataReport.model_validate_json(p.read_text())

    @staticmethod
    def time_components(ctx: StageContext, report, tz: str):
        from ..ingest.metadata import MetadataReport, TimeHint, time_components

        comps = []
        dur = ctx.results["events"].get("duration_s")
        if report is not None:
            comps += time_components(report, tz, duration_s=dur)
        hint = (ctx.results.get("visual") or {}).get("timestamp_hint")
        if hint:
            th = TimeHint(field="video timestamp OCR", naive=datetime.fromisoformat(hint["start_naive"]),
                          source="ocr", trust="low", reason="burned-in timestamp read by OCR")
            comps += time_components(MetadataReport(path="", times=[th]), tz, duration_s=dur)
        tot = sum(c.weight for c in comps)  # several sources: share the unit mass
        for c in comps:
            c.weight /= tot
        return comps

    def run(self, ctx, out_dir, progress):
        from ..fusion.candidates import rerank
        from ..fusion.combine import gps_mixture_layer, layers_from_results
        from ..fusion.export import run_fusion, write_outputs
        from ..ingest.metadata import GPS_ALPHA
        from ..visual.layers import build_visual_layers, sun_clue_from_clues
        from ..visual.schemas import VisualClues
        from ..visual.sun import sun_rerank

        c = ctx.constraints
        grid = _grid(ctx)
        w = {"acoustic": 1.0, "visual": 0.5, "metadata": 1.0, **ctx.params.weights}
        warns: list[str] = []
        layers, mres = [], []
        # acoustic
        ac = ModuleResult.model_validate({k: v for k, v in ctx.results["score"].items()
                                          if k in ModuleResult.model_fields})
        lay, _ = layers_from_results([ac], ctx.out_dirs["score"])
        layers += lay
        mres.append(ac.model_copy(update={"layers": []}))
        # visual
        vis = ctx.results.get("visual") or {}
        vc = None
        clues_p = ctx.out_dirs["visual"] / "clues.json"
        if "skipped" not in vis and clues_p.exists():
            vc = VisualClues.model_validate_json(clues_p.read_text())
            progress(self.name, 0.1, "visual clue layers")
            vmr = build_visual_layers(vc, c.bbox, grid, out_dir / "visual", settings=ctx.settings,
                                      check_cancel=ctx.check_cancel)
            lay, _ = layers_from_results([vmr], out_dir / "visual")
            layers += lay
            mres.append(vmr.model_copy(update={"layers": []}))
        else:
            mres.append(ModuleResult(module="visual", status="skipped", message=vis.get("skipped", "not run")))
        # metadata
        report = self._metadata(ctx)
        pinned: list[tuple[float, float, str]] = []
        mstat, mmsg = "skipped", ctx.results["metadata"].get("skipped", "")
        if report is not None:
            mstat, mmsg = "ok", ""
            gps = report.best_gps
            if gps is not None:
                inside = c.bbox[0] <= gps.lat <= c.bbox[2] and c.bbox[1] <= gps.lon <= c.bbox[3]
                if inside:
                    layers.append(gps_mixture_layer(grid, gps.lat, gps.lon, gps.accuracy_m, c.bbox,
                                                    alpha=GPS_ALPHA))
                    if ctx.params.trust_gps:
                        pinned.append((gps.lat, gps.lon, "GPS from file metadata (trusted by user)"))
                else:
                    warns.append("Metadata GPS lies outside the search region and was ignored.")
            elif not report.times:
                mmsg = "no location or time hints in the file"
        mres.append(ModuleResult(module="metadata", status=mstat, message=mmsg))  # type: ignore[arg-type]

        progress(self.name, 0.4, "fusing layers")
        fr = run_fusion(grid, layers=layers, module_results=mres, domain=c.bbox, weights=w, pinned=pinned)
        cands = fr.candidates
        # late evidence: T_s prior from metadata/OCR (soft), then sun
        t_lo, t_hi = ts_window(ctx)
        comps = self.time_components(ctx, report, _tz(ctx))
        if comps and cands:
            span = max((t_hi - t_lo).total_seconds(), 1.0)
            ell = []
            for cd in cands:
                if cd.t_s is None:
                    ell.append(None)
                    continue
                dens = sum(k.weight * math.exp(-0.5 * ((cd.t_s - k.t_s).total_seconds() / k.sigma_s) ** 2)
                           / (k.sigma_s * math.sqrt(2 * math.pi)) for k in comps)
                ell.append(math.log(GPS_ALPHA * dens * span + (1 - GPS_ALPHA)))
            cands = rerank(cands, "time_prior", ell, weight=w["metadata"])
        sun_used = False
        if vc is not None and cands:
            clue = sun_clue_from_clues(vc, time_window=(t_lo, t_hi))
            if clue is not None:
                cands = sun_rerank(cands, clue, weight=w["visual"])
                sun_used = True
        fr.candidates = cands
        fr.notes.append(f"time prior components: {len(comps)}" if comps else "no usable time hints")
        write_outputs(fr, out_dir)
        return {"n_candidates": len(cands), "module_status": fr.module_status, "sun_rerank": sun_used,
                "time_prior_components": len(comps), "warnings": warns, "weights": w}


class RenderStage(_Base):
    name = "render"
    depends_on = ("fusion",)

    def params(self, ctx):
        return {"v": PIPELINE_VERSION, "weights": ctx.params.weights}

    def run(self, ctx, out_dir, progress):
        from ..fusion.combine import Posterior
        from ..render.heatmap_png import render_posterior_heatmap

        fd = ctx.out_dirs["fusion"]
        post = Posterior.load(fd / "posterior.npz")
        render_posterior_heatmap(post, out_dir / "heatmap.png", bounds=ctx.constraints.bbox)
        results = build_results(ctx, _read_json(fd / "fusion.json"))
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
        return {"status": results["status"], "n_candidates": len(results["candidates"])}


def build_results(ctx: StageContext, fusion: dict[str, Any]) -> dict[str, Any]:
    """Assemble results.json (schema in the module docstring)."""
    R = ctx.results
    warns = list(R["fusion"].get("warnings", []))
    sc, ev = R["score"], R["events"]
    flights_info = {}
    fp = ctx.out_dirs["adsb"] / "flights.json"
    if fp.exists():
        flights_info = _read_json(fp)
    cands = []
    for c in fusion["candidates"]:
        cands.append({
            "rank": c["rank"], "lat": c["lat"], "lon": c["lon"], "prob_mass": c.get("prob_mass"),
            "mass_radius_m": c.get("mass_radius_m"), "hpd_level": c.get("hpd_level"),
            "log_density": c.get("log_density"), "margin_to_runner_up": c.get("margin_to_runner_up"),
            "t_s": c.get("t_s"), "flights": c.get("flights", []), "contributions": c.get("contributions", {}),
            "notes": c.get("notes", []), "pinned": c.get("pinned", False),
        })
    ranked = [c for c in cands if not c["pinned"]]
    top = ranked[0] if ranked else None
    margin = top["margin_to_runner_up"] if top else None
    used_icao = {f for c in cands for f in c["flights"]}
    modules: dict[str, Any] = {}
    status = R["fusion"]["module_status"]
    for m in ("acoustic", "visual", "metadata"):
        src = R.get({"acoustic": "score", "visual": "visual", "metadata": "metadata"}[m]) or {}
        modules[m] = {"status": status.get(m, "absent"), "weight": R["fusion"]["weights"].get(m),
                      "message": src.get("message") or src.get("skipped") or ""}
    modules["acoustic"].update(margin_z=sc.get("margin_z"), n_events=sc.get("n_events"))
    modules["visual"]["cost_usd"] = (R.get("visual") or {}).get("cost_usd", 0.0)
    for m, mod in modules.items():
        if mod["status"] in ("skipped", "degraded", "failed", "absent"):
            warns.append(f"{m} module {mod['status']}" + (f": {mod['message']}" if mod["message"] else ""))
    if ev["n_used"] == 1:
        warns.append("Only one aircraft event was used; single-event results are much less certain.")
    if ev["n_used"] == 0:
        warns.append("No usable aircraft event was found; the result is uninformative.")
    if margin is not None and margin < LOW_MARGIN_NATS:
        warns.append("Low margin: the top two candidates are nearly equally likely.")
    if sc.get("status") == "degraded":
        warns.append("Acoustic search found no consistent flight match; acoustic evidence is flat.")
    notes = [n for n in fusion.get("notes", []) if n not in warns]
    attribution = list(dict.fromkeys(fusion.get("attribution", []) + [ADSB_ATTRIBUTION]))
    ok = top is not None and modules["acoustic"]["status"] == "ok"
    out_status = "no_result" if not cands else ("ok" if ok else "degraded")
    tw = (R.get("adsb") or {}).get("t_s_window") or [_iso(x) for x in ts_window(ctx)]
    return {
        "schema": RESULTS_SCHEMA, "status": out_status, "candidates": cands,
        "best": {"lat": top["lat"], "lon": top["lon"], "t_s": top["t_s"], "flights": top["flights"],
                 "margin_nats": margin} if top else None,
        "margin": {"nats": margin, "acoustic_z": sc.get("margin_z")},
        "T_s": {"best": top["t_s"] if top else None, "window": tw,
                "prior_components": R["fusion"].get("time_prior_components", 0)},
        "flights": {k: v for k, v in flights_info.items() if k in used_icao},
        "modules": modules, "module_status": status,
        "events": {"source": ev["source"], "n_total": ev["n_total"], "n_used": ev["n_used"],
                   "ids_used": ev["ids_used"], "duration_s": ev["duration_s"]},
        "hpd": fusion.get("hpd", []), "warnings": warns, "notes": notes, "attribution": attribution,
        "cost_usd": modules["visual"]["cost_usd"],
    }


def distance_to(results: dict[str, Any], lat: float, lon: float) -> float | None:
    """Distance (m) from the top non-pinned candidate to a point (reporting helper)."""
    b = results.get("best")
    return None if not b else float(haversine_m(b["lat"], b["lon"], lat, lon))


def real_stages() -> list:
    from ..visual.stage import VisualStage

    return [IngestStage(), MetadataStage(), SpectrogramStage(), EventsStage(), AdsbStage(), DemStage(),
            _Merged("cpa", ("events", "adsb", "dem")), _Merged("match", ("cpa",)), ScoreStage(),
            VisualStage(), FusionStage(), RenderStage()]


def register_real_stages() -> None:
    for st in real_stages():
        register_stage(st)


def validate_results(d: dict[str, Any]) -> None:
    """Raise AssertionError unless `d` follows the results.json schema (used by tests and the UI contract)."""
    assert d.get("schema") == RESULTS_SCHEMA, "schema"
    assert d["status"] in ("ok", "degraded", "no_result")
    for k, t in (("candidates", list), ("flights", dict), ("modules", dict), ("module_status", dict),
                 ("events", dict), ("hpd", list), ("warnings", list), ("attribution", list), ("notes", list),
                 ("margin", dict), ("T_s", dict)):
        assert isinstance(d.get(k), t), k
    for c in d["candidates"]:
        for k in ("rank", "lat", "lon", "prob_mass", "margin_to_runner_up", "t_s", "flights", "contributions"):
            assert k in c, f"candidate.{k}"
        assert -90 <= c["lat"] <= 90 and -180 <= c["lon"] <= 180
        assert isinstance(c["flights"], list) and isinstance(c["contributions"], dict)
    for k in ("acoustic", "visual", "metadata"):
        assert {"status", "weight", "message"} <= set(d["modules"][k]), k
    assert {"nats", "acoustic_z"} <= set(d["margin"])
    assert {"best", "window"} <= set(d["T_s"])
    assert {"source", "n_total", "n_used", "ids_used"} <= set(d["events"])
