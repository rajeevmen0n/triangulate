"""FastAPI app (WP19): job endpoints, SSE progress, artifact serving. Local by default (127.0.0.1)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

from .. import __version__
from ..config import Settings, get_settings
from ..jobs import store as S
from ..jobs.pipeline import JobParams, Stage, workspace
from ..jobs.retention import DAY, purge_expired
from ..jobs.runner import JobRunner, create_job, make_store

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
CHUNK = 1 << 20
_CACHES = ("adsb", "dem", "osm", "geoid", "types", "stages")
END_STATES = S.TERMINAL | {S.INTERRUPTED}

VISUAL_NOTICE = (
    "If you enable the visual module, selected video frames and OCR text are sent to the Claude API "
    "(Anthropic). No audio is sent. This needs your explicit consent for each upload. Spending is capped "
    "per job."
)
ATTRIBUTION = [
    "Flight data © adsb.lol contributors (ODbL 1.0)",
    "© OpenStreetMap contributors (ODbL 1.0)",
    "Contains Copernicus DEM (GLO-30) © DLR e.V. 2020 and © Airbus Defence and Space GmbH 2020, "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved",
    "Map: Leaflet 1.9.4 © Vladimir Agafonkin and contributors (BSD 2-Clause)",
    "Aircraft type designators: ICAO Doc 8643 (bundled subset; not a copy of third-party data)",
]
USE_NOTICE = (
    "For OSINT verification, journalism, research and checking your own recordings. Not a tool for "
    "locating private individuals. One file per job; no bulk or automated use. Results are probabilistic: "
    "read the uncertainty region, not the pin."
)


class RerunBody(BaseModel):
    """Everything optional: omitted fields are inherited from the original job."""

    constraints: dict[str, Any] | None = None  # full replacement of the job's constraints
    events: list[dict[str, Any]] | None = None  # edited events (replace auto-detection)
    visual: bool | None = None
    visual_consent: bool | None = None
    weights: dict[str, float] | None = None


def _json_field(raw: str | None, name: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        raise HTTPException(422, f"{name} is not valid JSON: {e}") from e


def _bool(v: bool | str | None) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "on", "yes")
    return bool(v)


def _params_or_422(**kw: Any) -> JobParams:
    try:
        p = JobParams(**kw)
    except ValidationError as e:
        raise HTTPException(422, detail=json.loads(e.json())) from e
    if p.visual and not p.visual_consent:
        raise HTTPException(409, "visual module requires explicit consent (visual_consent=true)")
    return p


def _du(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def create_app(
    settings: Settings | None = None,
    stages: list[Stage] | None = None,
    start_runner: bool = True,
    poll_interval: float = 0.5,
) -> FastAPI:
    s = settings or get_settings()
    s.ensure_dirs()
    store = make_store(s)
    purge_expired(s, store)  # retention on startup (WP24); queued/running jobs are never purged
    runner = JobRunner(store, s, stages)

    def check_rate_limit() -> None:
        """At most `max_jobs_per_day` new jobs per rolling 24 h (0 disables). Covers uploads and reruns."""
        if s.max_jobs_per_day > 0 and store.count_since(time.time() - DAY) >= s.max_jobs_per_day:
            raise HTTPException(
                429, f"rate limit: at most {s.max_jobs_per_day} jobs per 24 hours "
                "(TRIANGULATE_MAX_JOBS_PER_DAY)"
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_runner:
            runner.start()
        try:
            yield
        finally:
            if start_runner:
                runner.stop(timeout=2)

    app = FastAPI(title="triangulate", version=__version__, lifespan=lifespan)
    app.state.settings, app.state.store, app.state.runner = s, store, runner

    def view(job: dict[str, Any]) -> dict[str, Any]:
        ws = workspace(s, job["id"])
        files = sorted(p.name for p in ws.iterdir() if p.is_file()) if ws.is_dir() else []
        return {
            "id": job["id"], "created": job["created"], "status": job["status"], "stage": job["stage"],
            "progress": job["progress"], "message": job["message"], "error": job["error"],
            "cancel_requested": job["cancel_requested"], "has_result": job["status"] == S.DONE
            and bool(job["result_path"]) and Path(job["result_path"]).is_file(),
            "params": job["params"], "artifacts": files,
        }

    def get_job(job_id: str) -> dict[str, Any]:
        j = store.get(job_id)
        if j is None:
            raise HTTPException(404, "no such job")
        return j

    # -- meta
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok", "version": __version__, "llm_enabled": s.llm_enabled,
            "llm_budget_usd": s.llm_budget_usd, "max_upload_bytes": s.max_upload_bytes,
            "cache_bytes": {n: _du(s.cache_subdir(n)) for n in _CACHES},
            "providers": {
                "adsb": {"name": "adsb.lol", "network": True},
                "dem": {"name": "Copernicus GLO-30", "network": True},
                "osm": {"name": "OpenStreetMap (Overpass/Nominatim)", "network": True},
                "basemap": {"enabled": s.basemap_enabled, "network": s.basemap_enabled},
                "claude": {"enabled": s.llm_enabled, "network": s.llm_enabled, "opt_in_per_job": True},
            },
            "network_note": "Audio, ADS-B, DEM and OCR processing stay local. Third-party calls: "
            "adsb.lol/GitHub, Copernicus S3, OSM services, the optional basemap, and the Claude API only for "
            "jobs with the visual module and consent.",
        }

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "basemap_enabled": s.basemap_enabled, "basemap_url": s.basemap_url if s.basemap_enabled else None,
            "llm_enabled": s.llm_enabled, "visual_notice": VISUAL_NOTICE, "use_notice": USE_NOTICE,
            "attribution": ATTRIBUTION, "max_upload_bytes": s.max_upload_bytes,
            "llm_budget_usd": s.llm_budget_usd, "default_h_obs_m": s.h_obs_m,
            "retention_days": s.retention_days, "max_jobs_per_day": s.max_jobs_per_day,
        }

    # -- jobs
    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [view(j) for j in reversed(store.list())]

    @app.post("/api/jobs", status_code=201)
    async def post_job(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        constraints: str = Form(...),
        visual: str | None = Form(None),
        visual_consent: str | None = Form(None),
        events: str | None = Form(None),
        weights: str | None = Form(None),
    ) -> dict[str, Any]:
        check_rate_limit()
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > s.max_upload_bytes + 1_000_000:
            raise HTTPException(413, f"upload exceeds limit of {s.max_upload_bytes} bytes")
        cons = _json_field(constraints, "constraints")
        kw: dict[str, Any] = {
            "constraints": cons, "visual": _bool(visual), "visual_consent": _bool(visual_consent),
            "events": _json_field(events, "events"),
        }
        w = _json_field(weights, "weights")
        if w is not None:
            kw["weights"] = w
        params = _params_or_422(**kw)  # validates before any bytes are stored
        suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(file.filename or "").suffix)[:10]
        tmp_dir = s.data / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
            dst = Path(td) / ("upload" + suffix)
            with dst.open("wb") as out:
                while chunk := await file.read(CHUNK):
                    total += len(chunk)
                    if total > s.max_upload_bytes:
                        raise HTTPException(413, f"upload exceeds limit of {s.max_upload_bytes} bytes")
                    out.write(chunk)
            if total == 0:
                raise HTTPException(422, "empty file")
            jid = create_job(store, s, params, dst)
        runner.submit(jid)
        return view(store.get(jid))  # type: ignore[arg-type]

    @app.get("/api/jobs/{job_id}")
    def get_job_ep(job_id: str) -> dict[str, Any]:
        return view(get_job(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        get_job(job_id)
        if not store.request_cancel(job_id):
            raise HTTPException(409, "job already finished")
        return view(get_job(job_id))

    @app.post("/api/jobs/{job_id}/rerun", status_code=201)
    def rerun(job_id: str, body: RerunBody) -> dict[str, Any]:
        old = get_job(job_id)
        p = dict(old["params"])
        for k in ("constraints", "events", "visual", "visual_consent", "weights"):
            v = getattr(body, k)
            if v is not None:
                p[k] = v
        params = _params_or_422(**p)
        src = workspace(s, job_id) / params.input_name
        if not src.is_file():
            raise HTTPException(410, "original input is no longer available")
        check_rate_limit()
        jid = create_job(store, s, params, src)  # stage cache is reused where inputs are unchanged
        runner.submit(jid)
        return view(store.get(jid))  # type: ignore[arg-type]

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete(job_id: str) -> None:
        j = get_job(job_id)
        if j["status"] == S.RUNNING:
            store.request_cancel(job_id)
            raise HTTPException(409, "job is running; cancel requested, delete again once it has stopped")
        store.delete(job_id)
        shutil.rmtree(workspace(s, job_id), ignore_errors=True)

    @app.get("/api/jobs/{job_id}/result")
    def result(job_id: str) -> JSONResponse:
        j = get_job(job_id)
        if j["status"] != S.DONE or not j["result_path"]:
            raise HTTPException(409, f"job is {j['status']}; no result yet")
        p = Path(j["result_path"])
        if not p.is_file():
            raise HTTPException(404, "result file missing")
        data = json.loads(p.read_text())
        if isinstance(data, dict):  # exports carry the attribution wherever results are shown (plan 7)
            data.setdefault("attribution", ATTRIBUTION)
        return JSONResponse(data)

    @app.get("/api/jobs/{job_id}/artifacts/{name:path}")
    def artifact(job_id: str, name: str) -> FileResponse:
        get_job(job_id)
        ws = workspace(s, job_id).resolve()
        try:
            p = (ws / name).resolve()
            p.relative_to(ws)
        except (ValueError, OSError):
            raise HTTPException(404, "no such artifact") from None
        if "\x00" in name or not p.is_file():
            raise HTTPException(404, "no such artifact")
        return FileResponse(p)

    @app.get("/api/jobs/{job_id}/stream")
    async def stream(job_id: str, request: Request) -> EventSourceResponse:
        get_job(job_id)
        try:
            last = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            last = 0

        async def gen() -> AsyncIterator[dict[str, str]]:
            seq = last
            while True:
                if await request.is_disconnected():
                    return
                job = store.get(job_id)
                if job is None:  # deleted meanwhile
                    yield {"event": "end", "data": json.dumps({"status": "deleted"})}
                    return
                ended = job["status"] in END_STATES  # read status first, then drain the log
                for e in store.log_since(job_id, seq):
                    seq = e["seq"]
                    msg = e["message"]
                    if e["kind"] == "error":
                        msg = (msg.strip().splitlines() or [""])[-1]  # no tracebacks over the wire
                    data = {"seq": seq, "ts": e["ts"], "kind": e["kind"], "stage": e["stage"],
                            "progress": e["progress"], "message": msg}
                    yield {"id": str(seq), "event": e["kind"], "data": json.dumps(data)}
                if ended:
                    yield {"event": "end", "data": json.dumps(
                        {"status": job["status"], "error": job["error"], "progress": job["progress"]})}
                    return
                await asyncio.sleep(poll_interval)

        return EventSourceResponse(gen(), ping=15)

    if (STATIC_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
