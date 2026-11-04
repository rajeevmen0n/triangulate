# ruff: noqa: E501
"""adsb.lol historical traces (github.com/adsblol/globe_history_YYYY), plan section 3.4.

Flow: release discovery -> one streaming pass over the split tar that builds a per-day index
(`index.parquet` + `manifest.json`, stops at the first `heatmap/` member) -> region queries pick
aircraft from the index and fetch only their gzip members with HTTP Range requests (coalesced,
parallel, split at part boundaries) -> decode + clip.

Cache layout (`<cache>/adsblol/<YYYY-MM-DD>/<variant>/`): `index.parquet`, `manifest.json`,
`members/<icao>.gz` (raw gzip members). The full tar is never stored.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import tarfile
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl

from ..config import get_settings
from ..core.progress import Progress, null_progress
from ..core.types import AIRCRAFT_SCHEMA, TRACK_SCHEMA, TrackQuery, TrackSet
from .provider import ProviderError, ReleaseNotFound, ReleaseNotPublished, utc_days
from .readsb_trace import decode_trace, query_cells, summarize_trace

log = logging.getLogger(__name__)

LICENCE = "Flight data (c) adsb.lol contributors, ODbL 1.0 (https://github.com/adsblol)"
API_BASE = "https://api.github.com"
PUBLISH_DELAY = timedelta(hours=3)

INDEX_SCHEMA: dict[str, Any] = {
    **{k: v for k, v in AIRCRAFT_SCHEMA.items()},
    "tmin": pl.Float64, "tmax": pl.Float64,
    "lat_min": pl.Float64, "lat_max": pl.Float64, "lon_min": pl.Float64, "lon_max": pl.Float64,
    "n_points": pl.Int64, "offset": pl.Int64, "size": pl.Int64, "cells": pl.List(pl.Int32),
}


@dataclass(frozen=True)
class Part:
    name: str
    size: int
    url: str


@dataclass(frozen=True)
class Release:
    tag: str
    release_id: int | None
    published_at: str | None
    parts: tuple[Part, ...]

    @property
    def total(self) -> int:
        return sum(p.size for p in self.parts)


@dataclass(frozen=True)
class RangePlan:
    """One coalesced global byte range [start, end) and the (icao, offset, size) members inside it."""

    start: int
    end: int
    members: tuple[tuple[str, int, int], ...]


def plan_ranges(members: list[tuple[str, int, int]], gap: int = 65536) -> list[RangePlan]:
    """Coalesce member data ranges (icao, offset, size) that are < `gap` bytes apart."""
    out: list[RangePlan] = []
    cur: list[tuple[str, int, int]] = []
    start = end = 0
    for m in sorted(members, key=lambda m: m[1]):
        _, off, size = m
        if cur and off - end < gap:
            cur.append(m)
            end = max(end, off + size)
        else:
            if cur:
                out.append(RangePlan(start, end, tuple(cur)))
            cur, start, end = [m], off, off + size
    if cur:
        out.append(RangePlan(start, end, tuple(cur)))
    return out


def split_at_parts(start: int, end: int, part_sizes: list[int]) -> list[tuple[int, int, int]]:
    """Split global [start, end) at part boundaries: [(part_index, local_start, local_end), ...]."""
    bounds = [0]
    for s in part_sizes:
        bounds.append(bounds[-1] + s)
    if end > bounds[-1]:
        raise ValueError("range beyond the end of the release")
    pieces = []
    i = bisect.bisect_right(bounds, start) - 1
    pos = start
    while pos < end:
        stop = min(end, bounds[i + 1])
        pieces.append((i, pos - bounds[i], stop - bounds[i]))
        pos = stop
        i += 1
    return pieces


class _ChunkReader:
    """Minimal file-like (read(n)) over an iterator of byte chunks, for tarfile 'r|'."""

    def __init__(self, it: Iterator[bytes]):
        self.it, self.buf, self.pos = it, b"", 0

    def read(self, n: int = -1) -> bytes:
        out: list[bytes] = []
        need = n
        while need != 0:
            if self.pos >= len(self.buf):
                try:
                    self.buf, self.pos = next(self.it), 0
                except StopIteration:
                    break
                continue
            take = self.buf[self.pos : self.pos + (need if need > 0 else len(self.buf))]
            self.pos += len(take)
            if need > 0:
                need -= len(take)
            out.append(take)
        return b"".join(out)


def _summ_batch(batch: list[tuple[str, int, int, bytes]], day_start: float, qcells: np.ndarray | None):
    return [summarize_trace(raw, day_start, qcells) for _, _, _, raw in batch]


def _trace_icao(name: str) -> str | None:
    base = name.rsplit("/", 1)[-1]
    if not (base.startswith("trace_full_") and base.endswith(".json")):
        return None
    icao = base[len("trace_full_") : -len(".json")].lower()
    ok = 6 <= len(icao) <= 7 and all(c in "0123456789abcdef~" for c in icao)
    return icao if ok else None


class AdsbLolProvider:
    name = "adsblol"
    licence = LICENCE

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        client: httpx.Client | None = None,
        *,
        variant: str = "prod",
        github_token: str | None = None,
        api_base: str = API_BASE,
        range_gap: int = 65536,
        range_workers: int = 8,
        restream_fraction: float = 0.4,
        index_workers: int | None = None,
        chunk_size: int = 1 << 20,
        max_retries: int = 5,
        now: Callable[[], datetime] | None = None,
    ):
        st = get_settings()
        self.cache_root = Path(cache_dir) if cache_dir else st.cache_subdir("adsb") / "adsblol"
        token = github_token or (st.github_token.get_secret_value() if st.github_token else None)
        headers = {"User-Agent": st.user_agent, "Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._owns_client = client is None
        self.client = client or httpx.Client(
            headers=headers, follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0)
        )
        self._api_headers = headers
        self.variant = variant
        self.api_base = api_base.rstrip("/")
        self.range_gap, self.range_workers = range_gap, range_workers
        self.restream_fraction = restream_fraction
        self.index_workers = (min(4, os.cpu_count() or 1) if index_workers is None else index_workers)
        self.chunk_size, self.max_retries = chunk_size, max_retries
        self._now = now or (lambda: datetime.now(UTC))

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    # -- paths -----------------------------------------------------------------------------------

    def day_dir(self, day: date) -> Path:
        return self.cache_root / day.isoformat() / self.variant

    def index_path(self, day: date) -> Path:
        return self.day_dir(day) / "index.parquet"

    def manifest_path(self, day: date) -> Path:
        return self.day_dir(day) / "manifest.json"

    def member_path(self, day: date, icao: str) -> Path:
        return self.day_dir(day) / "members" / f"{icao}.gz"

    def tag(self, day: date) -> str:
        return f"v{day:%Y.%m.%d}-planes-readsb-{self.variant}-0"

    # -- release discovery -------------------------------------------------------------------------

    def release(self, day: date) -> Release:
        tag = self.tag(day)
        url = f"{self.api_base}/repos/adsblol/globe_history_{day.year}/releases/tags/{tag}"
        r = self.client.get(url, headers=self._api_headers)
        if r.status_code == 404:
            end = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
            if self._now() < end + PUBLISH_DELAY:
                raise ReleaseNotPublished(f"{tag} is not published yet (about 3 h after the UTC day ends)")
            raise ReleaseNotFound(f"no adsb.lol release {tag}")
        if r.status_code in (403, 429):
            raise ProviderError("GitHub API rate limit; set GITHUB_TOKEN to raise it")
        r.raise_for_status()
        js = r.json()
        assets = sorted(
            (a for a in js.get("assets", []) if ".tar" in a["name"]), key=lambda a: a["name"]
        )
        if not assets:
            raise ReleaseNotFound(f"release {tag} has no tar assets")
        parts = tuple(Part(a["name"], int(a["size"]), a["browser_download_url"]) for a in assets)
        return Release(tag, js.get("id"), js.get("published_at"), parts)

    def days_available(self, t0: datetime, t1: datetime) -> list[date]:
        out = []
        for d in utc_days(t0, t1):
            if self.manifest_path(d).exists():
                out.append(d)
                continue
            try:
                self.release(d)
                out.append(d)
            except ReleaseNotFound:
                pass
        return out

    # -- streaming + index -------------------------------------------------------------------------

    def _iter_stream(self, rel: Release, etags: dict[str, str]) -> Iterator[bytes]:
        """Yield the concatenated parts; resume with Range after transport errors."""
        for part in rel.parts:
            pos, fails = 0, 0
            while pos < part.size:
                headers = {"Range": f"bytes={pos}-"} if pos else {}
                try:
                    with self.client.stream("GET", part.url, headers=headers) as r:
                        r.raise_for_status()
                        if pos and r.status_code != 206:
                            raise ProviderError("server ignored Range while resuming")
                        if r.headers.get("etag"):
                            etags[part.name] = r.headers["etag"]
                        for chunk in r.iter_raw(self.chunk_size):
                            pos += len(chunk)
                            yield chunk
                except (httpx.TransportError, httpx.DecodingError) as e:
                    fails += 1
                    if fails > self.max_retries:
                        raise ProviderError(f"stream of {part.name} failed at byte {pos}: {e}") from e
                    log.warning("stream interrupted at %s+%d (%s); resuming", part.name, pos, e)
                    time.sleep(min(2.0 * fails, 10.0))

    def build_index(
        self,
        day: date,
        rel: Release | None = None,
        keep: TrackQuery | None = None,
        progress: Progress = null_progress,
    ) -> dict[str, Any]:
        """Stream the whole day once (stopping at `heatmap/`), write index + manifest.

        Members whose coarse cells intersect `keep` are also stored raw in the member cache.
        """
        rel = rel or self.release(day)
        t_start = time.time()
        day_start = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
        qcells = None
        if keep is not None:
            qcells = query_cells(keep.bbox, keep.t0.timestamp(), keep.t1.timestamp(), day_start)
        etags: dict[str, str] = {}
        gen = self._iter_stream(rel, etags)
        tar = tarfile.open(fileobj=_ChunkReader(gen), mode="r|")  # type: ignore[arg-type]
        records: list[dict[str, Any]] = []
        versions: dict[str, int] = {}
        bad = 0
        traces_end = 0
        pool = ProcessPoolExecutor(self.index_workers) if self.index_workers > 1 else None
        pending: deque[tuple[Future, list[tuple[str, int, int, bytes]]]] = deque()
        batch: list[tuple[str, int, int, bytes]] = []
        batch_bytes = 0

        def absorb(items: list[tuple[str, int, int, bytes]], sums: list[dict[str, Any] | None]) -> None:
            nonlocal bad
            for (icao, off, size, raw), s in zip(items, sums, strict=True):
                if s is None:
                    bad += 1
                    continue
                hit = s.pop("hit")
                ver = s.pop("version", None)
                if ver is not None:
                    versions[str(ver)] = versions.get(str(ver), 0) + 1
                s["icao"] = icao  # the file name is authoritative for the cache key
                s.update(offset=off, size=size)
                records.append(s)
                if hit:
                    self._write_member(day, icao, raw)

        def flush() -> None:
            nonlocal batch, batch_bytes
            if not batch:
                return
            items, batch, batch_bytes = batch, [], 0
            if pool is None:
                absorb(items, _summ_batch(items, day_start, qcells))
                return
            pending.append((pool.submit(_summ_batch, items, day_start, qcells), items))
            while len(pending) >= 2 * self.index_workers:
                drain(block=True)

        def drain(block: bool) -> None:
            while pending and (block or pending[0][0].done()):
                fut, items = pending[0]
                if block:
                    wait([fut], return_when=FIRST_COMPLETED)
                pending.popleft()
                absorb(items, fut.result())
                block = False

        try:
            for m in tar:
                name = m.name
                if "heatmap/" in name:
                    break
                icao = _trace_icao(name) if m.isfile() else None
                if icao is None:
                    continue
                f = tar.extractfile(m)
                raw = f.read() if f else b""
                traces_end = m.offset_data + m.size
                batch.append((icao, m.offset_data, m.size, raw))
                batch_bytes += len(raw)
                if len(batch) >= 200 or batch_bytes >= 16 << 20:
                    flush()
                    drain(block=False)
                    progress("adsb", min(m.offset_data / max(rel.total, 1), 0.99), f"indexing {len(records)} aircraft")
            flush()
            while pending:
                drain(block=True)
        finally:
            gen.close()
            if pool is not None:
                pool.shutdown(cancel_futures=True)
        if not records:
            raise ProviderError(f"no traces found in {rel.tag}")

        df = pl.DataFrame(records, schema=INDEX_SCHEMA)
        d = self.day_dir(day)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "index.parquet.tmp"
        df.write_parquet(tmp)
        os.replace(tmp, self.index_path(day))
        manifest = {
            "release": rel.tag, "release_id": rel.release_id, "published_at": rel.published_at,
            "variant": self.variant, "day": day.isoformat(),
            "parts": [{"name": p.name, "size": p.size, "url": p.url, "etag": etags.get(p.name)} for p in rel.parts],
            "total_size": rel.total, "traces_end": traces_end, "n_aircraft": df.height,
            "n_bad_members": bad, "readsb_versions": versions,
            "licence": LICENCE, "index_seconds": round(time.time() - t_start, 1),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        tmpm = d / "manifest.json.tmp"
        tmpm.write_text(json.dumps(manifest, indent=1))
        os.replace(tmpm, self.manifest_path(day))
        progress("adsb", 1.0, f"indexed {df.height} aircraft")
        return manifest

    def _write_member(self, day: date, icao: str, raw: bytes) -> None:
        p = self.member_path(day, icao)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, p)

    def ensure_index(self, day: date, progress: Progress = null_progress, keep: TrackQuery | None = None) -> dict[str, Any]:
        """Return the day manifest, building the index (one full streaming pass) if absent."""
        if self.manifest_path(day).exists() and self.index_path(day).exists():
            return json.loads(self.manifest_path(day).read_text())
        return self.build_index(day, keep=keep, progress=progress)

    def load_index(self, day: date) -> pl.DataFrame:
        return pl.read_parquet(self.index_path(day))

    # -- region query ------------------------------------------------------------------------------

    @staticmethod
    def select(index: pl.DataFrame, q: TrackQuery, day: date) -> pl.DataFrame:
        """Index rows whose bbox/time/cells can intersect the query (superset, then decode+clip)."""
        s, w, n, e = q.bbox
        t0, t1 = q.t0.timestamp(), q.t1.timestamp()
        day_start = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
        qc = query_cells(q.bbox, t0, t1, day_start)
        cand = index.filter(
            (pl.col("lat_max") >= s) & (pl.col("lat_min") <= n) & (pl.col("lon_max") >= w)
            & (pl.col("lon_min") <= e) & (pl.col("tmax") >= t0) & (pl.col("tmin") <= t1)
        )
        hits = (
            cand.select("icao", "cells").explode("cells", empty_as_null=True).filter(pl.col("cells").is_in(qc.tolist()))
            .select("icao").unique()
        )
        return cand.join(hits, on="icao", how="semi")

    def _release_from_manifest(self, man: dict[str, Any]) -> Release:
        parts = tuple(Part(p["name"], p["size"], p["url"]) for p in man["parts"])
        return Release(man["release"], man.get("release_id"), man.get("published_at"), parts)

    def _get_range(self, part: Part, lo: int, hi: int) -> bytes:
        err: Exception | None = None
        for attempt in range(3):
            try:
                r = self.client.get(part.url, headers={"Range": f"bytes={lo}-{hi - 1}"})
                if r.status_code != 206 or len(r.content) != hi - lo:
                    raise ProviderError(f"bad range response {r.status_code} ({len(r.content)} B) for {part.name}")
                return r.content
            except (httpx.TransportError, ProviderError) as e:
                err = e
                time.sleep(0.5 * (attempt + 1))
        raise ProviderError(f"range fetch failed: {err}") from err

    def fetch_members(self, day: date, man: dict[str, Any], rows: pl.DataFrame) -> int:
        """Ensure the raw members for `rows` (icao, offset, size) are cached; returns bytes fetched."""
        missing = [
            (r["icao"], r["offset"], r["size"]) for r in rows.select("icao", "offset", "size").iter_rows(named=True)
            if not self.member_path(day, r["icao"]).exists()
        ]
        if not missing:
            return 0
        rel = self._release_from_manifest(man)
        sizes = [p.size for p in rel.parts]
        plans = plan_ranges(missing, self.range_gap)
        jobs = [(plan, split_at_parts(plan.start, plan.end, sizes)) for plan in plans]

        def run(job: tuple[RangePlan, list[tuple[int, int, int]]]) -> int:
            plan, pieces = job
            buf = b"".join(self._get_range(rel.parts[i], lo, hi) for i, lo, hi in pieces)
            for icao, off, size in plan.members:
                self._write_member(day, icao, buf[off - plan.start : off - plan.start + size])
            return len(buf)

        with ThreadPoolExecutor(self.range_workers) as ex:
            return sum(ex.map(run, jobs))

    def _fetch_day(self, day: date, q: TrackQuery, progress: Progress) -> tuple[list[pl.DataFrame], list[dict[str, Any]]]:
        t0, t1 = q.t0.timestamp(), q.t1.timestamp()
        built = not (self.manifest_path(day).exists() and self.index_path(day).exists())
        man = self.ensure_index(day, progress, keep=q)
        index = self.load_index(day)
        sel = self.select(index, q, day)
        have = [self.member_path(day, i).exists() for i in sel["icao"]]
        if not all(have):
            span = max(man["traces_end"], 1)
            if not built and sel["size"].sum() > self.restream_fraction * span:
                man = self.build_index(day, keep=q, progress=progress)
            else:
                self.fetch_members(day, man, sel)
        frames, acs = [], []
        s, w, n, e = q.bbox
        for r in sel.iter_rows(named=True):
            ac, df = decode_trace(self.member_path(day, r["icao"]).read_bytes())
            ac["icao"] = r["icao"]
            df = df.filter(
                pl.col("t").is_between(t0, t1) & pl.col("lat").is_between(s, n) & pl.col("lon").is_between(w, e)
            )
            if not q.include_ground:
                df = df.filter(~pl.col("on_ground"))
            if df.height:
                frames.append(df)
                acs.append(ac)
        return frames, acs

    def fetch(self, q: TrackQuery, progress: Progress = null_progress) -> TrackSet:
        days = utc_days(q.t0, q.t1)
        frames: list[pl.DataFrame] = []
        acs: dict[str, dict[str, Any]] = {}
        for i, day in enumerate(days):
            def sub(stage: str, frac: float, msg: str = "", i: int = i) -> None:
                progress(stage, (i + frac) / len(days), msg)

            f, a = self._fetch_day(day, q, sub)
            frames += f
            for ac in a:
                acs.setdefault(ac["icao"], ac)
        if not frames:
            return TrackSet.empty(self.name, self.licence)
        tracks = pl.concat(frames).sort("icao", "t").unique(["icao", "t"], keep="first", maintain_order=True)
        aircraft = pl.DataFrame(list(acs.values()), schema=AIRCRAFT_SCHEMA).sort("icao")
        progress("adsb", 1.0, f"{aircraft.height} aircraft")
        return TrackSet(
            tracks=tracks.select(list(TRACK_SCHEMA)), aircraft=aircraft, provider=self.name, licence=self.licence
        )
