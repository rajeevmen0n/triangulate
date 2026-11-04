# ruff: noqa: E501
import gzip
import io
import json
import random
import tarfile
from datetime import UTC, date, datetime

import httpx
import orjson
import polars as pl
import pytest

from triangulate.adsb.adsblol import AdsbLolProvider, Part, plan_ranges, split_at_parts
from triangulate.adsb.provider import ReleaseNotFound, ReleaseNotPublished
from triangulate.core.types import TRACK_SCHEMA, TrackQuery

DAY = date(2026, 9, 8)
DAY_TS = int(datetime(2026, 9, 8, tzinfo=UTC).timestamp())
TAG = "v2026.09.08-planes-readsb-prod-0"


def trace(icao, lat, lon, n=40, t_off=3600 * 12, typ="C172"):
    rnd = random.Random(icao)
    rows = [[t_off + 6 * i, lat + 0.0005 * i, lon + 0.0005 * i, 1000 + rnd.randint(0, 9999), 90.0, 45.0, 0, 0,
             {"category": "A1"} if i == 0 else None, "adsb_icao", None, None, None, None] for i in range(n)]
    rows.insert(0, [t_off - 6, lat, lon, "ground", 0.0, 0.0, 2, None, None, "adsb_icao"])
    return {"icao": icao, "r": "N" + icao[:3].upper(), "t": typ, "desc": "X", "dbFlags": 0,
            "timestamp": DAY_TS, "version": "v", "trace": rows}


def build_tar():
    # near Oyster Bay (40.88,-73.53), near Boston, a TIS-B track, and a big filler
    docs = {
        "a00001": trace("a00001", 40.87, -73.55),
        "a00002": trace("a00002", 42.36, -71.06),
        "~00003": trace("~00003", 40.90, -73.50),
        "a00004": trace("a00004", 33.0, -100.0, n=300),
        "a00005": trace("a00005", 34.0, -101.0, n=60),
        "a00006": trace("a00006", 35.0, -102.0, n=60),
    }
    # padding so that traces span is large relative to a single selected member
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        def add(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        add("./LICENSE-cc0.txt", b"cc0")
        add("./acas/acas.json.gz", gzip.compress(b"{}"))
        for icao, d in docs.items():
            add(f"./traces/{icao[-2:]}/trace_full_{icao}.json", gzip.compress(orjson.dumps(d)))
            add(f"./traces/{icao[-2:]}/trace_recent_{icao}.json", b"ignored")
        add("./heatmap/00.bin.ttf", b"H" * 400_000)
    data = buf.getvalue()
    return data, docs


class Fake:
    """Serves a split tar + GitHub release JSON through an httpx MockTransport and counts requests."""

    def __init__(self, data: bytes, split: int, published: bool = True):
        self.parts = {"x.tar.aa": data[:split], "x.tar.ab": data[split:]}
        self.published = published
        self.api_calls = 0
        self.full_gets = 0
        self.range_gets = 0
        self.bytes_streamed = 0
        self.log: list[str] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "api.test" in url:
            self.api_calls += 1
            if not self.published or TAG not in url:
                return httpx.Response(404, json={})
            assets = [{"name": f"{TAG}.tar.{s}", "size": len(self.parts[f"x.tar.{s}"]),
                       "browser_download_url": f"https://dl.test/{s}"} for s in ("ab", "aa")]  # unsorted on purpose
            return httpx.Response(200, json={"id": 7, "published_at": "2026-09-09T03:25:00Z", "assets": assets})
        body = self.parts[f"x.tar.{url[-2:]}"]
        rng = req.headers.get("range")
        if rng:
            lo, _, hi = rng.removeprefix("bytes=").partition("-")
            lo = int(lo)
            hi = int(hi) if hi else len(body) - 1
            self.range_gets += 1
            self.log.append(f"R {url[-2:]} {lo}-{hi}")
            return httpx.Response(206, content=body[lo : hi + 1])
        self.full_gets += 1
        self.log.append(f"F {url[-2:]}")
        outer = self

        def gen():
            for i in range(0, len(body), 4096):
                outer.bytes_streamed += min(4096, len(body) - i)
                yield body[i : i + 4096]

        return httpx.Response(200, content=gen())


@pytest.fixture
def setup(tmp_path):
    data, docs = build_tar()
    # part boundary: 512-aligned and inside the member of a00004
    probe = tarfile.open(fileobj=io.BytesIO(data))
    m = probe.getmember("./traces/04/trace_full_a00004.json")
    split = (m.offset_data + m.size // 2) // 512 * 512
    assert m.size > 1024
    assert m.offset_data < split < m.offset_data + m.size
    fake = Fake(data, split)
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), follow_redirects=True)

    def make(**kw):
        return AdsbLolProvider(cache_dir=tmp_path / "c", client=client, api_base="https://api.test",
                               index_workers=kw.pop("index_workers", 0), chunk_size=4096, **kw)
    return make, fake, docs, data, split, tmp_path


def query(bbox=(40.5, -74.0, 41.2, -73.0), h0=11, h1=14, ground=False):
    return TrackQuery(bbox=bbox, t0=datetime(2026, 9, 8, h0, tzinfo=UTC), t1=datetime(2026, 9, 8, h1, tzinfo=UTC),
                      include_ground=ground)


def test_plan_and_split():
    ms = [("a", 1000, 100), ("b", 1100 + 500, 100), ("c", 200_000, 50), ("d", 100_000, 10)]
    plans = plan_ranges(ms, gap=64 * 1024)
    assert [(p.start, p.end, [m[0] for m in p.members]) for p in plans] == [
        (1000, 1700, ["a", "b"]), (100_000, 100_010, ["d"]), (200_000, 200_050, ["c"])
    ]
    assert plan_ranges([("a", 0, 10), ("b", 20, 10)], gap=5)[0].end == 10
    assert len(plan_ranges([("a", 0, 10), ("b", 20, 10)], gap=5)) == 2
    assert split_at_parts(10, 90, [100, 100]) == [(0, 10, 90)]
    assert split_at_parts(90, 110, [100, 100]) == [(0, 90, 100), (1, 0, 10)]
    assert split_at_parts(100, 150, [100, 100]) == [(1, 0, 50)]
    with pytest.raises(ValueError):
        split_at_parts(150, 250, [100, 100])


def test_index_stops_at_heatmap_and_offsets(setup):
    make, fake, docs, data, split, tmp = setup
    p = make()
    man = p.build_index(DAY)
    assert man["n_aircraft"] == 6 and man["release"] == TAG
    assert [x["name"] for x in man["parts"]] == [f"{TAG}.tar.aa", f"{TAG}.tar.ab"]  # sorted by suffix
    assert man["licence"].startswith("Flight data")
    # stopped at heatmap: did not stream the 400 kB heatmap member
    assert fake.bytes_streamed < len(data) - 300_000
    idx = p.load_index(DAY)
    assert set(idx["icao"]) == set(docs)
    for r in idx.iter_rows(named=True):
        raw = data[r["offset"] : r["offset"] + r["size"]]
        assert orjson.loads(gzip.decompress(raw))["icao"] == r["icao"]
        assert r["n_points"] == len(docs[r["icao"]]["trace"])
    r = idx.filter(pl.col("icao") == "a00001").row(0, named=True)
    assert r["typecode"] == "C172" and r["category"] == "A1" and r["lat_min"] == pytest.approx(40.87)
    # the boundary-straddling member really straddles
    r4 = idx.filter(pl.col("icao") == "a00004").row(0, named=True)
    assert r4["offset"] < split < r4["offset"] + r4["size"]


def test_fetch_reassembles_boundary_member_with_ranges(setup):
    make, fake, docs, data, split, tmp = setup
    p = make()
    p.build_index(DAY)  # no member kept
    fake.full_gets = fake.range_gets = 0
    ts = p.fetch(query(bbox=(32.5, -100.5, 33.5, -99.5)))  # a00004 only
    assert fake.full_gets == 0 and fake.range_gets >= 2  # split into two range requests
    assert ts.aircraft["icao"].to_list() == ["a00004"]
    assert ts.tracks.height == 300  # ground row removed
    assert ts.tracks["t"].min() == DAY_TS + 12 * 3600
    assert ts.tracks.schema == pl.Schema(TRACK_SCHEMA)
    assert gzip.decompress(p.member_path(DAY, "a00004").read_bytes()) == gzip.decompress(
        tarfile.open(fileobj=io.BytesIO(data)).extractfile("./traces/04/trace_full_a00004.json").read()
    )


def test_first_fetch_streams_once_then_only_ranges(setup):
    make, fake, docs, data, split, tmp = setup
    p = make()
    ts = p.fetch(query())  # Oyster Bay region: a00001 + ~00003
    assert sorted(ts.aircraft["icao"]) == ["a00001", "~00003"]
    assert fake.full_gets == 2 and fake.range_gets == 0  # whole-day stream: two parts, once
    assert "ODbL" in ts.licence
    # other region: index cached, only Range requests (restream threshold raised)
    fake.full_gets = fake.range_gets = fake.api_calls = 0
    p2 = make(restream_fraction=1.0)
    ts2 = p2.fetch(query(bbox=(42.0, -72.0, 43.0, -70.0)))
    assert ts2.aircraft["icao"].to_list() == ["a00002"]
    assert fake.full_gets == 0 and fake.range_gets >= 1 and fake.api_calls == 0
    # repeat: fully cached, no asset requests at all
    fake.range_gets = 0
    ts3 = make(restream_fraction=1.0).fetch(query(bbox=(42.0, -72.0, 43.0, -70.0)))
    assert fake.full_gets == 0 and fake.range_gets == 0 and ts3.tracks.equals(ts2.tracks)


def test_clipping_ground_and_time(setup):
    make, fake, docs, data, split, tmp = setup
    p = make()
    ts = p.fetch(query(ground=True))
    assert ts.tracks.filter(pl.col("icao") == "a00001")["on_ground"].sum() == 1
    # time window excluding all points
    assert p.fetch(query(h0=0, h1=2)).tracks.height == 0
    # bbox clip
    small = p.fetch(query(bbox=(40.870, -73.551, 40.8715, -73.5485)))
    assert 0 < small.tracks.height < 40


def test_multiprocess_index_matches_inline(setup):
    make, fake, docs, data, split, tmp = setup
    a = make(index_workers=2)
    a.build_index(DAY)
    ia = a.load_index(DAY).sort("icao")
    b = AdsbLolProvider(cache_dir=tmp / "c2", client=a.client, api_base="https://api.test", index_workers=0)
    b.build_index(DAY)
    assert ia.equals(b.load_index(DAY).sort("icao"))


def test_release_errors(setup):
    make, fake, *_ = setup
    fake.published = False
    p = make(now=lambda: datetime(2026, 9, 9, 1, tzinfo=UTC))
    with pytest.raises(ReleaseNotPublished):
        p.release(DAY)
    p = make(now=lambda: datetime(2026, 9, 20, tzinfo=UTC))
    with pytest.raises(ReleaseNotFound):
        p.release(DAY)
    assert p.days_available(datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 8, 5, tzinfo=UTC)) == []


def test_manifest_written(setup):
    make, fake, *_ = setup
    p = make()
    p.build_index(DAY)
    m = json.loads(p.manifest_path(DAY).read_text())
    assert m["traces_end"] > 0 and m["n_bad_members"] == 0 and Part  # smoke


@pytest.mark.network
@pytest.mark.slow
def test_real_release_index():
    import time

    p = AdsbLolProvider()
    t = time.time()
    man = p.ensure_index(DAY)
    print(f"index: {man['n_aircraft']} aircraft, built in {man['index_seconds']} s (test {time.time() - t:.0f} s)")
    assert man["n_aircraft"] > 30_000
    assert p.index_path(DAY).exists() and p.manifest_path(DAY).exists()
    # region query uses Range requests only after the index exists
    n = {"full": 0, "range": 0}

    def hook(req: httpx.Request) -> None:
        if "api.github.com" in str(req.url):
            return
        n["range" if "range" in req.headers else "full"] += 1

    p.client.event_hooks["request"] = [hook]
    q = TrackQuery(bbox=(40.6, -73.8, 41.2, -73.2), t0=datetime(2026, 9, 8, 20, 30, tzinfo=UTC),
                   t1=datetime(2026, 9, 8, 21, 30, tzinfo=UTC))
    ts = p.fetch(q)
    assert ts.aircraft.height > 0
    first = dict(n)
    p.fetch(q)
    assert n == first and n["full"] == 0
