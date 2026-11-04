"""WP16: Nominatim/Overpass clients, QL builder and clue layers (HTTP mocked from recorded fixtures)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import numpy as np
import pytest

from triangulate.config import Settings
from triangulate.core.geo import haversine_m
from triangulate.core.types import GridSpec
from triangulate.fusion.combine import layers_from_results
from triangulate.fusion.grid import Layer
from triangulate.visual._http import RateLimiter
from triangulate.visual.geocode import GeocodeError, Geocoder
from triangulate.visual.layers import (
    MAX_TOTAL_NATS,
    build_visual_layers,
    parse_sun_text,
    sun_clue_from_clues,
)
from triangulate.visual.overpass import Overpass, OverpassError, QueryRejected, build_ql
from triangulate.visual.schemas import OsmHint, OsmNear, OsmTag, SentFrame, VisualClues, validate_clue_report

FIX = json.loads((Path(__file__).parent.parent / "fixtures" / "osm" / "oyster_bay_bayview.json").read_text())
BBOX = (40.84, -73.58, 40.92, -73.46)
STREET = [(40.875877, -73.5269864)]  # a point on the Oyster Bay "Bayview Avenue" way (fixture hit 2)
UA = "triangulate/0.1"


class Server:
    """httpx handler replaying the fixture; counts requests per service."""

    def __init__(self, status: int = 200):
        self.nominatim: list[httpx.Request] = []
        self.overpass: list[httpx.Request] = []
        self.status = status

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.status != 200:
            return httpx.Response(self.status, text="busy")
        if "overpass" in req.url.host:
            self.overpass.append(req)
            return httpx.Response(200, json=FIX[1]["json"])
        self.nominatim.append(req)
        p = req.url.params
        if p.get("country") or p.get("state"):  # admin polygon: a box around the whole bbox
            poly = {"type": "Polygon", "coordinates": [[[-74.0, 40.5], [-73.5, 40.5], [-73.5, 41.2],
                                                        [-74.0, 41.2], [-74.0, 40.5]]]}
            return httpx.Response(200, json=[{"lat": "40.8", "lon": "-73.8", "osm_type": "relation",
                                              "osm_id": 1, "boundingbox": ["40.5", "41.2", "-74.0", "-73.5"],
                                              "geojson": poly, "display_name": "Area"}])
        if "bayview" in p.get("q", "").lower():
            return httpx.Response(200, json=FIX[0]["json"])
        return httpx.Response(200, json=[])


def clients(tmp_path, server=None, **kw):
    server = server or Server()
    c = httpx.Client(transport=httpx.MockTransport(server))
    lim = RateLimiter(0.0)
    g = Geocoder(tmp_path / "geocode", UA, client=c, limiter=lim, **kw)
    o = Overpass(tmp_path / "overpass", UA, client=c, limiter=lim)
    return server, g, o


def make_vc(clues, regions=(), entities=()):
    frames = {"f0001": 1.0, "f0002": 2.0}
    rep, _ = validate_clue_report({"clues": clues, "region_hypotheses": list(regions),
                                   "searchable_entities": list(entities)}, frames)
    return VisualClues(report=rep, frames_sent=[SentFrame(frame_id=k, t_s=v) for k, v in frames.items()])


def street_clue(conf=0.8, **kw):
    d = {"type": "street_name", "value": "BAYVIEW AVE", "evidence": [{"frame_id": "f0001"}],
         "confidence": conf, "geocode_queries": ["Bayview Avenue, Oyster Bay"], **kw}
    return d


def settings(tmp_path):
    return Settings(data=tmp_path / "data")


# ------------------------------------------------------------------------------------ QL builder

def test_ql_basic_is_bbox_bound_with_limits():
    h = OsmHint(tags=[OsmTag(key="highway", op="exists"), OsmTag(key="name", value="Bayview Avenue")])
    ql = build_ql(h, BBOX)
    assert "[timeout:25]" in ql and "[maxsize:67108864]" in ql
    assert '["name"="Bayview Avenue"]' in ql and '["highway"]' in ql
    assert "(40.840000,-73.580000,40.920000,-73.460000)" in ql and "out center tags 200;" in ql


def test_ql_near_uses_around_and_expanded_bbox():
    h = OsmHint(tags=[OsmTag(key="amenity", value="cafe")],
                near=OsmNear(tags=[OsmTag(key="railway", value="station")], distance_m=500))
    ql = build_ql(h, BBOX)
    assert "nwr.a(around.b:500);" in ql and "->.a;" in ql and "->.b;" in ql


@pytest.mark.parametrize("tag", [
    OsmTag(key="name\"];out;//", value="x"),
    OsmTag(key="password", value="x"),
    OsmTag(key="note", value="x"),
    OsmTag(key="name", value='x"]; node(1); out; //'),
    OsmTag.model_construct(key="name", op="=", value="a\nb"),
    OsmTag(key="name", value="x;"),
    OsmTag(key="name", value="{{bbox}}"),
    OsmTag(key="name", op="=", value=""),
])
def test_ql_rejects_injection_and_unlisted_keys(tag):
    with pytest.raises(QueryRejected):
        build_ql(OsmHint(tags=[tag, OsmTag(key="amenity", value="cafe")]), BBOX)


def test_ql_rejects_unselective_and_regex_is_literal():
    with pytest.raises(QueryRejected):
        build_ql(OsmHint(tags=[OsmTag(key="building", op="exists")]), BBOX)
    ql = build_ql(OsmHint(tags=[OsmTag(key="name", op="~", value="St. (Mary)")]), BBOX)
    assert r'St\.\ \(Mary\)' in ql
    assert 'name:fr' not in ql
    assert '["name:fr"="a"]' in build_ql(OsmHint(tags=[OsmTag(key="name:fr", value="a")]), BBOX)


# ------------------------------------------------------------------------------------ geocoder

def test_geocoder_clamps_to_bbox_and_sends_user_agent(tmp_path):
    srv, g, _ = clients(tmp_path)
    hits = g.search("Bayview Avenue, Oyster Bay", BBOX, countrycodes=["US"])
    req = srv.nominatim[0]
    assert req.headers["user-agent"] == UA
    q = req.url.params
    assert q["bounded"] == "1" and q["viewbox"] == "-73.580000,40.920000,-73.460000,40.840000"
    assert q["countrycodes"] == "us"
    assert len(hits) == 3 and hits[0].geometry().geom_type == "LineString"
    # hits outside the bbox are dropped locally even if the server returns them
    assert g.search("Bayview Avenue", (10.0, 10.0, 10.1, 10.1)) == []


def test_geocoder_rate_limit_is_one_per_second(tmp_path):
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    lim = RateLimiter(1.0, clock=lambda: now[0], sleep=sleep)
    srv = Server()
    g = Geocoder(tmp_path / "gc", UA, client=httpx.Client(transport=httpx.MockTransport(srv)), limiter=lim)
    for i in range(3):
        g.search(f"query number {i}", BBOX)
    assert len(srv.nominatim) == 3
    assert len(sleeps) == 2 and all(s >= 0.99 for s in sleeps)


def test_geocoder_cache_hit_on_rerun_and_empty_is_cached(tmp_path):
    srv, g, _ = clients(tmp_path)
    g.search("Bayview Avenue", BBOX)
    g.search("nothing here", BBOX)
    assert len(srv.nominatim) == 2
    g2 = Geocoder(tmp_path / "geocode", UA, client=httpx.Client(transport=httpx.MockTransport(srv)))
    assert len(g2.search("Bayview Avenue", BBOX)) == 3
    assert g2.search("nothing here", BBOX) == []
    assert len(srv.nominatim) == 2 and g2.n_requests == 0 and g2.n_cache_hits == 2
    assert list((tmp_path / "geocode").glob("*.json"))


def test_geocoder_budget_and_block(tmp_path):
    _, g, _ = clients(tmp_path, max_requests=1)
    g.search("a b", BBOX)
    with pytest.raises(GeocodeError, match="budget"):
        g.search("c d", BBOX)
    srv, g, _ = clients(tmp_path / "x", Server(status=429))
    with pytest.raises(GeocodeError):
        g.search("a b", BBOX)
    assert g.blocked
    with pytest.raises(GeocodeError, match="blocked"):
        g.search("e f", BBOX)


def test_overpass_budget_cache_and_errors(tmp_path):
    srv, _, o = clients(tmp_path)
    h = OsmHint(tags=[OsmTag(key="highway", op="exists"), OsmTag(key="name", value="Bayview Avenue")])
    feats, capped = o.query_hint(h, BBOX)
    assert len(feats) == 1 and not capped and srv.overpass[0].headers["user-agent"] == UA
    o.query_hint(h, BBOX)
    assert len(srv.overpass) == 1 and o.n_cache_hits == 1
    _, _, bad = clients(tmp_path / "y", Server(status=504))
    with pytest.raises(OverpassError):
        bad.query_hint(h, BBOX)


# ------------------------------------------------------------------------------------ layers

def _peak_latlon(lay: Layer, spec: GridSpec):
    from triangulate.fusion.grid import cell_centers_latlon

    p = max(lay.patches, key=lambda p: float(np.nanmax(p.values)))
    j, i = np.unravel_index(np.nanargmax(p.values), p.values.shape)
    la, lo = cell_centers_latlon(spec, p.level, p.ix0 + i, p.iy0 + j)
    return float(la), float(lo)


def test_street_kernel_peak_within_300m_of_street(tmp_path):
    spec = GridSpec.for_bbox(BBOX)
    srv, g, o = clients(tmp_path)
    vc = make_vc([street_clue(osm_hints=[{"tags": [{"key": "highway", "op": "exists", "value": ""},
                                                   {"key": "name", "value": "Bayview Avenue"}]}])])
    res = build_visual_layers(vc, BBOX, spec, tmp_path / "job", settings=settings(tmp_path),
                              geocoder=g, overpass=o)
    assert res.status == "ok" and res.layers
    geo = next(x for x in res.provenance["layers"] if x["kind"] == "geocode")
    osm = next(x for x in res.provenance["layers"] if x["kind"] == "osm")
    assert geo["depth_nats"] > 0 and osm["depth_nats"] > 0
    layers, _ = layers_from_results([res], tmp_path / "job")
    assert {lay.module for lay in layers} == {"visual"} and len(layers) == len(res.layers)
    lay = next(x for x in layers if x.name == geo["name"])
    la, lo = _peak_latlon(lay, spec)
    street = srv.nominatim and [h for h in g.search("Bayview Avenue, Oyster Bay", BBOX)]
    from shapely.geometry import MultiLineString

    line = MultiLineString([h.geometry() for h in street if h.geometry().geom_type == "LineString"])
    d = min(haversine_m(la, lo, y, x) for g_ in line.geoms for x, y in g_.coords)
    assert d < 300
    # far point is lower than the street
    on = lay.value_at_latlon(*STREET[0])
    off = lay.value_at_latlon(40.86, -73.56)
    assert on > off + 0.5
    assert any("OpenStreetMap" in a for a in res.provenance["attribution"])
    osm_lay = next(x for x in layers if x.name == osm["name"])
    assert osm_lay.value_at_latlon(*STREET[0]) > osm_lay.value_at_latlon(40.9, -73.5)
    # rerun hits caches only
    n_req = len(srv.nominatim) + len(srv.overpass)
    g2 = Geocoder(tmp_path / "geocode", UA, client=httpx.Client(transport=httpx.MockTransport(srv)))
    o2 = Overpass(tmp_path / "overpass", UA, client=httpx.Client(transport=httpx.MockTransport(srv)))
    build_visual_layers(vc, BBOX, spec, tmp_path / "job2", settings=settings(tmp_path),
                        geocoder=g2, overpass=o2)
    assert len(srv.nominatim) + len(srv.overpass) == n_req and g2.n_requests == 0


def test_layer_is_in_wp12_grid_format_and_fuses(tmp_path):
    from triangulate.fusion.combine import fuse

    spec = GridSpec.for_bbox(BBOX)
    _, g, o = clients(tmp_path)
    res = build_visual_layers(make_vc([street_clue()]), BBOX, spec, tmp_path / "job",
                              settings=settings(tmp_path), geocoder=g, overpass=o)
    layers, _ = layers_from_results([res], tmp_path / "job")
    post = fuse(spec, layers, domain=BBOX)
    assert math.isclose(post.total_mass(), 1.0, rel_tol=1e-3)


def test_country_level_low_confidence_is_weak(tmp_path):
    spec = GridSpec.for_bbox(BBOX)
    # polygon covers only the west half of the bbox lon (-74.0..-73.5), so the layer is informative
    _, g, o = clients(tmp_path)
    vc = make_vc([{"type": "language_script", "value": "English", "confidence": 0.3,
                   "implication": {"countries": [{"code": "US", "p": 0.9}]}}])
    res = build_visual_layers(vc, BBOX, spec, tmp_path / "job", settings=settings(tmp_path),
                              geocoder=g, overpass=o)
    (info,) = res.provenance["layers"]
    assert info["kind"] == "country" and info["confidence"] <= 0.3 and info["depth_nats"] < 0.5
    # confident country clue is still capped at 0.6
    vc = make_vc([{"type": "language_script", "value": "English", "confidence": 0.99,
                   "implication": {"countries": [{"code": "US", "p": 1.0}]}}])
    res = build_visual_layers(vc, BBOX, spec, tmp_path / "job3", settings=settings(tmp_path),
                              geocoder=g, overpass=o)
    assert res.provenance["layers"][0]["confidence"] <= 0.6
    assert res.provenance["layers"][0]["depth_nats"] < 1.0


def test_many_clues_cannot_dominate(tmp_path):
    spec = GridSpec.for_bbox(BBOX)
    _, g, o = clients(tmp_path)
    clues = [street_clue(conf=0.99, geocode_queries=[f"Bayview Avenue part {i}"]) for i in range(10)]
    res = build_visual_layers(make_vc(clues), BBOX, spec, tmp_path / "job", settings=settings(tmp_path),
                              geocoder=g, overpass=o)
    info = res.provenance["layers"]
    assert len(info) == 10
    assert all(x["depth_nats"] <= 2.31 for x in info)
    assert sum(x["depth_nats"] * x["weight"] for x in info) <= MAX_TOTAL_NATS + 1e-6
    assert all(x["confidence"] <= 0.85 for x in info)


def test_failures_degrade_not_raise(tmp_path):
    spec = GridSpec.for_bbox(BBOX)
    _, g, o = clients(tmp_path, Server(status=429))
    res = build_visual_layers(make_vc([street_clue()]), BBOX, spec, tmp_path / "job",
                              settings=settings(tmp_path), geocoder=g, overpass=o)
    assert res.layers == [] and res.status == "degraded" and res.provenance["layer_warnings"]


def test_skipped_clues_pass_through(tmp_path):
    spec = GridSpec.for_bbox(BBOX)
    vc = VisualClues(status="skipped", message="no key")
    res = build_visual_layers(vc, BBOX, spec, tmp_path / "job", settings=settings(tmp_path))
    assert res.status == "skipped" and res.layers == []


# ------------------------------------------------------------------------------------ sun adapter

def test_sun_text_parsing():
    f = parse_sun_text("Pole shadow is about 1.5 times the pole's height; shadow points north-east.")
    assert f["ratio"] == 1.5 and f["azimuth"] == 45
    assert parse_sun_text("sun elevation about 35 degrees")["elevation"] == 35
    assert parse_sun_text("shadow 3 m long, height 2 m")["ratio"] == 1.5
    assert parse_sun_text("long shadows, late afternoon") == {}
    assert parse_sun_text("the shadow falls toward the camera")["rel_azimuth"] == 180


def test_sun_adapter_to_sunclue():
    vc = make_vc([
        {"type": "sun_shadow", "value": "shadow/height ratio 2.0, shadow points south", "confidence": 0.9},
        {"type": "sun_shadow", "value": "short shadows", "confidence": 0.95},
    ])
    sc = sun_clue_from_clues(vc)
    assert sc is not None and sc.shadow_length_over_height == 2.0 and sc.shadow_azimuth_deg == 180
    assert sc.confidence <= 0.8 * 0.7 + 1e-9
    none = make_vc([{"type": "sun_shadow", "value": "short shadows", "confidence": 0.9}])
    assert sun_clue_from_clues(none) is None
    rel = make_vc([{"type": "sun_shadow", "value": "elevation 40 degrees, shadow away from the camera",
                    "confidence": 0.5}])
    assert sun_clue_from_clues(rel).shadow_rel_azimuth_deg is None  # needs a heading
    s2 = sun_clue_from_clues(rel, camera_heading_deg=[90.0])
    assert s2.shadow_rel_azimuth_deg == 0.0 and s2.sun_elevation_deg == 40


@pytest.mark.network
def test_live_nominatim_one_query(tmp_path):
    g = Geocoder.from_settings(settings(tmp_path))
    hits = g.search("Bayview Avenue, Oyster Bay", BBOX)
    assert hits and all(40.8 < h.lat < 41.0 for h in hits)
    area = g.admin_area("US")
    assert area is not None
