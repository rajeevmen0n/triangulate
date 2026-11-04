import gzip
import json
import math

import numpy as np
import polars as pl

from triangulate.adsb import aircraft_types as at
from triangulate.adsb import tracks as tk
from triangulate.config import Settings
from triangulate.core.geo import enu_to_geodetic, geodetic_to_enu
from triangulate.core.types import AIRCRAFT_SCHEMA, TRACK_SCHEMA, TrackSet

ORIGIN = (40.88, -73.53)


def types():
    return at.load_types(Settings(data="/nonexistent-triangulate-test"))


def test_type_engine_codes():
    t = types()
    assert t.source == "doc8643-fallback"
    assert t.engine("C172") == "P" and t.classify("C172")[0]
    assert t.engine("B738") == "J" and not t.classify("B738")[0]
    assert t.engine("C208") == "T" and t.classify("C208")[0]
    assert t.lookup("EC35").is_helicopter
    assert t.classify("EC35")[0]
    assert not t.classify("EC35", exclude_helicopters=True)[0]


def test_missing_type_fallback():
    t = types()
    assert not t.classify(None, "A3", 300)[0]
    assert not t.classify("ZZZZ", "A5", 300)[0]
    assert t.classify(None, "A3", 200)[0]
    assert t.classify(None, "A1", 300)[0]
    assert t.classify(None, None, 300)[0]
    assert t.classify(None, "A2", 300)[0]


def test_tar1090_parse_and_manifest(tmp_path):
    doc = {"C172": ["CESSNA 172", "L1P", "L"], "XJ99": ["Test jet", "L2J", "M"]}
    raw = gzip.compress(json.dumps(doc).encode())

    class R:
        content = raw
        def raise_for_status(self): pass

    class C:
        def get(self, url): return R()

    s = Settings(data=tmp_path)
    p = at.download_types(s, client=C())
    assert p.exists()
    man = json.loads((p.parent / "manifest.json").read_text())
    assert man["licence_status"] == "unclear" and man["n_types"] == 2
    t = at.load_types(s)
    assert t.source == "tar1090-db+fallback" and t.engine("XJ99") == "J" and t.engine("B738") == "J"


def mk(icao, typ, cat, n=5, gs=100.0):
    rows = {k: [None] * n for k in TRACK_SCHEMA}
    rows.update(icao=[icao] * n, typecode=[typ] * n, category=[cat] * n, t=[float(i) for i in range(n)],
                lat=[40.9] * n, lon=[-73.5] * n, gs_kt=[gs] * n, on_ground=[False] * n)
    return pl.DataFrame(rows, schema=TRACK_SCHEMA)


def test_exclude_jets():
    tr = pl.concat([mk("a", "C172", "A1"), mk("b", "B738", "A3", gs=450), mk("c", None, "A3", gs=300),
                    mk("d", None, "A1", gs=90), mk("e", "EC35", "A7")])
    ac = pl.DataFrame({k: [None] * 5 for k in AIRCRAFT_SCHEMA}, schema=AIRCRAFT_SCHEMA).with_columns(
        pl.Series("icao", list("abcde")))
    ts = TrackSet(tracks=tr, aircraft=ac)
    cls = tk.classify_aircraft(ts, types())
    assert dict(zip(cls["icao"], cls["keep"], strict=True)) == dict(a=True, b=False, c=False, d=True, e=True)
    assert set(tk.exclude_jets(ts, types()).tracks["icao"]) == {"a", "d", "e"}
    assert set(tk.exclude_jets(ts, types(), exclude_helicopters=True).tracks["icao"]) == {"a", "d"}


def circle_frame(t, radius=3000.0, h=1000.0, geom=True, baro_off=0.0):
    w = 2 * math.pi / 600.0
    e, n = radius * np.cos(w * t), radius * np.sin(w * t)
    lat, lon, _ = enu_to_geodetic(e, n, np.full_like(t, h), *ORIGIN)
    N = len(t)
    d = {k: [None] * N for k in TRACK_SCHEMA}
    d.update(icao=["x"] * N, t=list(t), lat=list(lat), lon=list(lon), on_ground=[False] * N,
             gs_kt=[radius * w / tk.KT_MS] * N)
    ft = h / tk.FT_M
    d["alt_geom_ft"] = [ft] * N if geom else [None] * N
    d["alt_baro_ft"] = [ft - baro_off / tk.FT_M] * N
    return pl.DataFrame(d, schema=TRACK_SCHEMA)


def test_pchip_circle_within_5m():
    t = np.arange(0.0, 300.0, 5.0)
    out = tk.prepare_tracks(circle_frame(t), ORIGIN)
    assert out["t"].to_list() == [float(i) for i in range(0, 296)]
    tt = out["t"].to_numpy()
    w = 2 * math.pi / 600.0
    err = np.hypot(out["e"].to_numpy() - 3000 * np.cos(w * tt), out["n"].to_numpy() - 3000 * np.sin(w * tt))
    assert err.max() < 5.0
    assert np.abs(out["u"].to_numpy() - 1000.0).max() < 1.0  # ENU u: ellipsoid height about h0=0
    assert set(out["alt_src"]) == {"geom"}


def test_gap_split_and_ground_dropped():
    t = np.concatenate([np.arange(0, 100, 5.0), np.arange(200, 300, 5.0)])
    out = tk.prepare_tracks(circle_frame(t), ORIGIN)
    assert sorted(out["seg"].unique()) == [0, 1]
    assert out.filter(pl.col("seg") == 0)["t"].max() == 95.0
    assert out.filter(pl.col("seg") == 1)["t"].min() == 200.0
    # gap of exactly 90 s is not split
    t2 = np.array([0.0, 90.0, 95.0])
    assert tk.prepare_tracks(circle_frame(t2), ORIGIN)["seg"].n_unique() == 1
    g = circle_frame(np.arange(0, 20.0, 5.0)).with_columns(pl.lit(True).alias("on_ground"))
    assert tk.prepare_tracks(g, ORIGIN).height == 0


def test_altitude_precedence():
    t = np.arange(0.0, 60.0, 5.0)
    base = circle_frame(t, geom=False, baro_off=-30.0)  # baro lower than 1000 m by... only baro available
    out = tk.prepare_tracks(base, ORIGIN)
    assert set(out["alt_src"]) == {"baro"} and set(out["alt_sigma_m"]) == {tk.SIGMA_BARO_M}
    # mixed: geom at first half, baro-only after; baro is shifted by the median geom-baro offset
    f = circle_frame(t, baro_off=33.0).with_columns(
        pl.when(pl.col("t") < 30).then(pl.col("alt_geom_ft")).otherwise(None).alias("alt_geom_ft"))
    out = tk.prepare_tracks(f, ORIGIN)
    assert abs(out["u"].to_numpy() - 1000.0).max() < 1.0
    assert out.filter(pl.col("t") >= 40)["alt_src"].unique().to_list() == ["baro"]
    assert out.filter(pl.col("t") <= 20)["alt_src"].unique().to_list() == ["geom"]
    e, n, u = geodetic_to_enu(40.9, -73.5, 0.0, *ORIGIN)
    assert u < 0  # sanity: ENU helper is used with ellipsoid height
