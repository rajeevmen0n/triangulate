# ruff: noqa: E501
import gzip

import numpy as np
import orjson

from triangulate.adsb import readsb_trace as rt


def make_doc(icao="a1b2c3", ts=1788825600, rows=None, **kw):
    doc = {"icao": icao, "r": "N123AB", "t": "C172", "desc": "CESSNA 172", "dbFlags": 0,
           "ownOp": "ACME", "year": "1999", "version": "readsb-x", "timestamp": ts,
           "trace": rows if rows is not None else []}
    doc.update(kw)
    return doc


def gz(doc) -> bytes:
    return gzip.compress(orjson.dumps(doc))


ROWS = [
    [0.0, 40.0, -73.0, "ground", 0.0, 90.0, 2, None, {"category": "A1", "flight": "N123AB "}, "adsb_icao", None, None, None, None],
    [5.5, 40.001, -73.001, 1200, 80.0, 91.0, 8, 64, None, "adsb_icao", 1250, 80, 70, 1.5],
    [11.0, 40.002, -73.002, 1300, 81.0, 92.0, 1, None, None, "mlat", None, 128, None, None],
    [20.0, None, None, None, None, None, 0],  # short row with nulls
]


def test_decode_rows_exact():
    ac, df = rt.decode_trace(gz(make_doc(rows=ROWS)), with_flags=True)
    assert df.height == 4
    assert df["t"].to_list() == [1788825600.0, 1788825605.5, 1788825611.0, 1788825620.0]
    assert df["on_ground"].to_list() == [True, False, False, False]
    assert df["alt_baro_ft"].to_list() == [None, 1200.0, 1300.0, None]
    assert df["alt_geom_ft"].to_list() == [None, 1250.0, None, None]
    assert df["vrate_fpm"].to_list() == [None, 64.0, 128.0, None]  # falls back to geom rate
    assert df["flags"].to_list() == [2, 8, 1, 0]
    assert df["source"].to_list() == ["adsb_icao", "adsb_icao", "mlat", None]
    assert df["category"].to_list() == ["A1"] * 4  # propagated forward
    assert df["lat"].to_list()[3] is None
    assert ac["reg"] == "N123AB" and ac["typecode"] == "C172" and ac["category"] == "A1"
    assert ac["icao"] == "a1b2c3"


def test_tisb_icao_and_plain_json():
    doc = make_doc(icao="~1a2b3c", rows=ROWS[:2])
    ac, df = rt.decode_trace(orjson.dumps(doc))  # non-gz also accepted
    assert ac["icao"] == "~1a2b3c" and df["icao"][0] == "~1a2b3c"


def test_flag_constants():
    assert (rt.FLAG_STALE, rt.FLAG_NEW_LEG, rt.FLAG_GEOM_VRATE, rt.FLAG_GEOM_ALT) == (1, 2, 4, 8)


def test_summary_and_cells():
    ds = 1788825600.0
    s = rt.summarize_trace(gz(make_doc(rows=ROWS)), ds)
    assert s["n_points"] == 3 and s["tmin"] == ds and s["tmax"] == ds + 11
    assert s["lat_min"] == 40.0 and s["lon_max"] == -73.0
    qc = rt.query_cells((39.9, -73.1, 40.1, -72.9), ds, ds + 60, ds)
    assert np.isin(s["cells"], qc).any()
    qc2 = rt.query_cells((10, 10, 11, 11), ds, ds + 60, ds)
    assert not np.isin(s["cells"], qc2).any()
    qc3 = rt.query_cells((39.9, -73.1, 40.1, -72.9), ds + 7200, ds + 8000, ds)
    assert not np.isin(s["cells"], qc3).any()
    assert rt.summarize_trace(b"garbage", ds) is None
