from datetime import UTC, date, datetime, time

import pytest
from pydantic import ValidationError

from triangulate import USER_AGENT_BASE, cli
from triangulate.config import Settings
from triangulate.core import cache, geo, progress, timeutil
from triangulate.core.types import Constraints, Event, GridSpec, Range, TrackSet

OYSTER_BBOX = (40.50, -74.00, 41.20, -73.00)


def test_smoke_imports():
    import pvlib  # noqa: F401
    import tesserocr  # noqa: F401


def test_cli_help(capsys):
    assert cli.main([]) == 0
    assert "triangulate" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


# --- timeutil -----------------------------------------------------------------------------------
def test_ny_sept8_needs_two_utc_days():
    assert timeutil.local_date_to_utc_days(date(2026, 9, 8), "America/New_York") == [
        date(2026, 9, 8), date(2026, 9, 9)]


def test_utc_tz_single_day_and_tokyo():
    assert timeutil.local_date_to_utc_days(date(2026, 9, 8), "UTC") == [date(2026, 9, 8)]
    assert timeutil.local_date_to_utc_days(date(2026, 9, 8), "Asia/Tokyo") == [
        date(2026, 9, 7), date(2026, 9, 8)]


def test_local_window_to_utc():
    t0, t1 = timeutil.local_window_to_utc(
        date(2026, 9, 8), "America/New_York", time(15, 30), time(18, 30))
    assert t0 == datetime(2026, 9, 8, 19, 30, tzinfo=UTC)
    assert t1 == datetime(2026, 9, 8, 22, 30, tzinfo=UTC)


def test_dst_day_lengths():
    t0, t1 = timeutil.local_date_to_utc_interval(date(2026, 3, 8), "America/New_York")
    assert (t1 - t0).total_seconds() == 23 * 3600
    t0, t1 = timeutil.local_date_to_utc_interval(date(2026, 11, 1), "America/New_York")
    assert (t1 - t0).total_seconds() == 25 * 3600


def test_infer_tz():
    assert timeutil.infer_tz(40.8767, -73.5264) == "America/New_York"


# --- geo ----------------------------------------------------------------------------------------
def test_haversine_and_bbox():
    assert geo.haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    assert geo.parse_bbox("40.5,-74,41.2,-73") == OYSTER_BBOX
    with pytest.raises(ValueError):
        geo.parse_bbox("41,-74,40,-73")
    assert geo.bbox_contains(OYSTER_BBOX, 40.8767, -73.5264)
    h, w = geo.bbox_size_m(OYSTER_BBOX)
    assert 76_000 < h < 80_000 and 80_000 < w < 88_000


def test_laea_and_enu_roundtrip():
    lat0, lon0 = 40.85, -73.5
    x, y = geo.to_laea(40.9, -73.4, lat0, lon0)
    lat, lon = geo.from_laea(x, y, lat0, lon0)
    assert (lat, lon) == pytest.approx((40.9, -73.4), abs=1e-7)
    e, n, u = geo.geodetic_to_enu(40.9, -73.4, 1000.0, lat0, lon0, 0.0)
    lat, lon, h = geo.enu_to_geodetic(e, n, u, lat0, lon0, 0.0)
    assert (float(lat), float(lon), float(h)) == pytest.approx((40.9, -73.4, 1000.0), abs=1e-6)
    assert n > 0 and e > 0


# --- types --------------------------------------------------------------------------------------
def test_constraints_require_region_and_date():
    with pytest.raises(ValidationError):
        Constraints(date_start=date(2026, 9, 8))  # no bbox
    with pytest.raises(ValidationError):
        Constraints(bbox=OYSTER_BBOX)  # no date
    c = Constraints(bbox=OYSTER_BBOX, date_start=date(2026, 9, 8))
    assert c.tz is None and c.h_obs_m == 5.0
    with pytest.raises(ValidationError):
        Constraints(bbox=OYSTER_BBOX, date_start=date(2026, 9, 1), date_end=date(2026, 9, 30))


def test_event_and_trackset():
    r = Range(lo=1, hi=2)
    e = Event(id="e1", t_h=2152, t_h_range=r, f0=130, f0_range=r, v=140, v_range=r, d=5000, d_range=r)
    assert Event.model_validate_json(e.model_dump_json()) == e
    assert TrackSet.empty().tracks.columns[0] == "icao"


def test_gridspec_for_bbox():
    g = GridSpec.for_bbox(OYSTER_BBOX)
    nx, ny = g.shape(g.base_level)
    assert nx * ny <= 4_000_000
    assert g.cell_size(g.base_level) in (50, 100)
    cell = g.xy_to_cell(*g.cell_center_xy((g.base_level, 3, 4)), g.base_level)
    assert cell == (g.base_level, 3, 4)


# --- config / cache / progress ------------------------------------------------------------------
def test_user_agent_and_budget(monkeypatch, tmp_path):
    for k in ("TRIANGULATE_CONTACT", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.user_agent == USER_AGENT_BASE == "triangulate/0.1"
    assert s.llm_budget_usd == 2.0 and s.basemap_enabled and not s.llm_enabled
    monkeypatch.setenv("TRIANGULATE_CONTACT", "me@example.org")
    assert Settings(_env_file=None).user_agent == "triangulate/0.1 (me@example.org)"


def test_dotenv_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-test\nTRIANGULATE_BASEMAP_ENABLED=false\nOTHER=1\n")
    s = Settings(_env_file=env)
    assert s.llm_enabled and not s.basemap_enabled
    assert "sk-test" not in repr(s)


def test_stage_cache(tmp_path):
    k1 = cache.stage_key("spec", {"sha": "a"}, {"win": 2048})
    assert k1 == cache.stage_key("spec", {"sha": "a"}, {"win": 2048})
    assert k1 != cache.stage_key("spec", {"sha": "a"}, {"win": 1024})
    c = cache.StageCache(tmp_path)
    assert not c.has(k1) and c.get_json(k1) is None
    c.put_json(k1, {"x": 1})
    assert c.has(k1) and c.get_json(k1) == {"x": 1}


def test_progress():
    assert progress.overall_fraction("ingest", 0.0) == 0.0
    assert progress.overall_fraction("render", 1.0) == pytest.approx(1.0)
    a = progress.overall_fraction("adsb", 0.5)
    assert 0 < a < progress.overall_fraction("cpa", 0.0) + 1e-9
    progress.null_progress("x", 0.5, "m")
