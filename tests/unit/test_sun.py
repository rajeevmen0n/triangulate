from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
from pvlib import solarposition

from triangulate.fusion.candidates import FusionCandidate
from triangulate.visual.sun import SunClue, predicted_sun, sun_logliks, sun_rerank

OB = (40.87, -73.53)  # Oyster Bay NY
FAR = (40.87 - 2.25, -73.53 - 5.1)  # ~500 km toward the sun (SW)
T = datetime(2026, 9, 8, 20, 45, tzinfo=UTC)


def cand(lat, lon, t=T):
    return FusionCandidate(rank=1, log_density=0.0, lat=lat, lon=lon, t_s=t, prob_mass=0.5)


def truth(t=T):
    el, az = predicted_sun(*OB, [t])
    return float(el[0]), float(az[0])


def test_matches_pvlib():
    el, az = truth()
    sp = solarposition.get_solarposition(pd.DatetimeIndex([T]), *OB)
    assert math.isclose(el, sp["apparent_elevation"].iloc[0])
    assert 20 < el < 35 and 220 < az < 260


def test_ratio_true_beats_far():
    el, _ = truth()
    clue = SunClue(shadow_length_over_height=1 / math.tan(math.radians(el)))
    a, b = sun_logliks([cand(*OB), cand(*FAR)], clue)
    assert a > b + 1
    out = sun_rerank([cand(*FAR), cand(*OB)], clue)
    assert abs(out[0].lat - OB[0]) < 1e-6 and "sun" in out[0].late


def test_azimuth_consistent_vs_inconsistent():
    _, az = truth()
    good = SunClue(shadow_azimuth_deg=(az + 180) % 360)
    bad = SunClue(shadow_azimuth_deg=az)  # shadow towards the sun
    c = cand(*OB)
    assert sun_logliks([c], good)[0] > sun_logliks([c], bad)[0] + 3


def test_relative_azimuth_heading_options_and_ambiguity():
    _, az = truth()
    shadow = (az + 180) % 360
    heading = 90.0
    c = cand(*OB)
    ok = SunClue(shadow_rel_azimuth_deg=shadow - heading, camera_heading_deg=heading)
    flip = SunClue(shadow_rel_azimuth_deg=shadow - heading, camera_heading_deg=[heading, heading + 180])
    unknown = SunClue(shadow_rel_azimuth_deg=shadow - heading)
    assert sun_logliks([c], ok)[0] > sun_logliks([c], flip)[0] > sun_logliks([c], SunClue(
        shadow_rel_azimuth_deg=shadow - heading, camera_heading_deg=heading + 180))[0]
    assert sun_logliks([c], unknown) == [None]


def test_low_confidence_neutral_not_veto():
    el, _ = truth()
    r = 1 / math.tan(math.radians(el))
    z = sun_logliks([cand(*OB), cand(*FAR)], SunClue(shadow_length_over_height=r, confidence=0.0))
    assert z[0] == z[1]
    hi = sun_logliks([cand(*OB), cand(*FAR)], SunClue(shadow_length_over_height=r, confidence=1.0))
    lo = sun_logliks([cand(*OB), cand(*FAR)], SunClue(shadow_length_over_height=r, confidence=0.3))
    assert (lo[0] - lo[1]) < (hi[0] - hi[1])
    assert all(math.isfinite(v) for v in hi + lo)


def test_unknown_time_skipped_and_window_marginalised():
    el, _ = truth()
    r = 1 / math.tan(math.radians(el))
    nt = cand(*OB, t=None)
    assert sun_logliks([nt], SunClue(shadow_length_over_height=r)) == [None]
    out = sun_rerank([nt], SunClue(shadow_length_over_height=r))
    assert out[0].late == {}
    w = (datetime(2026, 9, 8, 20, 15, tzinfo=UTC), datetime(2026, 9, 8, 21, 10, tzinfo=UTC))
    _, az = truth()
    clue = SunClue(shadow_length_over_height=r, shadow_azimuth_deg=(az + 180) % 360, time_window=w)
    a, b = sun_logliks([nt, cand(*FAR, t=None)], clue)
    assert a is not None and b is not None and a > b
    # window result is below the point-time peak (marginalising over time dilutes it)
    assert a < sun_logliks([cand(*OB)], clue)[0]
