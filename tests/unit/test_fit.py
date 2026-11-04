"""WP08 curve fitting, ranges and traced polylines."""

import json
import math

import numpy as np
import pytest
from wp08_helpers import errors, make_scene

from triangulate.acoustic.detect import Detection
from triangulate.acoustic.fit import (
    FitConfig,
    event_from_detection,
    fit_params,
    fit_trace,
    model_curve,
    parse_line_points,
    sweep_coverage,
)
from triangulate.acoustic.physics import cpa_slope, straight_line_freq
from triangulate.acoustic.spectrogram import compute_spectrogram


@pytest.fixture(scope="module")
def one_pass():
    sc = make_scene(300, [dict(t_h=150.0, v=110.0, d=2500.0, f0=140.0, snr_db=0.0)], seed=3)
    return sc, compute_spectrogram(sc.audio)


def test_model_curve_matches_physics():
    class E:
        f0, v, d, t_h = 130.0, 117.0, 4280.0, 100.0

    t = np.linspace(0, 200, 11)
    assert np.allclose(model_curve(E, t), straight_line_freq(t, 130.0, 117.0, 4280.0, 100.0))


def test_fit_from_coarse_guess(one_pass):
    sc, spec = one_pass
    tr = sc.truth[0]
    # a template-grid-like guess: v off by 5 m/s, d off by 25 %, t_h by 0.5 s, f0 by 1 %
    fr = fit_params(spec, tr.f0 * 1.01, tr.v - 5, tr.d * 1.25, tr.tau_h + 0.5)
    assert fr.ok and 1 in fr.harmonics and 2 in fr.harmonics
    assert abs(fr.t_h - tr.tau_h) < 0.5
    assert abs(fr.v / tr.v - 1) < 0.03
    assert abs(fr.d / tr.d - 1) < 0.10
    assert abs(fr.f0 / tr.f0 - 1) < 0.003
    assert sweep_coverage(fr.points, fr.params, 343.0) >= 0.75
    assert fr.rms_hz < 1.5


def test_event_ranges_contain_truth_and_minimum_margins(one_pass):
    sc, spec = one_pass
    tr = sc.truth[0]
    det = Detection(t_h=tr.tau_h + 1.0, f0=tr.f0 * 0.99, v=tr.v + 5, d=tr.d * 0.8, snr=10.0)
    ev = event_from_detection(spec, det, FitConfig())
    er = errors(ev, tr)
    assert er["inside"], ev
    assert ev.v_range.lo <= 0.9 * ev.v + 1e-6 and ev.v_range.hi >= 1.1 * ev.v - 1e-6
    assert ev.d_range.lo <= 0.7 * ev.d + 1e-3 and ev.d_range.hi >= 1.3 * ev.d - 1e-3
    assert ev.t_h_range.hi - ev.t_h_range.lo >= 6.0 - 1e-6
    assert ev.f0_range.lo <= 0.97 * ev.f0 + 1e-6
    assert ev.ridge_points and all(len(p) == 2 for p in ev.ridge_points)
    # ridge points are reported as fundamental frequency (f_obs / h)
    rp = np.array(ev.ridge_points)
    assert np.median(np.abs(rp[:, 1] - model_curve(ev, rp[:, 0]))) < 1.0


def test_fit_failure_is_flagged():
    sc = make_scene(200, [], seed=1)
    spec = compute_spectrogram(sc.audio)
    ev = event_from_detection(spec, Detection(t_h=100.0, f0=150.0, v=100.0, d=2000.0, snr=7.0))
    assert "fit-failed" in ev.quality_flags or "sweep-gap" in ev.quality_flags


def _trace_points(f0=480.0, v=90.0, d=1100.0, t_h=1276.0, step=2.0, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(t_h - 30, t_h + 30, step)
    f = straight_line_freq(t, f0, v, d, t_h) + rng.normal(0, jitter, t.size)
    return [[float(a), float(b)] for a, b in zip(t, f, strict=True)]


def test_parse_line_points_formats(tmp_path):
    pts = _trace_points()
    a = parse_line_points(pts)
    assert a.shape == (len(pts), 2)
    assert np.allclose(parse_line_points({"LINE_POINTS": pts[::-1]}), a)  # sorted by time
    assert np.allclose(parse_line_points(json.dumps({"points": [{"t": x, "f": y} for x, y in pts]})), a)
    assert np.allclose(parse_line_points("\n".join(f"{x},{y}" for x, y in pts)), a)
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"LINE_POINTS": pts}))
    assert np.allclose(parse_line_points(p), a)
    assert np.allclose(parse_line_points(str(p)), a)
    with pytest.raises(ValueError):
        parse_line_points([[0, 100], [1, 99]])
    with pytest.raises(ValueError):
        parse_line_points([[0, 100], [1, -5], [2, 90]])


def test_fit_trace_recovers_curve():
    pts = _trace_points(jitter=0.5)
    ev = fit_trace({"LINE_POINTS": pts})
    assert "traced" in ev.quality_flags and ev.traced_line
    assert abs(ev.t_h - 1276.0) < 1.0
    assert abs(ev.f0 / 480.0 - 1) < 0.01
    # a 60-s trace does not reach the asymptotes, so v/d are looser; ranges must still contain truth
    assert ev.v_range.contains(90.0) and ev.d_range.contains(1100.0)
    assert ev.t_h_range.contains(1276.0) and ev.f0_range.contains(480.0)
    # the slope at t_h is what the trace pins down (plan 3.1, cpa_slope)
    s_fit, s_true = float(cpa_slope(ev.f0, ev.v, ev.d)), float(cpa_slope(480.0, 90.0, 1100.0))
    assert math.isclose(s_fit, s_true, rel_tol=0.1)
