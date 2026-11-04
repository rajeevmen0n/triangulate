"""WP06 physics: video numbers, closed form vs independent root finding, track solver, CPA, slopes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq

from triangulate.acoustic import physics as ph

C = 343.0


def _brute_emission(t_r: float, pos_fn, obs, c: float = C) -> float:
    """Independent reference: Brent root of t_e + |P(t_e) - X|/c - t_r on an analytic position fn."""
    obs = np.asarray(obs, float)

    def F(t):
        return t + float(np.linalg.norm(pos_fn(t) - obs)) / c - t_r

    hi = t_r
    lo = t_r - 4.0 * (F(t_r) + 1000.0)
    return brentq(F, lo, hi, xtol=1e-12, rtol=1e-15, maxiter=500)


def _straight(v, d_h, h, t_c=0.0):
    """Analytic straight line along +x, CPA at (0, d_h, h) at time t_c; observer at origin."""
    p_c = np.array([0.0, d_h, h])
    u = np.array([v, 0.0, 0.0])
    return lambda t: p_c + u * (t - t_c)


def _track_from_fn(pos_fn, vel_fn, t0, t1, dt=1.0):
    t = np.arange(t0, t1 + 0.5 * dt, dt)
    return ph.Track(t=t, pos=np.array([pos_fn(x) for x in t]), vel=np.array([vel_fn(x) for x in t]))


# --- Numbers from the video (plan 3.1) -----------------------------------------------------------


def test_video_doppler_numbers():
    assert ph.doppler_shift(130.0, 121.16) == pytest.approx(96.07, abs=0.005)
    assert ph.doppler_shift(130.0, 62.7) == pytest.approx(109.9, abs=0.05)  # naive (wrong) model
    # receding lowers, approaching raises
    assert ph.doppler_shift(100.0, 50.0) < 100.0 < ph.doppler_shift(100.0, -50.0)


def test_video_slope_readout():
    # "max RoC" readout of the slider tool = slope at the heard CPA, f0 v^2 / (c d)
    assert abs(ph.cpa_slope(131.5, 117.0, 4280.0)) == pytest.approx(1.222, rel=0.02)
    assert abs(ph.cpa_slope(130.0, 107.0, 3230.0)) == pytest.approx(1.343, rel=0.002)
    assert ph.cpa_slope(131.5, 117.0, 4280.0) < 0  # falling curve


def test_video_asymptotes():
    assert np.allclose(ph.asymptotes(131.5, 117.0), (199.58, 98.05), atol=0.01)
    assert np.allclose(ph.asymptotes(130.0, 107.0), (188.94, 99.09), atol=0.01)


def test_video_worked_example_is_retarded_consistent():
    # The video's delayed-mode numbers satisfy t_r = t_e + r/c: 16.910 + 2775/343 = 25.000
    assert 16.910 + 2775.0 / C == pytest.approx(25.0, abs=2e-3)


def test_plan_closed_form_sanity_example():
    # plan 3.1: v = 140, d = sqrt(980^2 + 1020^2), t_r = 25 s after the emission CPA -> 16.95 s, 96.26 Hz
    d = math.hypot(980.0, 1020.0)
    s = ph.straight_line_state(25.0, 140.0, d, ph.heard_time(0.0, d), C)
    assert float(s.s) == pytest.approx(16.95, abs=0.005)
    assert 130.0 * float(s.g) == pytest.approx(96.26, abs=0.01)
    # brute force on the same geometry
    te = _brute_emission(25.0, _straight(140.0, 980.0, 1020.0), np.zeros(3))
    assert float(s.s) == pytest.approx(te, abs=1e-9)


def test_speed_of_sound():
    assert ph.speed_of_sound() == 343.0
    assert ph.speed_of_sound(20.0) == pytest.approx(343.2, abs=0.1)
    assert ph.speed_of_sound(0.0) == pytest.approx(331.3)


# --- Closed form ----------------------------------------------------------------------------------


@pytest.mark.parametrize("v,d", [(60.0, 300.0), (117.0, 4280.0), (160.0, 12000.0), (25.0, 30.0)])
def test_closed_form_matches_brute_force(v, d):
    h = 0.6 * d
    pos = _straight(v, math.sqrt(d * d - h * h), h, t_c=10.0)
    t_h = ph.heard_time(10.0, d)
    for t_r in np.linspace(t_h - 200, t_h + 200, 41):
        te = _brute_emission(t_r, pos, np.zeros(3))
        st_ = ph.straight_line_state(t_r, v, d, t_h)
        assert float(st_.s) + 10.0 == pytest.approx(te, abs=1e-8)
        p = pos(te)
        r = float(np.linalg.norm(p))
        assert float(st_.r) == pytest.approx(r, rel=1e-10)
        assert float(st_.v_r) == pytest.approx(v * p[0] / r, abs=1e-8)


def test_closed_form_cpa_point():
    v, d = 120.0, 2500.0
    t_h = 100.0
    st_ = ph.straight_line_state(t_h, v, d, t_h)
    assert abs(float(st_.s)) < 1e-12 and abs(float(st_.v_r)) < 1e-10
    assert float(ph.straight_line_freq(t_h, 150.0, v, d, t_h)) == pytest.approx(150.0, abs=1e-9)
    # monotone falling and bounded by the asymptotes
    t = np.linspace(-2000, 2000, 4001) + t_h
    f = ph.straight_line_freq(t, 150.0, v, d, t_h)
    fpre, fpost = ph.asymptotes(150.0, v)
    assert np.all(np.diff(f) < 0) and f.max() < fpre and f.min() > fpost


def test_closed_form_broadcasts():
    t = np.linspace(-50, 50, 11)[:, None, None]
    v = np.array([80.0, 120.0])[None, :, None]
    d = np.array([500.0, 1000.0, 4000.0])[None, None, :]
    f = ph.straight_line_freq(t, 130.0, v, d, 0.0)
    assert f.shape == (11, 2, 3)
    assert f[5, 1, 2] == pytest.approx(130.0)


def test_slope_analytic_and_maximum():
    f0, v, d, t_h = 131.5, 117.0, 4280.0, 50.0
    t = np.linspace(t_h - 300, t_h + 300, 600001)
    f = ph.straight_line_freq(t, f0, v, d, t_h)
    num = np.gradient(f, t)
    ana = ph.straight_line_slope(t, f0, v, d, t_h)
    assert np.max(np.abs(num - ana)) < 1e-6
    # slope at t_h
    slope_th = float(ph.straight_line_slope(t_h, f0, v, d, t_h))
    assert slope_th == pytest.approx(ph.cpa_slope(f0, v, d), rel=1e-12)
    # true maximum: value and location t_r = t_c = t_h - d/c
    k = np.argmax(np.abs(num))
    assert abs(num[k]) == pytest.approx(ph.max_slope(f0, v, d), rel=1e-6)
    assert t[k] == pytest.approx(t_h - d / C, abs=2e-3)
    assert ph.max_slope(f0, v, d) == pytest.approx(abs(ph.cpa_slope(f0, v, d)) / (1 - v**2 / C**2) ** 1.5)


def test_inversions():
    f0, v, d = 128.0, 97.0, 3100.0
    fpre, fpost = ph.asymptotes(f0, v)
    f0b, vb = ph.from_asymptotes(fpre, fpost)
    assert f0b == pytest.approx(f0) and vb == pytest.approx(v)
    assert ph.distance_from_slope(f0, v, ph.cpa_slope(f0, v, d)) == pytest.approx(d)
    assert ph.emission_cpa_time(ph.heard_time(12.0, d), d) == pytest.approx(12.0)


@pytest.mark.parametrize("frac", [0.5, 0.9, 0.95])
def test_asymptote_time(frac):
    f0, v, d, t_h = 140.0, 130.0, 5000.0, 0.0
    before, after = ph.asymptote_time(frac, v, d)
    assert before > 0 and after > 0
    fpre, fpost = ph.asymptotes(f0, v)
    assert float(ph.straight_line_freq(t_h + after, f0, v, d, t_h)) == pytest.approx(
        f0 + frac * (fpost - f0), abs=1e-9)
    assert float(ph.straight_line_freq(t_h - before, f0, v, d, t_h)) == pytest.approx(
        f0 + frac * (fpre - f0), abs=1e-9)


# --- Track-based retarded solver --------------------------------------------------------------


def test_solver_matches_closed_form_on_straight_track():
    v, d_h, h = 140.0, 980.0, 1020.0
    d = math.hypot(d_h, h)
    pos = _straight(v, d_h, h, t_c=0.0)
    track = _track_from_fn(pos, lambda t: np.array([v, 0.0, 0.0]), -400, 400)
    t_r = np.linspace(-150, 150, 3001) + d / C
    em = ph.solve_emission(t_r, track, np.zeros(3))
    f_cf = ph.straight_line_freq(t_r, 130.0, v, d, d / C)
    assert np.max(np.abs(em.freq(130.0) - f_cf)) < 0.01  # plan acceptance; actually ~1e-11
    assert np.max(np.abs(em.freq(130.0) - f_cf)) < 1e-8
    assert em.valid.all()
    assert em.iterations.max() <= 6
    # the solution satisfies the retarded equation
    assert np.max(np.abs(em.t_e + em.r / C - t_r)) < 1e-9


def test_solver_heard_cpa_time():
    """t_h = t_c + d/c within 1 ms, found on the solver output (v_r = 0 crossing in reception time)."""
    v, d_h, h, t_c = 95.0, 1500.0, 800.0, 1_788_907_598.0  # epoch-scale time base
    d = math.hypot(d_h, h)
    pos = _straight(v, d_h, h, t_c=t_c)
    track = _track_from_fn(pos, lambda t: np.array([v, 0.0, 0.0]), t_c - 300, t_c + 300)

    def vr_at(t_r):
        return float(ph.solve_emission(np.array([t_r]), track, np.zeros(3)).v_r[0])

    t_zero = brentq(vr_at, t_c - 20, t_c + 40, xtol=1e-7)
    assert t_zero == pytest.approx(t_c + d / C, abs=1e-3)
    assert ph.heard_time(t_c, d) == pytest.approx(t_zero, abs=1e-3)


def test_solver_on_circle_vs_brute_force():
    """Curved track (standard-rate-ish turn), analytic positions; solver uses 1-s Hermite samples."""
    R, v = 1200.0, 60.0
    w = v / R
    ce = np.array([2000.0, -500.0, 900.0])

    def pos(t):
        return ce + np.array([R * math.cos(w * t), R * math.sin(w * t), 0.0])

    def vel(t):
        return np.array([-R * w * math.sin(w * t), R * w * math.cos(w * t), 0.0])

    track = _track_from_fn(pos, vel, 0.0, 400.0)
    obs = np.array([150.0, 80.0, 3.0])
    t_r = np.linspace(20.0, 380.0, 721)
    em = ph.solve_emission(t_r, track, obs)
    te_ref = np.array([_brute_emission(x, pos, obs) for x in t_r])
    assert np.max(np.abs(em.t_e - te_ref)) < 1e-4
    vr_ref = np.array([ph.radial_velocity(pos(t), vel(t), obs) for t in te_ref])
    assert np.max(np.abs(em.v_r - vr_ref)) < 5e-3
    # also with finite-difference node velocities
    tr2 = ph.Track.from_positions(track.t, track.pos)
    em2 = ph.solve_emission(t_r, tr2, obs)
    assert np.max(np.abs(em2.freq(120.0) - em.freq(120.0))) < 0.01


def test_solver_batch_observers_and_shapes():
    rng = np.random.default_rng(3)
    track = _track_from_fn(_straight(110.0, 0.0, 0.0), lambda t: np.array([110.0, 0.0, 0.0]), -600, 600)
    obs = rng.normal(size=(7, 3)) * [3000.0, 3000.0, 10.0]
    t_r = np.linspace(-60, 60, 50).reshape(5, 10)
    em = ph.solve_emission(t_r, track, obs)
    assert em.t_e.shape == (7, 5, 10)
    for k in range(7):
        one = ph.solve_emission(t_r, track, obs[k])
        assert one.t_e.shape == (5, 10)
        np.testing.assert_allclose(em.t_e[k], one.t_e, rtol=0, atol=1e-12)
        np.testing.assert_allclose(em.v_r[k], one.v_r, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        ph.solve_emission(t_r, track, np.zeros(2))


def test_solver_valid_mask_outside_track():
    track = _track_from_fn(_straight(100.0, 1000.0, 500.0), lambda t: np.array([100.0, 0.0, 0.0]), 0, 100)
    em = ph.solve_emission(np.array([-50.0, 50.0, 300.0]), track, np.zeros(3))
    assert em.valid.tolist() == [False, True, False]
    # extrapolation is linear, so the closed form still holds outside the span
    d = math.hypot(1000.0, 500.0)
    f_cf = ph.straight_line_freq(np.array([-50.0, 50.0, 300.0]), 100.0, 100.0, d, d / C)
    np.testing.assert_allclose(em.freq(100.0), f_cf, atol=1e-8)


def test_track_validation_and_state():
    with pytest.raises(ValueError):
        ph.Track(t=np.array([0.0, 0.0]), pos=np.zeros((2, 3)), vel=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ph.Track(t=np.array([0.0, 1.0]), pos=np.zeros((2, 2)), vel=np.zeros((2, 2)))
    track = _track_from_fn(_straight(50.0, 0.0, 0.0), lambda t: np.array([50.0, 0.0, 0.0]), 0, 10)
    p, v = track.state(np.array([[2.5, 7.25]]))
    assert p.shape == (1, 2, 3) and v.shape == (1, 2, 3)
    np.testing.assert_allclose(p[0, :, 0], [125.0, 362.5])
    np.testing.assert_allclose(v[0, :, 0], [50.0, 50.0])


@settings(max_examples=60, deadline=None)
@given(
    speed_frac=st.floats(0.0, 0.499),
    heading=st.floats(0.0, 2 * math.pi),
    climb_frac=st.floats(-0.3, 0.3),
    ox=st.floats(-5000, 5000), oy=st.floats(-5000, 5000), oz=st.floats(-2000, 2000),
    t_r=st.floats(-500, 500),
)
def test_property_straight_solver_converges(speed_frac, heading, climb_frac, ox, oy, oz, t_r):
    """For any straight track with |v| < 0.5c, Newton converges and matches the closed form."""
    v = speed_frac * C
    vz = climb_frac * v
    vh = math.sqrt(max(v * v - vz * vz, 0.0))
    vel = np.array([vh * math.cos(heading), vh * math.sin(heading), vz])
    p0 = np.array([0.0, 0.0, 1000.0])
    t = np.linspace(-1000.0, 1000.0, 41)
    track = ph.Track(t=t, pos=p0 + np.outer(t, vel), vel=np.tile(vel, (t.size, 1)))
    obs = np.array([ox, oy, oz])
    em = ph.solve_emission(np.array([t_r]), track, obs)
    assert em.iterations[0] < 50
    assert abs(em.t_e[0] + em.r[0] / C - t_r) < 1e-7
    # closed form: CPA of the infinite line
    rel = obs - p0
    if v > 1e-6:
        t_c = float(rel @ vel) / (v * v)
        d = float(np.linalg.norm(p0 + vel * t_c - obs))
        if d > 1.0:
            g = ph.straight_line_factor(t_r, v, d, t_c + d / C)
            assert float(em.g[0]) == pytest.approx(float(g), abs=1e-9)


@settings(max_examples=40, deadline=None)
@given(
    amp=st.floats(0.0, 3000.0), period=st.floats(30.0, 600.0), phase=st.floats(0.0, 2 * math.pi),
    v_frac=st.floats(0.0, 0.45), dt=st.floats(0.5, 5.0), seed=st.integers(0, 10_000),
)
def test_property_curved_solver_converges(amp, period, phase, v_frac, dt, seed):
    """Wiggly tracks with speed < 0.5c: converged residual, and agreement with Brent on the interpolant."""
    w = 2 * math.pi / period
    rng = np.random.default_rng(seed)
    dirn = rng.normal(size=3)
    dirn /= np.linalg.norm(dirn)
    a = min(amp, 0.45 * C / (2 * w))  # wiggle speed <= 0.225c
    base = v_frac * C * 0.5 * dirn  # <= 0.225c
    t = np.arange(0.0, 900.0 + dt, dt)
    ph1, ph2 = w * t + phase, w * t
    wig = np.stack([a * np.sin(ph1), a * np.cos(ph2), 0.2 * a * np.sin(ph2)], 1)
    dwig = np.stack([a * w * np.cos(ph1), -a * w * np.sin(ph2), 0.2 * a * w * np.cos(ph2)], 1)
    pos = np.outer(t, base) + wig
    vel = base + dwig
    track = ph.Track(t=t, pos=pos, vel=vel)
    assert track.speed.max() < 0.5 * C
    obs = rng.normal(size=3) * 2000.0
    t_r = np.sort(rng.uniform(50.0, 850.0, 25))
    em = ph.solve_emission(t_r, track, obs)
    assert np.all(em.iterations < 50)
    np.testing.assert_allclose(em.t_e + em.r / C, t_r, atol=1e-7)

    def pos_fn(x):
        return track.state(np.array([x]))[0][0]

    for j in (0, 12, 24):
        assert em.t_e[j] == pytest.approx(_brute_emission(t_r[j], pos_fn, obs), abs=1e-6)


# --- Reference CPA finder ----------------------------------------------------------------------


def test_track_passes_straight():
    v, d_h, h, t_c = 120.0, 2000.0, 1200.0, 500.0
    track = _track_from_fn(_straight(v, d_h, h, t_c), lambda t: np.array([v, 0.0, 0.0]), 200, 800)
    (p,) = ph.track_passes(track, np.zeros(3))
    d = math.hypot(d_h, h)
    assert p.t_cpa == pytest.approx(t_c, abs=1e-6)
    assert p.d == pytest.approx(d, abs=1e-6)
    assert p.v == pytest.approx(v)
    assert p.t_h == pytest.approx(t_c + d / C, abs=1e-6)
    assert ph.track_passes(track, np.zeros(3), r_max=1000.0) == []


def test_track_passes_racetrack_multiple():
    R, v = 1500.0, 50.0
    w = v / R
    ce = np.array([0.0, 0.0, 600.0])
    period = 2 * math.pi / w

    def pos(t):
        return ce + np.array([R * math.cos(w * t), R * math.sin(w * t), 0.0])

    def vel(t):
        return np.array([-R * w * math.sin(w * t), R * w * math.cos(w * t), 0.0])

    track = _track_from_fn(pos, vel, -0.25 * period, 2.25 * period)
    obs = np.array([3000.0, 0.0, 0.0])  # outside the circle, nearest at angle 0
    passes = ph.track_passes(track, obs)
    assert len(passes) == 3  # t = 0, period, 2*period
    for k, p in enumerate(passes):
        assert p.t_cpa == pytest.approx(k * period, abs=1e-3)
        assert p.d == pytest.approx(math.hypot(1500.0, 600.0), abs=1e-3)
