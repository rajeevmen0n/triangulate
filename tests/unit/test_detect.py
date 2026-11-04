"""WP08 event detection (template bank, contrast, explain-away) on synthetic scenes + audio.mp3 smoke."""

import time
from pathlib import Path

import numpy as np
import pytest
from wp08_helpers import errors, make_scene, match, random_pass

from triangulate.acoustic.detect import (
    BAD_FLAGS,
    DetectConfig,
    build_bank,
    find_events,
    harmonic_stack,
)
from triangulate.acoustic.spectrogram import compute_spectrogram

ROOT = Path(__file__).resolve().parents[2]
DEBUG = ROOT / "data" / "debug"

AIRCRAFT = [
    dict(label="a1", t_h=160.0, v=130.0, d=4500.0, f0=120.0, snr_db=-6.0, heading=30.0),
    dict(label="a2", t_h=470.0, v=70.0, d=1200.0, f0=180.0, snr_db=0.0, heading=200.0),
    dict(label="a3", t_h=760.0, v=100.0, d=2800.0, f0=95.0, snr_db=6.0, heading=120.0,
         harmonics=(1.0, 0.8, 0.5)),
]
CAR = dict(label="car", t_h=320.0, v=28.0, d=25.0, f0=110.0, snr_db=6.0, height=1.0, vehicle=True,
           harmonics=(1.0, 0.5), half_span=120.0)


@pytest.fixture(scope="module")
def bank():
    return build_bank()


def _usable(e):
    return not set(BAD_FLAGS) & set(e.quality_flags)


def _check_scene(events, truth, tol=1.0):
    found = {}
    for tr in truth:
        if tr.vehicle:
            continue
        e = match(events, tr)
        assert e is not None, f"{tr.label} not detected: {[(round(x.t_h), round(x.f0)) for x in events]}"
        er = errors(e, tr)
        assert er["t"] <= 2.0 * tol, (tr, e)
        assert er["v"] <= 0.07 * tol, (tr, e)
        assert er["d"] <= 0.25 * tol, (tr, e)
        assert er["f0"] <= 0.01 * tol, (tr, e)
        assert er["inside"], (tr, e)
        assert _usable(e), e.quality_flags
        found[tr.label] = e
    # no false positives: every usable event with SNR > 6 is one of the aircraft
    ids = {id(e) for e in found.values()}
    fps = [e for e in events if e.snr > 6 and _usable(e) and id(e) not in ids]
    assert not fps, [(e.t_h, e.f0, e.v, e.d, e.snr, e.quality_flags) for e in fps]
    # the car is never reported as an aircraft (and if found, it is labelled)
    car = [tr for tr in truth if tr.vehicle]
    for tr in car:
        near = [e for e in events if abs(e.t_h - tr.tau_h) <= 15]
        assert all("non-aircraft" in e.quality_flags for e in near), near
    return found


def test_bank_layout(bank):
    cfg = DetectConfig()
    assert len(bank) == len(cfg.v_grid) * cfg.n_d + len(cfg.vehicle_v) * len(cfg.vehicle_d) == 468
    for tp in bank.templates[:: 37]:
        assert abs(np.linalg.norm(tp.A) - 1) < 1e-4
        assert np.allclose(tp.A.sum(axis=1), 0, atol=1e-4)  # zero-mean columns
        assert tp.t_rel[tp.a0] == 0.0
        # the ridge passes through the anchor (ln g(t_h) = 0) and falls with time
        assert abs(tp.lg[tp.a0]) < 1e-9 and tp.lg[0] > 0 > tp.lg[-1]
    a = [tp for tp in bank.templates if not tp.vehicle]
    assert min(-tp.t_rel[0] for tp in a) >= cfg.half_min - cfg.dt
    assert max(tp.t_rel[-1] for tp in a) <= cfg.half_max + cfg.dt


def test_harmonic_stack():
    du = 0.004
    Z = np.zeros((3, 1000), np.float32)
    k2 = round(np.log(2) / du)
    Z[:, 300] = 1.0
    Z[:, 300 + k2] = 1.0
    H = harmonic_stack(Z, du, (1.0, 0.6, 0.4))
    assert H[0, 300] == pytest.approx(1.6)
    assert H[0, 300 - k2] == pytest.approx(0.6)  # sub-harmonic ghost is weaker


@pytest.fixture(scope="module")
def scene15():
    return make_scene(900, AIRCRAFT + [CAR], seed=11)


def test_scene_three_aircraft_car_hum_clicks(scene15, bank):
    res = find_events(compute_spectrogram(scene15.audio), bank=bank)
    found = _check_scene(res.events, scene15.truth)
    sel = [e for e in res.events if e.selected]
    assert len(sel) == 2 and {id(e) for e in sel} <= {id(e) for e in found.values()}
    assert [e.id for e in res.events] == [f"E{i:02d}" for i in range(1, len(res.events) + 1)]
    assert all(res.events[i].t_h <= res.events[i + 1].t_h for i in range(len(res.events) - 1))
    for e in found.values():
        assert 1 in e.harmonics_seen


def test_scene_mp3_32k(bank):
    sc = make_scene(900, AIRCRAFT + [CAR], seed=12, mp3_kbps=32)
    res = find_events(compute_spectrogram(sc.audio), bank=bank)
    _check_scene(res.events, sc.truth, tol=1.5)


def test_overlapping_events(bank):
    passes = [dict(label="x", t_h=150.0, v=120.0, d=3000.0, f0=150.0, snr_db=0.0, heading=10.0),
              dict(label="y", t_h=165.0, v=80.0, d=1500.0, f0=135.0, snr_db=0.0, heading=250.0)]
    sc = make_scene(300, passes, seed=5, hum_hz=(60.0, 125.0))
    res = find_events(compute_spectrogram(sc.audio), bank=bank)
    _check_scene(res.events, sc.truth, tol=1.5)


def test_noise_only_no_events(bank):
    sc = make_scene(600, [], seed=21)
    res = find_events(compute_spectrogram(sc.audio), bank=bank)
    assert not [e for e in res.events if _usable(e)], res.events


# --- slow: full-size acceptance runs ----------------------------------------------------------------


@pytest.mark.slow
def test_one_hour_scene_runtime(bank):
    passes = [dict(p, t_h=p["t_h"] * 4.0) for p in AIRCRAFT] + [dict(CAR, t_h=1900.0)]
    sc = make_scene(3600, passes, seed=31)
    spec = compute_spectrogram(sc.audio)
    t0 = time.perf_counter()
    res = find_events(spec)  # includes building the bank
    dt = time.perf_counter() - t0
    _check_scene(res.events, sc.truth)
    assert dt <= 180.0, res.timings


def _trials(n, kbps, tol):
    bank = build_bank()
    inside, bad = 0, []
    for i in range(n):
        rng = np.random.default_rng(100 + i)
        p = random_pass(rng, 150 + rng.uniform(-20, 20))
        sc = make_scene(300, [p], seed=i, mp3_kbps=kbps)
        res = find_events(compute_spectrogram(sc.audio), bank=bank)
        tr = sc.truth[0]
        e = match(res.events, tr)
        if e is None:
            bad.append((i, "missed"))
            continue
        er = errors(e, tr)
        inside += er["inside"]
        ok = er["t"] <= 2 * tol and er["v"] <= 0.07 * tol and er["d"] <= 0.25 * tol and er["f0"] <= 0.01 * tol
        if not ok:
            bad.append((i, er))
    return inside / n, bad


@pytest.mark.slow
@pytest.mark.parametrize("kbps,tol", [(None, 1.0), (32, 1.5)])
def test_randomised_trials(kbps, tol):
    frac_inside, bad = _trials(50, kbps, tol)
    assert frac_inside >= 0.95, (frac_inside, bad)
    assert len(bad) <= 2, bad  # accuracy tolerances hold in >= 96 % of trials


# --- slow smoke test on the real recording --------------------------------------------------------


@pytest.mark.slow
def test_audio_mp3_smoke():
    """Decodes PCM samples only (never metadata). Reports what is found near the known events."""
    from triangulate.acoustic.detect import debug_plot
    from triangulate.acoustic.spectrogram import decode_and_whiten

    path = ROOT / "audio.mp3"
    if not path.exists():
        pytest.skip("audio.mp3 not present")
    spec = decode_and_whiten(path)
    res = find_events(spec)
    for lo, hi, flo, fhi in [(2000, 2300, 40, 700), (2950, 3200, 40, 700), (1180, 1340, 100, 900)]:
        debug_plot(spec, res.events, DEBUG / f"wp08_audio_{lo}-{hi}.png", lo, hi, flo, fhi,
                   title=f"audio.mp3 {lo}-{hi} s, WP08 events")
    for e in res.events:
        print(e.id, f"t_h {e.t_h:.1f} f0 {e.f0:.1f} v {e.v:.0f} [{e.v_range.lo:.0f},{e.v_range.hi:.0f}] "
              f"d {e.d:.0f} [{e.d_range.lo:.0f},{e.d_range.hi:.0f}] snr {e.snr:.1f} sel {e.selected} "
              f"{e.quality_flags}")
    print("timings", res.timings)

    def near(t):
        c = [e for e in res.events if abs(e.t_h - t) <= 10 and _usable(e)]
        return max(c, key=lambda e: e.snr) if c else None

    e1, e2 = near(2152.0), near(3073.0)
    assert e1 is not None and e2 is not None
    # plan WP08: ranges overlap the video's measured ranges [N section 2 10:33]
    from triangulate.core.types import Range

    assert e1.v_range.overlaps(Range(lo=130, hi=154)) and e1.d_range.overlaps(Range(lo=4300, hi=7800))
    assert e2.v_range.overlaps(Range(lo=88, hi=110)) and e2.d_range.overlaps(Range(lo=1390, hi=4690))
    assert e1.selected and e2.selected
