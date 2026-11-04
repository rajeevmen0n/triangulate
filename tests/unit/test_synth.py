"""WP06 synthesis: the generated audio carries exactly the modelled Doppler curve."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
from scipy.signal import ShortTimeFFT, correlate, hilbert
from scipy.signal.windows import hann

from triangulate.acoustic import physics as ph
from triangulate.acoustic import synth as sy

FS = 4000
C = 343.0
OBS = np.zeros(3)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _stft_peaks(x, f_lo_fn, f_hi_fn, fs=FS):
    """Frame-centre times and parabolic-interpolated peak frequency inside a per-frame search band.

    Hann 2048 (0.512 s), hop 500, zero-padded to 4096: the pipeline's STFT settings (plan 3.2).
    """
    sft = ShortTimeFFT(hann(2048, sym=False), hop=500, fs=fs, mfft=4096)
    S = np.abs(sft.stft(np.asarray(x, dtype=float))) ** 2  # (F, T)
    t = sft.t(len(x))
    f = sft.f
    lo, hi = sft.lower_border_end[1], sft.upper_border_begin(len(x))[1]
    out_t, out_f = [], []
    for j in range(lo, hi):
        band = np.nonzero((f >= f_lo_fn(t[j])) & (f <= f_hi_fn(t[j])))[0]
        k = band[np.argmax(S[band, j])]
        a, b, c = np.log(S[k - 1:k + 2, j] + 1e-30)
        delta = 0.5 * (a - c) / (a - 2 * b + c)
        out_t.append(t[j])
        out_f.append(f[k] + delta * (f[1] - f[0]))
    return np.array(out_t), np.array(out_f)


def _pass(t_h=60.0, v=120.0, d=2000.0, f0=130.0, harmonics=(1.0,), **kw):
    tr = sy.straight_pass_track(OBS, t_h, v, d, **kw)
    return sy.SynthSource(tr, f0, harmonics=harmonics, phases=tuple(0.3 * i for i in range(len(harmonics))))


def test_stft_peak_matches_model():
    """Acceptance: instantaneous frequency (STFT peak) within 0.5 Hz of the model."""
    v, d, f0, t_h = 120.0, 2000.0, 130.0, 60.0
    res = sy.synthesize(120.0, [_pass(t_h, v, d, f0)], OBS, noise_rms=0.01, seed=4)

    def model(t):
        return ph.straight_line_freq(t, f0, v, d, t_h)

    t, fpk = _stft_peaks(res.audio, lambda x: model(x) - 15, lambda x: model(x) + 15)
    m = (t > 10) & (t < 110)
    err = np.abs(fpk[m] - model(t[m]))
    assert m.sum() > 700
    assert err.max() < 0.5
    assert np.median(err) < 0.1


def test_harmonics_follow_model():
    v, d, f0, t_h = 140.0, 4300.0, 120.0, 60.0
    src = _pass(t_h, v, d, f0, harmonics=(1.0, 0.6, 0.4))
    res = sy.synthesize(120.0, [src], OBS, noise_rms=0.005, seed=5)
    for h in (2, 3):
        def model(t, h=h):
            return h * ph.straight_line_freq(t, f0, v, d, t_h)

        t, fpk = _stft_peaks(res.audio, lambda x, mm=model: mm(x) - 20, lambda x, mm=model: mm(x) + 20)
        m = (t > 10) & (t < 110)
        assert np.abs(fpk[m] - model(t[m])).max() < 0.5 * h


def test_hilbert_instantaneous_frequency_is_exact():
    """Phase-derivative check: the synthesis is exact retarded Doppler, not a frequency integral."""
    v, d, f0, t_h = 120.0, 2000.0, 130.0, 60.0
    res = sy.synthesize(120.0, [_pass(t_h, v, d, f0)], OBS)
    a = hilbert(res.audio.astype(float))
    fi = np.diff(np.unwrap(np.angle(a))) * FS / (2 * np.pi)
    t = (np.arange(fi.size) + 0.5) / FS
    m = (t > 10) & (t < 110)
    assert np.abs(fi[m] - ph.straight_line_freq(t[m], f0, v, d, t_h)).max() < 0.02


def test_curved_track_matches_solver():
    """Arbitrary (turning, climbing) track: STFT peak follows the track-based solver."""
    tr = sy.circle_track(np.array([1500.0, 800.0, 700.0]), 1100.0, 60.0, -10.0, 220.0, climb_rate=3.0)
    src = sy.SynthSource(tr, 110.0, harmonics=(1.0,))
    res = sy.synthesize(200.0, [src], OBS, noise_rms=0.005, seed=6)

    def model(t):
        return ph.solve_emission(np.atleast_1d(t), tr, OBS).freq(110.0)

    t, fpk = _stft_peaks(res.audio, lambda x: model(x)[0] - 12, lambda x: model(x)[0] + 12)
    m = (t > 15) & (t < 190)
    assert np.abs(fpk[m] - model(t[m])).max() < 0.5


def test_amplitude_spherical_spreading():
    v, d, t_h = 100.0, 1500.0, 100.0
    res = sy.synthesize(200.0, [_pass(t_h, v, d, 130.0)], OBS)
    env = np.abs(hilbert(res.audio.astype(float)))
    t = np.arange(env.size) / FS
    em = ph.solve_emission(t, sy.straight_pass_track(OBS, t_h, v, d), OBS)
    sel = (t > 20) & (t < 180)
    np.testing.assert_allclose(env[sel], 1000.0 / em.r[sel], rtol=0.01)
    k = np.argmin(np.abs(t - t_h))
    assert env[k] == pytest.approx(1000.0 / d, rel=0.01)


def test_truth_and_time_base():
    """Truth tau_h equals the requested heard CPA; t0 shifts the whole scene (epoch base)."""
    t0 = 1_788_907_000.0
    t_h_abs = t0 + 75.0
    v, d = 110.0, 3200.0
    tr = sy.straight_pass_track(OBS, t_h_abs, v, d, heading_deg=30.0, side=-1, height=900.0)
    src = sy.SynthSource(tr, 125.0, harmonics=(1.0,), phases=(0.0,), label="ac1")
    res = sy.synthesize(150.0, [src], OBS, t0=t0)
    (truth,) = res.truth
    assert truth.label == "ac1" and len(truth.passes) == 1
    p = truth.passes[0]
    assert truth.tau_h[0] == pytest.approx(75.0, abs=1e-3)
    assert p.d == pytest.approx(d, abs=1e-3) and p.v == pytest.approx(v)
    assert p.t_h == pytest.approx(p.t_cpa + d / C, abs=1e-6)
    # same audio as the relative-time scene
    tr_rel = sy.straight_pass_track(OBS, 75.0, v, d, heading_deg=30.0, side=-1, height=900.0)
    rel = sy.synthesize(150.0, [sy.SynthSource(tr_rel, 125.0, harmonics=(1.0,), phases=(0.0,))], OBS)
    assert np.max(np.abs(res.audio - rel.audio)) < 1e-4
    assert res.audio.dtype == np.float32 and res.audio.size == 150 * FS
    assert res.duration == pytest.approx(150.0)


def test_silent_outside_track_span():
    tr = sy.straight_pass_track(OBS, 50.0, 100.0, 1000.0, half_span_s=20.0)
    res = sy.synthesize(120.0, [sy.SynthSource(tr, 100.0, taper_s=2.0)], OBS)
    lo, hi = tr.t_start + np.linalg.norm(tr.pos[0]) / C, tr.t_end + np.linalg.norm(tr.pos[-1]) / C
    t = res.times()
    assert np.all(res.audio[(t < lo) | (t > hi)] == 0.0)
    assert np.abs(res.audio[(t > lo + 5) & (t < hi - 5)]).max() > 0.1


def test_reproducible_and_seeded():
    kw = dict(noise_rms=0.05, hum=[(60.0, 0.02)], click_rate_per_min=10.0, click_amp=0.5)
    src = [sy.SynthSource(sy.straight_pass_track(OBS, 30.0, 90.0, 1500.0), 120.0)]
    a = sy.synthesize(60.0, src, OBS, seed=11, **kw).audio
    b = sy.synthesize(60.0, src, OBS, seed=11, **kw).audio
    c = sy.synthesize(60.0, src, OBS, seed=12, **kw).audio
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_pink_noise_spectrum():
    rng = np.random.default_rng(0)
    x = sy.pink_noise(FS * 120, FS, 0.1, rng)
    assert np.sqrt(np.mean(x**2)) == pytest.approx(0.1, rel=1e-6)
    p1 = sy.band_power(x, FS, 20, 40)
    p2 = sy.band_power(x, FS, 200, 400)
    # 1/f: equal power per octave/decade-ratio band
    assert 10 * np.log10(p1 / p2) == pytest.approx(0.0, abs=0.5)
    p3 = sy.band_power(x, FS, 100, 110)
    p4 = sy.band_power(x, FS, 1000, 1010)
    assert 10 * np.log10(p3 / p4) == pytest.approx(10.0, abs=1.0)  # -10 dB/decade at equal width


def test_background_components():
    res = sy.synthesize(60.0, [], OBS, hum=[(f, 0.05) for f in sy.DEMO_HUM_HZ], click_rate_per_min=30,
                        click_amp=1.0, seed=3, keep_components=True)
    spec = np.abs(np.fft.rfft(res.background.astype(float))) / res.audio.size
    f = np.fft.rfftfreq(res.audio.size, 1 / FS)
    for fh in sy.DEMO_HUM_HZ:
        k = np.argmin(np.abs(f - fh))
        assert spec[k] > 0.02  # ~0.025 for a 0.05-amplitude line
    assert np.all(res.signal == 0)
    # clicks: sparse, large excursions far above the hum floor
    assert np.sum(np.abs(res.background) > 0.5) > 5


def test_band_power_parseval():
    t = np.arange(FS * 10) / FS
    x = 0.3 * np.sin(2 * np.pi * 150 * t)
    assert sy.band_power(x, FS, 140, 160) == pytest.approx(0.045, rel=1e-6)
    assert sy.band_power(x, FS, 200, 300) < 1e-12
    assert sy.band_power(np.ones(400), 400, 0, 0) == pytest.approx(1.0)  # DC has one-sided weight 1


@needs_ffmpeg
@pytest.mark.parametrize("kbps", [64, 32])
def test_mp3_roundtrip_preserves_curve_and_alignment(kbps):
    v, d, f0, t_h = 120.0, 2000.0, 130.0, 40.0
    clean = sy.synthesize(80.0, [_pass(t_h, v, d, f0, harmonics=(1.0, 0.5))], OBS, noise_rms=0.01, seed=8)
    res = sy.synthesize(80.0, [_pass(t_h, v, d, f0, harmonics=(1.0, 0.5))], OBS, noise_rms=0.01, seed=8,
                        mp3_kbps=kbps)
    assert res.audio.size == clean.audio.size and res.audio.dtype == np.float32
    seg = slice(40_000, 200_000)
    cc = correlate(res.audio[seg].astype(float), clean.audio[seg].astype(float), mode="full", method="fft")
    assert np.argmax(cc) - (seg.stop - seg.start - 1) == 0  # no codec delay left
    assert np.corrcoef(res.audio[seg], clean.audio[seg])[0, 1] > 0.99

    def model(t):
        return ph.straight_line_freq(t, f0, v, d, t_h)

    t, fpk = _stft_peaks(res.audio, lambda x: model(x) - 15, lambda x: model(x) + 15)
    m = (t > 8) & (t < 72)
    assert np.abs(fpk[m] - model(t[m])).max() < 0.5


def test_straight_pass_track_geometry():
    for heading, side in [(0.0, 1), (90.0, -1), (237.0, 1)]:
        tr = sy.straight_pass_track(OBS + 5.0, 100.0, 80.0, 2500.0, heading_deg=heading, side=side,
                                    height=1000.0)
        (p,) = ph.track_passes(tr, OBS + 5.0)
        assert p.d == pytest.approx(2500.0, abs=1e-6)
        assert p.t_h == pytest.approx(100.0, abs=1e-6)
        assert np.allclose(tr.speed, 80.0)
        h = math.radians(heading)
        assert np.allclose(tr.vel[0, :2] / 80.0, [math.sin(h), math.cos(h)])
    with pytest.raises(ValueError):
        sy.straight_pass_track(OBS, 0.0, 80.0, 500.0, height=600.0)


def test_write_wav_roundtrip(tmp_path):
    import soundfile as sf

    x = (0.1 * np.sin(np.arange(4000) * 0.1)).astype(np.float32)
    p = sy.write_wav(tmp_path / "x.wav", x)
    y, fs = sf.read(p, dtype="float32")
    assert fs == FS and np.array_equal(x, y)
