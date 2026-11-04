"""WP20 evidence renderers: spectrogram PNG + JSON overlay, posterior heatmap, band maps.

All inputs are small and synthetic. Images are read back with Pillow, and the pixel positions are
checked against the expected frequencies and coordinates.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from triangulate.acoustic.physics import straight_line_freq
from triangulate.acoustic.spectrogram import Spectrogram, compute_spectrogram
from triangulate.core.geo import haversine_m
from triangulate.core.types import Event, GridSpec, Range
from triangulate.fusion.combine import fuse, kernel_layer
from triangulate.render.common import mercator_latlon_grid, safe_name, spec_bounds_latlon
from triangulate.render.heatmap_png import (
    density_rgba,
    render_band_map,
    render_band_maps,
    render_posterior_heatmap,
    write_heatmap,
)
from triangulate.render.spectrogram_png import (
    CURVE_COLOURS,
    OVERLAY_FORMAT,
    WindowMap,
    model_curves_for_event,
    render_spectrogram_window,
    spectrogram_overlay,
)

F0, V, D, TH = 130.0, 60.0, 800.0, 40.0  # straight pass: heard CPA at 40 s, f0 = 130 Hz
BBOX = (40.75, -73.70, 41.05, -73.30)  # ~33 x 34 km


def make_event() -> Event:
    return Event(
        id="E1", t_h=TH, t_h_range=Range(lo=37.0, hi=43.0), f0=F0, f0_range=Range(lo=127.0, hi=133.0),
        v=V, v_range=Range(lo=50.0, hi=70.0), d=D, d_range=Range(lo=600.0, hi=1000.0), snr=9.0,
        ridge_points=[(10.0, 140.0), (25.0, 125.0), (35.0, 118.0)],  # first one is outside the window
        traced_line=[(30.0, 120.0), (34.0, 117.0)],
    )


def ridge_hz(t):
    return straight_line_freq(t, F0, V, D, TH)


def synth_spectrogram(duration_s: float = 80.0, ridge=None) -> Spectrogram:
    """A Spectrogram on the real STFT grid (hop 500 at 4 kHz, bins 20-1000 Hz) with a Gaussian ridge."""
    dt = 500 / 4000
    t = (np.arange(int(duration_s / dt)) * 500 + 1024) / 4000
    freqs = np.fft.rfftfreq(4096, 1 / 4000)
    f = freqs[(freqs >= 20) & (freqs <= 1000)]
    Z = np.zeros((t.size, f.size), dtype=np.float32)
    if ridge is not None:
        fr = ridge(t)
        Z += 12.0 * np.exp(-0.5 * ((f[None, :] - fr[:, None]) / 2.0) ** 2)
    Z16 = Z.astype(np.float16)
    return Spectrogram(t=t, f=f, Z=Z16, db=np.zeros(Z16.shape, dtype=np.float32))


def read_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).astype(float)


def test_window_map_round_trip_and_corners():
    for scale in ("linear", "log"):
        wm = WindowMap(t0=10.0, t1=30.0, fmin=50.0, fmax=400.0, width_px=200, height_px=100, scale=scale)
        f = np.array([60.0, 123.0, 380.0])
        t = np.array([11.0, 20.0, 29.0])
        assert np.allclose(wm.y_to_f(wm.f_to_y(f)), f)
        assert np.allclose(wm.x_to_t(wm.t_to_x(t)), t)
        assert np.all(np.diff(wm.pixel_freqs()) < 0)  # top row is the highest frequency
    lin = WindowMap(t0=0.0, t1=10.0, fmin=100.0, fmax=200.0, width_px=50, height_px=100)
    assert lin.f_to_y(200.0) == pytest.approx(0.0)
    assert lin.f_to_y(100.0) == pytest.approx(100.0)
    assert lin.t_to_x(10.0) == pytest.approx(50.0)
    log = WindowMap(t0=0.0, t1=10.0, fmin=50.0, fmax=800.0, width_px=50, height_px=100, scale="log")
    assert log.f_to_y(np.sqrt(50.0 * 800.0)) == pytest.approx(50.0)  # geometric midpoint
    with pytest.raises(ValueError):
        WindowMap(t0=1.0, t1=1.0, fmin=10.0, fmax=20.0, width_px=10, height_px=10)


def test_png_size_and_model_curve_follows_ridge(tmp_path):
    spec = synth_spectrogram(80.0, ridge=ridge_hz)
    out = tmp_path / "spec.png"
    overlay = render_spectrogram_window(spec, out, t0=0.0, t1=80.0, fmin=90.0, fmax=180.0,
                                        width_px=400, height_px=300, cmap="gray",
                                        curves=model_curves_for_event(make_event(), (1,)))
    img = read_gray(out)
    assert img.shape == (300, 400)
    assert img.std() > 10.0  # not blank
    wm = WindowMap(t0=0.0, t1=80.0, fmin=90.0, fmax=180.0, width_px=400, height_px=300)
    tc = wm.pixel_times()
    bin_hz = float(spec.df)
    for col in range(10, 390, 30):
        row = int(np.argmax(img[:, col]))
        f_png = float(wm.y_to_f(row + 0.5))  # pixel centre
        f_expected = float(ridge_hz(tc[col]))
        assert abs(f_png - f_expected) < 1.5 + bin_hz, (col, f_png, f_expected)
    # the JSON curve sits on the same ridge (pixel space)
    px = np.asarray(overlay["curves"][0]["points_px"])
    assert np.allclose(wm.f_to_y(ridge_hz(wm.x_to_t(px[:, 0]))), px[:, 1], atol=0.5)


def test_overlay_json_maps_to_pixels(tmp_path):
    spec = synth_spectrogram(80.0, ridge=ridge_hz)
    overlay = render_spectrogram_window(spec, tmp_path / "w.png", t0=20.0, t1=60.0, fmin=90.0, fmax=180.0,
                                        width_px=320, height_px=200, events=[make_event()])
    data = json.loads((tmp_path / "w.json").read_text())
    assert data == json.loads(json.dumps(overlay))
    assert data["format"] == OVERLAY_FORMAT
    assert data["image"] == {"file": "w.png", "width_px": 320, "height_px": 200}
    wm = WindowMap(t0=20.0, t1=60.0, fmin=90.0, fmax=180.0, width_px=320, height_px=200)

    assert [c["harmonic"] for c in data["curves"]] == [1, 2, 3]
    for c in data["curves"]:
        pts = np.asarray(c["points"])
        px = np.asarray(c["points_px"])
        assert np.all(np.diff(pts[:, 0]) > 0)
        assert np.allclose(wm.t_to_x(pts[:, 0]), px[:, 0], atol=0.01)
        assert np.allclose(wm.f_to_y(pts[:, 1]), px[:, 1], atol=0.01)
        assert px[:, 0].min() >= 0 and px[:, 0].max() <= 320
    h1, h2 = np.asarray(data["curves"][0]["points"]), np.asarray(data["curves"][1]["points"])
    assert np.allclose(h2[:, 1], 2 * h1[:, 1], rtol=1e-3)
    assert np.allclose(h1[:, 1], ridge_hz(h1[:, 0]), rtol=1e-3)

    e = data["events"][0]
    assert e["t_h_px"] == pytest.approx(float(wm.t_to_x(TH)), abs=0.01)
    assert e["f0_px"] == pytest.approx(float(wm.f_to_y(F0)), abs=0.01)
    lo, hi = e["t_h_range_px"]
    assert lo < e["t_h_px"] < hi
    flo, fhi = e["f0_range_px"]
    assert flo < e["f0_px"] < fhi

    # ridge points and traced lines are clipped to the window (the t=10 s point is dropped)
    ridge = data["ridge_points"][0]
    assert len(ridge["points"]) == 2
    assert np.allclose(wm.f_to_y(np.asarray(ridge["points"])[:, 1]), np.asarray(ridge["points_px"])[:, 1],
                       atol=0.01)
    assert len(data["traced_lines"][0]["points"]) == 2


def test_burn_in_draws_curve_colour(tmp_path):
    spec = synth_spectrogram(80.0, ridge=None)  # blank background
    overlay = render_spectrogram_window(spec, tmp_path / "burn.png", t0=20.0, t1=60.0, fmin=90.0,
                                        fmax=180.0, width_px=320, height_px=200, events=[make_event()],
                                        burn_in=True, cmap="gray")
    img = np.asarray(Image.open(tmp_path / "burn.png").convert("RGB")).astype(int)
    x, y = overlay["curves"][0]["points_px"][len(overlay["curves"][0]["points_px"]) // 4]
    patch = img[max(int(y) - 2, 0): int(y) + 3, max(int(x) - 2, 0): int(x) + 3]
    target = np.asarray(CURVE_COLOURS[0])
    assert np.any(np.all(np.abs(patch - target) <= 10, axis=-1))
    plain = render_spectrogram_window(spec, tmp_path / "plain.png", t0=20.0, t1=60.0, fmin=90.0, fmax=180.0,
                                      width_px=320, height_px=200, events=[make_event()], cmap="gray",
                                      json_sidecar=False)
    assert plain["burned_in"] is False
    assert not (tmp_path / "plain.json").exists()


def test_overlay_without_files_and_default_window():
    spec = synth_spectrogram(20.0, ridge=None)
    curves = model_curves_for_event(make_event())
    wm, overlay = spectrogram_overlay(spec, width_px=100, height_px=50, curves=curves)
    assert wm.t0 == pytest.approx(spec.t[0] - spec.dt / 2)
    assert wm.fmin == pytest.approx(spec.f[0]) and wm.fmax == pytest.approx(spec.f[-1])
    assert len(overlay["curves"]) == 3


def test_chirp_from_compute_spectrogram_is_visible(tmp_path):
    fs = 4000
    t = np.arange(fs * 20) / fs
    x = (0.1 * np.sin(2 * np.pi * (120.0 * t + 0.5 * t * t))).astype(np.float32)  # 120 -> 140 Hz
    spec = compute_spectrogram(x, fs)
    render_spectrogram_window(spec, tmp_path / "chirp.png", fmin=100.0, fmax=160.0, width_px=300,
                              height_px=200, cmap="gray", json_sidecar=False)
    img = read_gray(tmp_path / "chirp.png")
    assert img.shape == (200, 300)
    assert img.std() > 5.0
    wm = WindowMap(t0=float(spec.t[0] - spec.dt / 2), t1=float(spec.t[-1] + spec.dt / 2), fmin=100.0,
                   fmax=160.0, width_px=300, height_px=200)
    for col in (150, 250):
        f_peak = float(wm.y_to_f(np.argmax(img[:, col]) + 0.5))
        assert 115.0 < f_peak < 145.0


def test_density_rgba_alpha_rules():
    rgba = density_rgba(np.array([[0.0, -np.inf], [np.nan, -20.0]]), dynamic_range_nats=8.0)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] > 200  # the peak is opaque
    assert rgba[0, 1, 3] == 0 and rgba[1, 0, 3] == 0 and rgba[1, 1, 3] == 0  # non-finite / far below
    empty = density_rgba(np.full((3, 3), -np.inf))
    assert np.all(empty[..., 3] == 0)


def _kernel_posterior(lat0: float, lon0: float):
    spec = GridSpec.for_bbox(BBOX, max_dense_cells=40_000)
    lay = kernel_layer(spec, lat0, lon0, sigma_m=2000.0, confidence=0.9, name="clue_test", module="visual")
    return spec, lay, fuse(spec, [lay], domain=BBOX)


def test_posterior_heatmap_peak_is_at_source(tmp_path):
    lat0, lon0 = 40.90, -73.50
    _, _, post = _kernel_posterior(lat0, lon0)
    info = render_posterior_heatmap(post, tmp_path / "heatmap.png", width_px=400, bounds=BBOX)
    img = np.asarray(Image.open(tmp_path / "heatmap.png"))
    assert img.shape == (info["image"]["height_px"], 400, 4)
    alpha = img[..., 3].astype(float)
    assert alpha.max() > 200
    LAT, LON = mercator_latlon_grid(BBOX, 400)
    w = alpha / alpha.sum()
    lat_c, lon_c = float((w * LAT).sum()), float((w * LON).sum())
    assert haversine_m(lat_c, lon_c, lat0, lon0) < 1000.0
    assert info["bounds"] == {"south": BBOX[0], "west": BBOX[1], "north": BBOX[2], "east": BBOX[3]}
    assert info["leaflet_bounds"] == [[BBOX[0], BBOX[1]], [BBOX[2], BBOX[3]]]
    assert json.loads((tmp_path / "heatmap.json").read_text())["crs"] == "EPSG:3857"


def test_write_heatmap_into_workdir(tmp_path):
    _, _, post = _kernel_posterior(40.90, -73.50)
    names = write_heatmap(post, tmp_path / "job", width_px=128)
    assert names == {"heatmap_png": "heatmap.png", "heatmap_json": "heatmap.json"}
    assert (tmp_path / "job" / "heatmap.png").stat().st_size > 0
    assert json.loads((tmp_path / "job" / "heatmap.json").read_text())["image"]["width_px"] == 128


def test_band_map_is_red_at_support_and_transparent_at_baseline(tmp_path):
    lat0, lon0 = 40.90, -73.50
    spec, lay, _ = _kernel_posterior(lat0, lon0)
    info = render_band_map(lay, tmp_path / "band.png", bounds=BBOX, width_px=300)
    img = np.asarray(Image.open(tmp_path / "band.png")).astype(float)
    assert img.shape[1] == 300
    alpha = img[..., 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0  # corners are beyond 5 sigma: baseline, so transparent
    LAT, LON = mercator_latlon_grid(BBOX, 300)
    w = alpha / alpha.sum()
    lat_c, lon_c = float((w * LAT).sum()), float((w * LON).sum())
    assert haversine_m(lat_c, lon_c, lat0, lon0) < 1000.0
    r, g, b = img[np.unravel_index(np.argmax(alpha), alpha.shape)[:2]][:3]
    assert r > b  # supports the location
    assert info["baseline_subtracted"] == pytest.approx(lay.fill)
    assert info["layer"]["name"] == "clue_test"


def test_band_maps_one_file_per_layer(tmp_path):
    spec, lay, _ = _kernel_posterior(40.90, -73.50)
    lay2 = kernel_layer(spec, 40.80, -73.60, sigma_m=1500.0, confidence=0.5, name="clue/two", module="visual")
    infos = render_band_maps([lay, lay2], tmp_path, bounds=BBOX, width_px=64)
    assert [i["file_stem"] for i in infos] == ["band_clue_test", "band_clue_two"]
    assert (tmp_path / "band_clue_two.png").exists() and (tmp_path / "band_clue_two.json").exists()
    assert safe_name("a b/c") == "a_b_c"


def test_mercator_grid_and_spec_bounds():
    LAT, LON = mercator_latlon_grid(BBOX, 200)
    assert LAT.shape[1] == 200 and LAT.shape[0] > 1
    assert np.all((LAT > BBOX[0]) & (LAT < BBOX[2]) & (LON > BBOX[1]) & (LON < BBOX[3]))
    assert LAT[0, 0] > LAT[-1, 0]  # row 0 is north
    spec = GridSpec.for_bbox(BBOX, max_dense_cells=40_000)
    s, w, n, e = spec_bounds_latlon(spec)
    assert s < BBOX[0] + 0.05 and n > BBOX[2] - 0.05 and w < BBOX[1] + 0.05 and e > BBOX[3] - 0.05
