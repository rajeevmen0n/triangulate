"""Spectrogram window PNGs with a matching JSON overlay (plan 2.2, WP20).

The PNG shows the whitened SNR map Z(t, f) for a time/frequency window. Row 0 is the top (f max),
column 0 is the window start. The JSON overlay lists the model curves, ridge points, traced lines
and event markers in both data units (t in audio seconds, f in Hz) and pixel space, so the web UI
can draw them as SVG on top of the PNG.

Pixel convention (continuous): pixel column i spans x in [i, i+1) and row j spans y in [j, j+1);
a data point maps to x = (t - t0) / (t1 - t0) * width and y = (g(fmax) - g(f)) / (g(fmax) - g(fmin))
* height, where g is the identity (linear axis) or ln (log axis). Pixel centres are at i + 0.5.

Public API
    render_spectrogram_window(spec, out_png, ...) -> overlay dict   (writes PNG and out_png.json)
    spectrogram_overlay(spec, ...) -> (WindowMap, overlay dict)     (pure, no files)
    model_curves_for_event(event, harmonics) -> list[ModelCurve]
    WindowMap                                                       (data <-> pixel mapping)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..acoustic.physics import C_DEFAULT, straight_line_freq
from ..acoustic.spectrogram import Z_CLIP, Spectrogram
from ..core.types import Event
from .common import colormap_rgba, to_uint8, write_json, write_png

__all__ = ["OVERLAY_FORMAT", "DEFAULT_HARMONICS", "ModelCurve", "WindowMap", "model_curves_for_event",
           "sample_model_curve", "spectrogram_overlay", "render_spectrogram_window"]

OVERLAY_FORMAT = "triangulate.spectrogram_overlay/1"
DEFAULT_HARMONICS: tuple[int, ...] = (1, 2, 3)
CURVE_COLOURS: tuple[tuple[int, int, int], ...] = ((0, 255, 255), (255, 160, 0), (140, 255, 0),
                                                   (255, 0, 255), (80, 140, 255))
RIDGE_COLOUR = (255, 255, 255)
TRACE_COLOUR = (255, 255, 0)
EVENT_COLOUR = (255, 60, 60)


@dataclass(frozen=True)
class WindowMap:
    """Maps audio time (s) and frequency (Hz) of a window to continuous pixel coordinates."""

    t0: float
    t1: float
    fmin: float
    fmax: float
    width_px: int
    height_px: int
    scale: str = "linear"  # "linear" or "log" (frequency axis)

    def __post_init__(self) -> None:
        if not self.t1 > self.t0:
            raise ValueError("t1 must be greater than t0")
        if not 0.0 < self.fmin < self.fmax:
            raise ValueError("need 0 < fmin < fmax")
        if self.scale not in ("linear", "log"):
            raise ValueError("scale must be 'linear' or 'log'")
        if self.width_px < 1 or self.height_px < 1:
            raise ValueError("image size must be positive")

    def _g(self, f):
        return np.log(f) if self.scale == "log" else f

    def _ginv(self, g):
        return np.exp(g) if self.scale == "log" else g

    def t_to_x(self, t):
        return (np.asarray(t, dtype=float) - self.t0) / (self.t1 - self.t0) * self.width_px

    def x_to_t(self, x):
        return self.t0 + np.asarray(x, dtype=float) / self.width_px * (self.t1 - self.t0)

    def f_to_y(self, f):
        g0, g1 = self._g(self.fmin), self._g(self.fmax)
        return (g1 - self._g(np.asarray(f, dtype=float))) / (g1 - g0) * self.height_px

    def y_to_f(self, y):
        g0, g1 = self._g(self.fmin), self._g(self.fmax)
        return self._ginv(g1 - np.asarray(y, dtype=float) / self.height_px * (g1 - g0))

    def pixel_times(self) -> np.ndarray:
        """Centre time of each pixel column (W,)."""
        return self.x_to_t(np.arange(self.width_px) + 0.5)

    def pixel_freqs(self) -> np.ndarray:
        """Centre frequency of each pixel row (H,), top row first (descending)."""
        return self.y_to_f(np.arange(self.height_px) + 0.5)


@dataclass(frozen=True)
class ModelCurve:
    """One modelled Doppler curve: harmonic h of a straight pass (t_h, f0, v, d), in Hz."""

    t_h: float
    f0: float
    v: float
    d: float
    harmonic: int = 1
    c: float = C_DEFAULT
    event_id: str = ""
    hypothesis: str = ""
    label: str = ""


def model_curves_for_event(event: Event, harmonics: Sequence[int] = DEFAULT_HARMONICS, *,
                           c: float = C_DEFAULT, hypothesis: str = "") -> list[ModelCurve]:
    """The fitted event's model curves, one per harmonic (f_h = h * f0 * g(t))."""
    return [ModelCurve(t_h=event.t_h, f0=event.f0, v=event.v, d=event.d, harmonic=int(h), c=c,
                       event_id=event.id, hypothesis=hypothesis, label=f"{event.id} H{h}")
            for h in harmonics]


def sample_model_curve(mc: ModelCurve, t: np.ndarray) -> np.ndarray:
    """Frequency (Hz) of the curve at reception times `t` (audio seconds)."""
    return mc.harmonic * np.asarray(straight_line_freq(t, mc.f0, mc.v, mc.d, mc.t_h, mc.c), dtype=float)


def _nearest_index(grid: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Index of the nearest entry of ascending `grid` for each query value."""
    if grid.size == 1:
        return np.zeros(q.shape, dtype=int)
    i = np.clip(np.searchsorted(grid, q), 1, grid.size - 1)
    left = np.abs(q - grid[i - 1])
    right = np.abs(grid[i] - q)
    return np.where(left <= right, i - 1, i)


def window_rgb(spec: Spectrogram, wm: WindowMap, cmap: str = "magma") -> np.ndarray:
    """(H, W, 3) uint8 image of Z over the window (nearest-neighbour sampling, Z/12 in [0, 1])."""
    it = _nearest_index(np.asarray(spec.t, dtype=float), wm.pixel_times())
    jf = _nearest_index(np.asarray(spec.f, dtype=float), wm.pixel_freqs())
    z = np.asarray(spec.Z, dtype=np.float32)[it[None, :], jf[:, None]] / Z_CLIP
    return to_uint8(colormap_rgba(cmap, z))[..., :3]


def _pts(wm: WindowMap, t: np.ndarray, f: np.ndarray) -> tuple[list, list]:
    """Data points and their pixel positions, keeping only finite samples."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f = t[ok], f[ok]
    x, y = wm.t_to_x(t), wm.f_to_y(f)
    return ([[round(float(a), 6), round(float(b), 4)] for a, b in zip(t, f, strict=True)],
            [[round(float(a), 2), round(float(b), 2)] for a, b in zip(x, y, strict=True)])


def _in_window(wm: WindowMap, pairs: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
    keep = (arr[:, 0] >= wm.t0) & (arr[:, 0] <= wm.t1)
    return arr[keep, 0], arr[keep, 1]


def _range_px(wm: WindowMap, lo: float, hi: float) -> list[float]:
    return [round(float(v), 2) for v in sorted((wm.t_to_x(lo), wm.t_to_x(hi)))]


def build_overlay(wm: WindowMap, *, image_file: str, events: Sequence[Event] = (),
                  curves: Sequence[ModelCurve] = (), n_samples: int | None = None,
                  burned_in: bool = False) -> dict[str, Any]:
    """The JSON overlay for a window (pure function; see the module docstring for conventions)."""
    n = int(n_samples or min(wm.width_px, 600))
    t_grid = np.linspace(wm.t0, wm.t1, n)
    ev_out, ridge_out, trace_out = [], [], []
    for ev in events:
        ev_out.append({
            "id": ev.id, "selected": bool(ev.selected), "snr": float(ev.snr),
            "quality_flags": list(ev.quality_flags),
            "t_h": float(ev.t_h), "t_h_range": [float(ev.t_h_range.lo), float(ev.t_h_range.hi)],
            "t_h_px": round(float(wm.t_to_x(ev.t_h)), 2),
            "t_h_range_px": _range_px(wm, ev.t_h_range.lo, ev.t_h_range.hi),
            "f0": float(ev.f0), "f0_range": [float(ev.f0_range.lo), float(ev.f0_range.hi)],
            "f0_px": round(float(wm.f_to_y(ev.f0)), 2),
            "f0_range_px": [round(float(v), 2) for v in sorted((wm.f_to_y(ev.f0_range.lo),
                                                                 wm.f_to_y(ev.f0_range.hi)))],
            "v": float(ev.v), "d": float(ev.d),
        })
        if ev.ridge_points:
            t, f = _in_window(wm, ev.ridge_points)
            pts, px = _pts(wm, t, f)
            ridge_out.append({"event_id": ev.id, "points": pts, "points_px": px})
        if ev.traced_line:
            t, f = _in_window(wm, ev.traced_line)
            pts, px = _pts(wm, t, f)
            trace_out.append({"event_id": ev.id, "points": pts, "points_px": px})
    curve_out = []
    for mc in curves:
        pts, px = _pts(wm, t_grid, sample_model_curve(mc, t_grid))
        curve_out.append({"id": mc.label or f"{mc.event_id} H{mc.harmonic}", "event_id": mc.event_id,
                          "hypothesis": mc.hypothesis, "harmonic": mc.harmonic,
                          "f0": mc.f0, "v": mc.v, "d": mc.d, "t_h": mc.t_h,
                          "points": pts, "points_px": px})
    return {
        "format": OVERLAY_FORMAT,
        "image": {"file": image_file, "width_px": wm.width_px, "height_px": wm.height_px},
        "window": {"t0": wm.t0, "t1": wm.t1, "fmin": wm.fmin, "fmax": wm.fmax, "scale": wm.scale},
        "coordinates": {"x": "t (audio s) -> px, pixel i spans [i, i+1)",
                        "y": "f (Hz) -> px, row 0 = fmax, pixel j spans [j, j+1)",
                        "points": "[t_s, f_hz]", "points_px": "[x_px, y_px]"},
        "burned_in": bool(burned_in),
        "events": ev_out, "curves": curve_out, "ridge_points": ridge_out, "traced_lines": trace_out,
    }


def _window_map(spec: Spectrogram, t0, t1, fmin, fmax, width_px, height_px, scale) -> WindowMap:
    dt = float(spec.dt)
    t0 = float(spec.t[0] - dt / 2) if t0 is None else float(t0)
    t1 = float(spec.t[-1] + dt / 2) if t1 is None else float(t1)
    fmin = float(spec.f[0]) if fmin is None else float(fmin)
    fmax = float(spec.f[-1]) if fmax is None else float(fmax)
    return WindowMap(t0=t0, t1=t1, fmin=fmin, fmax=fmax, width_px=int(width_px),
                     height_px=int(height_px), scale=scale)


def _burn_in(img: Image.Image, overlay: dict[str, Any]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i, c in enumerate(overlay["curves"]):
        px = [tuple(p) for p in c["points_px"]]
        if len(px) >= 2:
            draw.line(px, fill=CURVE_COLOURS[i % len(CURVE_COLOURS)], width=2)
    for r in overlay["ridge_points"]:
        for x, y in r["points_px"]:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=RIDGE_COLOUR)
    for tr in overlay["traced_lines"]:
        px = [tuple(p) for p in tr["points_px"]]
        if len(px) >= 2:
            draw.line(px, fill=TRACE_COLOUR, width=2)
    H = overlay["image"]["height_px"]
    for ev in overlay["events"]:
        draw.line([(ev["t_h_px"], 0), (ev["t_h_px"], H - 1)], fill=EVENT_COLOUR, width=1)
    return img


def spectrogram_overlay(spec: Spectrogram, *, t0: float | None = None, t1: float | None = None,
                        fmin: float | None = None, fmax: float | None = None, width_px: int = 1200,
                        height_px: int = 480, scale: str = "linear", image_file: str = "spectrogram.png",
                        events: Sequence[Event] = (), curves: Sequence[ModelCurve] = (),
                        n_samples: int | None = None) -> tuple[WindowMap, dict[str, Any]]:
    """Window map and overlay dict without writing files. Defaults: the full spectrogram extent."""
    wm = _window_map(spec, t0, t1, fmin, fmax, width_px, height_px, scale)
    return wm, build_overlay(wm, image_file=image_file, events=events, curves=curves, n_samples=n_samples)


def render_spectrogram_window(spec: Spectrogram, out_png: str | Path, *, t0: float | None = None,
                              t1: float | None = None, fmin: float | None = None, fmax: float | None = None,
                              width_px: int = 1200, height_px: int = 480, scale: str = "linear",
                              cmap: str = "magma", events: Sequence[Event] = (),
                              curves: Sequence[ModelCurve] | None = None,
                              harmonics: Sequence[int] = DEFAULT_HARMONICS, burn_in: bool = False,
                              json_sidecar: bool = True) -> dict[str, Any]:
    """Render a spectrogram window to `out_png` and its overlay to `out_png` with suffix `.json`.

    `events`: fitted events. Their model curves (one per harmonic in `harmonics`) are added unless
    `curves` is given explicitly; their ridge points, traced lines and t_h/f0 markers are always
    included. `burn_in=True` also draws the overlay into the PNG (for static reports). Returns the
    overlay dict.
    """
    out_png = Path(out_png)
    wm = _window_map(spec, t0, t1, fmin, fmax, width_px, height_px, scale)
    if curves is None:
        curves = [mc for ev in events for mc in model_curves_for_event(ev, harmonics)]
    overlay = build_overlay(wm, image_file=out_png.name, events=events, curves=curves, burned_in=burn_in)
    img = Image.fromarray(window_rgb(spec, wm, cmap))
    if burn_in:
        img = _burn_in(img, overlay)
    write_png(np.asarray(img), out_png)
    if json_sidecar:
        write_json(overlay, out_png.with_suffix(".json"))
    return overlay
