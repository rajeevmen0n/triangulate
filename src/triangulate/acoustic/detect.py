"""Aircraft Doppler event detection (plan 3.2 "Detection: log-frequency template bank"), WP08.

Pipeline (all on the whitened SNR map Z from ``spectrogram.py``):

1. ``log_resample`` Z onto u = ln f (du = 0.004) at 0.5 s, then stack harmonics
   ``H(t,u) = sum_h w_h Z(t, u + ln h)`` (w = 1, 0.6, 0.4) and normalise every u row to robust
   unit scale (median 0, MAD-sigma 1), so out-of-band harmonics do not bias high frequencies.
2. Template bank: straight-line closed-form curves in (t, u) for v in 25..160 m/s (step 5) and
   d in 250..12000 m (16 log-spaced), drawn as a Gaussian ridge (sigma_u = 1.5 bins) that is
   zero-mean within +-12 bins of the ridge in every column (rejects broadband structure), over
   t_h +- clip(time to 95 % of the asymptote, 15 s, 120 s). The shape in u depends only on (v, d);
   f0 is the u position of the anchor. A few steep "ground vehicle" templates (d 15..120 m,
   v 15..35 m/s) are added so that car/train passes are found and *labelled* rather than leaking
   into aircraft templates.
3. FFT cross-correlation of H with every template (threads, scipy.fft), each response normalised
   by its own robust (median, MAD) statistics -> detection SNR in sigma units. Running max/argmax
   over templates; local maxima (+-20 s, +-0.05 in u) above ``cand_min`` become candidates.
4. Greedy explain-away (matching pursuit): candidates are re-scored on the progressively masked
   map; the best one is accepted if its SNR > ``snr_min`` (6) and it passes the **flat-line contrast
   test** (its curve, gathered on a map from which every row's 30-s running mean was removed --
   i.e. with all constant-frequency structure taken out -- still scores >= 3 sigma). It is then
   curve-fitted (``fit.py``) and its fitted curve (all harmonics) is masked out, which removes its
   partial matches, harmonic ghosts (f0/2, 2 f0) and duplicates.
5. Labelling: fitted d < 250 m, v < 25 m/s, or a sweep (95 %-asymptote span) shorter than 20 s
   -> ``quality_flags`` gets "non-aircraft" and the event is not selected. The top
   ``n_select`` (2) aircraft events by SNR are marked ``selected``.

Public API:
    find_events(spec, config=None) -> DetectionResult  (events: list[core.types.Event], ...)
    detect(spec, config=None) -> list[Detection]        (bank detection only, no fit)
    build_bank(config) -> TemplateBank
    debug_plot(spec, events, path, t_lo, t_hi)         (matplotlib PNG, for data/debug/)
"""

from __future__ import annotations

import heapq
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter, uniform_filter1d

from ..core.types import Event
from .physics import C_DEFAULT, asymptote_time, straight_line_factor
from .spectrogram import LogSpectrogram, Spectrogram, log_resample

__all__ = [
    "DetectConfig", "Template", "TemplateBank", "Detection", "DetectionResult",
    "build_bank", "harmonic_stack", "detect", "find_events", "debug_plot", "BAD_FLAGS",
]


@dataclass(frozen=True)
class DetectConfig:
    v_grid: tuple[float, ...] = tuple(float(v) for v in range(25, 161, 5))
    d_min: float = 250.0
    d_max: float = 12000.0
    n_d: int = 16
    vehicle_v: tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 35.0)
    vehicle_d: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0)
    du: float = 0.004
    dt: float = 0.5
    fmin: float = 20.0
    fmax: float = 1000.0
    harmonic_weights: tuple[float, ...] = (1.0, 0.6, 0.4)
    sigma_u: float = 1.5  # ridge half-width (bins of du)
    kernel_half: int = 12  # zero-mean window half-width (bins)
    frac: float = 0.95  # template half-width = time to this fraction of the asymptote
    half_min: float = 15.0
    half_max: float = 120.0
    cand_min: float = 5.0  # candidate threshold (sigma) from the FFT stage
    snr_min: float = 6.0  # acceptance threshold (sigma)
    contrast_min: float = 3.0  # flat-line contrast (sigma)
    contrast_window_s: float = 30.0  # running-mean length that defines "constant frequency"
    nms_t: float = 20.0
    nms_u: float = 0.05
    max_candidates: int = 400
    max_events: int = 40
    n_select: int = 2
    min_aircraft_d: float = 250.0
    min_aircraft_v: float = 25.0
    min_sweep_s: float = 20.0
    c: float = C_DEFAULT
    n_threads: int = 4
    fit: bool = True


@dataclass(eq=False)
class Template:
    v: float
    d: float
    A: np.ndarray  # (Lt, Lu) float32, unit L2 norm
    a0: int  # row (time) index of t_h
    b0: int  # column (u) index of u0 = ln f0
    t_rel: np.ndarray  # (Lt,) seconds relative to t_h
    lg: np.ndarray  # (Lt,) ln g(t) / du (ridge offset in bins)
    vehicle: bool = False

    @property
    def support(self) -> tuple[float, float]:
        return float(self.t_rel[0]), float(self.t_rel[-1])


@dataclass(eq=False)
class TemplateBank:
    templates: list[Template]
    config: DetectConfig

    def __len__(self) -> int:
        return len(self.templates)


@dataclass
class Detection:
    t_h: float
    f0: float
    v: float
    d: float
    snr: float
    contrast: float = 0.0
    template: int = -1
    vehicle_template: bool = False


@dataclass(eq=False)
class DetectionResult:
    events: list[Event]
    detections: list[Detection]
    timings: dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0
    rejected: list[Detection] = field(default_factory=list)  # passed SNR, failed contrast
    duplicates: list[Detection] = field(default_factory=list)  # harmonic ghosts / leftovers of accepted


# --- bank -----------------------------------------------------------------------------------------


def _column_kernel_rows(lg: np.ndarray, rows: np.ndarray, sigma: float, half: int) -> np.ndarray:
    g = np.exp(-0.5 * ((rows[None, :] - lg[:, None]) / sigma) ** 2)
    win = np.abs(rows[None, :] - lg[:, None]) <= half
    g = np.where(win, g, 0.0)
    g -= win * (g.sum(1) / np.maximum(win.sum(1), 1))[:, None]
    return g


def _make_template(v: float, d: float, cfg: DetectConfig, vehicle: bool = False) -> Template:
    w1, w2 = asymptote_time(cfg.frac, v, d, cfg.c)
    w1 = float(np.clip(w1, cfg.half_min, cfg.half_max))
    w2 = float(np.clip(w2, cfg.half_min, cfg.half_max))
    j = np.arange(-math.ceil(w1 / cfg.dt), math.ceil(w2 / cfg.dt) + 1)
    t_rel = j * cfg.dt
    lg = np.log(straight_line_factor(t_rel, v, d, 0.0, cfg.c)) / cfg.du
    pad = cfg.kernel_half + 2
    lo = int(math.floor(lg.min())) - pad
    hi = int(math.ceil(lg.max())) + pad
    rows = np.arange(lo, hi + 1, dtype=float)
    A = _column_kernel_rows(lg, rows, cfg.sigma_u, cfg.kernel_half)
    A /= np.linalg.norm(A)
    return Template(v=float(v), d=float(d), A=A.astype(np.float32), a0=int(np.argmin(np.abs(j))),
                    b0=-lo, t_rel=t_rel, lg=lg, vehicle=vehicle)


def build_bank(cfg: DetectConfig | None = None) -> TemplateBank:
    cfg = cfg or DetectConfig()
    d_grid = np.geomspace(cfg.d_min, cfg.d_max, cfg.n_d)
    tm = [_make_template(v, d, cfg) for v in cfg.v_grid for d in d_grid]
    tm += [_make_template(v, d, cfg, vehicle=True) for v in cfg.vehicle_v for d in cfg.vehicle_d]
    return TemplateBank(templates=tm, config=cfg)


# --- maps -----------------------------------------------------------------------------------------


def _robust(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med, max(mad, 1e-6)


def harmonic_stack(Zl: np.ndarray, du: float, weights=(1.0, 0.6, 0.4)) -> np.ndarray:
    """H(t,u) = sum_h w_h Z(t, u + ln h); terms beyond the top of the band are dropped."""
    H = np.zeros_like(Zl, dtype=np.float32)
    U = Zl.shape[1]
    for h, w in enumerate(weights, start=1):
        k = int(round(math.log(h) / du))
        if k >= U:
            continue
        H[:, : U - k] += w * Zl[:, k:]
    return H


@dataclass(eq=False)
class _Maps:
    ls: LogSpectrogram
    Zl: np.ndarray  # (T,U) float32, masked progressively
    zrow_med: np.ndarray  # (U,) row medians of the unmasked Zl (mask fill value)
    hmed: np.ndarray  # (U,) row stats of unmasked H
    hsig: np.ndarray

    def hn(self, cfg: DetectConfig) -> np.ndarray:
        H = harmonic_stack(self.Zl, cfg.du, cfg.harmonic_weights)
        return ((H - self.hmed) / self.hsig).astype(np.float32)


def _prepare(spec: Spectrogram, cfg: DetectConfig) -> _Maps:
    ls = log_resample(spec, du=cfg.du, dt=cfg.dt, fmin=cfg.fmin, fmax=cfg.fmax)
    Zl = np.ascontiguousarray(ls.Z, dtype=np.float32)
    zmed = np.median(Zl, axis=0).astype(np.float32)
    H = harmonic_stack(Zl, cfg.du, cfg.harmonic_weights)
    sub = H[:: max(1, H.shape[0] // 4000)]
    hmed = np.median(sub, axis=0)
    hsig = 1.4826 * np.median(np.abs(sub - hmed), axis=0)
    hsig = np.maximum(hsig, 0.05).astype(np.float32)
    return _Maps(ls=ls, Zl=Zl, zrow_med=zmed, hmed=hmed.astype(np.float32), hsig=hsig)


def _gather(Hn: np.ndarray, tp: Template, ti: int, ui: int) -> float:
    """Template response with anchor (t_h, u0) at map index (ti, ui); out-of-map parts count 0."""
    T, U = Hn.shape
    Lt, Lu = tp.A.shape
    r0, c0 = ti - tp.a0, ui - tp.b0
    rs, re = max(r0, 0), min(r0 + Lt, T)
    cs, ce = max(c0, 0), min(c0 + Lu, U)
    if rs >= re or cs >= ce:
        return 0.0
    return float(np.sum(tp.A[rs - r0 : re - r0, cs - c0 : ce - c0] * Hn[rs:re, cs:ce]))


def _correlate_bank(Hn: np.ndarray, bank: TemplateBank, n_threads: int):
    """Normalised responses; returns (M max-SNR map, K argmax template, stats (n,2))."""
    T, U = Hn.shape
    lt = max(tp.A.shape[0] for tp in bank.templates)
    lu = max(tp.A.shape[1] for tp in bank.templates)
    P1 = sfft.next_fast_len(T + lt, real=True)
    P2 = sfft.next_fast_len(U + lu, real=True)
    FH = sfft.rfft2(Hn, s=(P1, P2), workers=n_threads)
    stats = np.zeros((len(bank), 2), dtype=np.float64)
    groups = [list(range(i, len(bank), n_threads)) for i in range(n_threads)]

    def run(idx: list[int]):
        M = np.full((T, U), -np.inf, dtype=np.float32)
        K = np.zeros((T, U), dtype=np.int16)
        for k in idx:
            tp = bank.templates[k]
            FT = sfft.rfft2(tp.A, s=(P1, P2), workers=1)
            R = sfft.irfft2(FH * np.conj(FT), s=(P1, P2), workers=1)
            rows = (np.arange(T) - tp.a0) % P1
            cols = (np.arange(U) - tp.b0) % P2
            S = R[rows][:, cols]
            med, sig = _robust(S[::5, ::3])
            stats[k] = (med, sig)
            S -= med
            S /= sig
            better = S > M
            M[better] = S[better]
            K[better] = k
        return M, K

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        parts = list(ex.map(run, [g for g in groups if g]))
    M, K = parts[0]
    for M2, K2 in parts[1:]:
        b = M2 > M
        M[b] = M2[b]
        K[b] = K2[b]
    return M, K, stats


def _candidates(M: np.ndarray, cfg: DetectConfig) -> list[tuple[float, int, int]]:
    nt = 2 * int(round(cfg.nms_t / cfg.dt)) + 1
    nu = 2 * int(round(cfg.nms_u / cfg.du)) + 1
    mf = maximum_filter(M, size=(nt, nu), mode="nearest")
    ti, ui = np.nonzero((M >= mf) & (M >= cfg.cand_min))
    order = np.argsort(-M[ti, ui])[: cfg.max_candidates]
    return [(float(M[ti[i], ui[i]]), int(ti[i]), int(ui[i])) for i in order]


def _deflat(Hn: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Remove constant-frequency structure: subtract each row's running mean over contrast_window_s."""
    n = max(3, int(round(cfg.contrast_window_s / cfg.dt)) | 1)
    return Hn - uniform_filter1d(Hn, size=n, axis=0, mode="nearest")


def _contrast_stats(Hd: np.ndarray, tp: Template, rng: np.random.Generator, n: int = 150):
    T, U = Hd.shape
    ti = rng.integers(0, T, n)
    ui = rng.integers(0, U, n)
    vals = np.array([_gather(Hd, tp, a, b) for a, b in zip(ti, ui, strict=True)])
    return _robust(vals)


def _mask_curve(maps: _Maps, cfg: DetectConfig, t: np.ndarray, f_fund: np.ndarray, widen: float = 1.0):
    """Fill Zl with row medians along h * f_fund(t) for every in-band harmonic (explain-away)."""
    ls = maps.ls
    u0 = ls.u[0]
    U = len(ls.u)
    ti = np.clip(np.round((t - ls.t[0]) / cfg.dt).astype(int), 0, len(ls.t) - 1)
    for h in range(1, 9):
        fh = h * f_fund
        ok = (fh > cfg.fmin) & (fh < cfg.fmax)
        if not ok.any():
            continue
        uc = (np.log(fh[ok]) - u0) / cfg.du
        # ridge half-width in u: ~3 Hz (Hann main lobe) plus a margin
        hw = np.maximum(4.0, 3.5 / fh[ok] / cfg.du) * widen
        for a, c, w in zip(ti[ok], uc, hw, strict=True):
            lo, hi = max(int(c - w), 0), min(int(c + w) + 1, U)
            if lo < hi:
                maps.Zl[a, lo:hi] = maps.zrow_med[lo:hi]


# --- detection ------------------------------------------------------------------------------------


def detect(spec: Spectrogram, config: DetectConfig | None = None) -> list[Detection]:
    """Bank detection only (no curve fit / labelling); see ``find_events`` for the full step."""
    cfg = config or DetectConfig()
    return find_events(spec, DetectConfig(**{**cfg.__dict__, "fit": False})).detections


def find_events(spec: Spectrogram, config: DetectConfig | None = None, bank: TemplateBank | None = None,
                progress=None) -> DetectionResult:
    """Detect, fit and label Doppler events. Returns events sorted by t_h (ids E01, E02, ...)."""
    from .fit import FitConfig, event_from_detection, model_curve

    cfg = config or DetectConfig()
    tim: dict[str, float] = {}
    t0 = time.perf_counter()
    bank = bank or build_bank(cfg)
    maps = _prepare(spec, cfg)
    Hn = maps.hn(cfg)
    tim["prepare"] = time.perf_counter() - t0
    if progress:
        progress(0.1, "template bank correlation")
    t1 = time.perf_counter()
    M, K, stats = _correlate_bank(Hn, bank, cfg.n_threads)
    tim["correlate"] = time.perf_counter() - t1
    cands = _candidates(M, cfg)
    del M
    if progress:
        progress(0.6, f"{len(cands)} candidates")
    t2 = time.perf_counter()
    ls = maps.ls
    rng = np.random.default_rng(1234)
    # heap entries: (-score, version, idx)
    cand = [(ti, ui, int(K[ti, ui])) for _, ti, ui in cands]
    del K
    heap = [(-s, 0, i) for i, (s, _, _) in enumerate(cands)]
    heapq.heapify(heap)
    version = 0
    accepted: list[Detection] = []
    events: list[Event] = []
    rejected: list[Detection] = []
    dropped: list[Detection] = []
    Hd: np.ndarray | None = None
    cstats: dict[int, tuple[float, float]] = {}
    fit_cfg = FitConfig(c=cfg.c)
    while heap and len(accepted) < cfg.max_events:
        negs, ver, i = heapq.heappop(heap)
        ti, ui, k = cand[i]
        tp = bank.templates[k]
        if ver != version:  # stale: re-score on the masked map
            s = (_gather(Hn, tp, ti, ui) - stats[k, 0]) / stats[k, 1]
            if s >= cfg.snr_min:
                heapq.heappush(heap, (-s, version, i))
            continue
        snr = -negs
        if snr < cfg.snr_min:
            break
        if Hd is None:
            Hd = _deflat(Hn, cfg)
        if k not in cstats:
            cstats[k] = _contrast_stats(Hd, tp, rng)
        contrast = (_gather(Hd, tp, ti, ui) - cstats[k][0]) / cstats[k][1]
        det = Detection(t_h=float(ls.t[ti]), f0=float(np.exp(ls.u[ui])), v=tp.v, d=tp.d, snr=float(snr),
                        contrast=float(contrast), template=k, vehicle_template=tp.vehicle)
        if contrast < cfg.contrast_min:
            rejected.append(det)
            continue  # stays in the map; other candidates are unaffected
        ev = event_from_detection(spec, det, fit_cfg) if cfg.fit else None
        # explain away: mask the (fitted, else template) curve
        if ev is not None and "fit-failed" not in ev.quality_flags:
            tt = np.arange(ev.t_h - 200.0, ev.t_h + 200.0, cfg.dt)
            ff = model_curve(ev, tt, c=cfg.c)
            _mask_curve(maps, cfg, tt, ff, widen=1.5)
            if ev.ridge_points:  # real curves deviate from the model: mask the observed ridge too
                rp = np.asarray(ev.ridge_points)
                _mask_curve(maps, cfg, rp[:, 0], rp[:, 1], widen=1.5)
        tt = det.t_h + tp.t_rel
        _mask_curve(maps, cfg, tt, det.f0 * np.exp(tp.lg * cfg.du), widen=1.5)
        Hn = maps.hn(cfg)
        Hd = None
        version += 1
        if ev is not None and (_is_ghost(ev, events) or _is_duplicate(ev, events, cfg.c)):
            dropped.append(det)
            continue
        accepted.append(det)
        if ev is not None:
            events.append(ev)
        if progress:
            progress(0.6 + 0.4 * len(accepted) / cfg.max_events, f"{len(accepted)} events")
    tim["pursuit_fit"] = time.perf_counter() - t2
    if cfg.fit:
        _label_and_select(events, cfg)
    tim["total"] = time.perf_counter() - t0
    return DetectionResult(events=events, detections=accepted, timings=tim, n_candidates=len(cands),
                           rejected=rejected, duplicates=dropped)


_RATIOS = (2.0, 3.0, 4.0, 1.5, 4.0 / 3.0, 2.5, 5.0 / 3.0)


def _is_ghost(ev: Event, events: list[Event]) -> bool:
    """A curve at a small-integer frequency ratio of an accepted event with the same t_h and v."""
    for e in events:
        if abs(e.t_h - ev.t_h) > 5.0 or abs(e.v - ev.v) > 0.2 * max(e.v, 1.0):
            continue
        r = math.log(max(e.f0, ev.f0) / min(e.f0, ev.f0))
        if any(abs(r - math.log(q)) < 0.03 for q in _RATIOS):
            return True
    return False


def _is_duplicate(ev: Event, events: list[Event], c: float, frac: float = 0.6) -> bool:
    """Most of ev's ridge lies on an accepted event's fitted curve (or its x2, x3, /2, /3 family)
    within that event's observed time span: a partial match / leftover of an accepted event."""
    from .fit import model_curve

    if not ev.ridge_points or not events:
        return False
    rp = np.asarray(ev.ridge_points)
    dup = np.zeros(len(rp), bool)
    for e in events:
        if not e.ridge_points:
            continue
        span = np.asarray(e.ridge_points)[:, 0]
        inside = (rp[:, 0] >= span.min() - 5.0) & (rp[:, 0] <= span.max() + 5.0)
        fa = model_curve(e, rp[:, 0], c=c)
        for r in (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0):
            dup |= inside & (np.abs(rp[:, 1] - r * fa) <= np.maximum(2.0, 0.012 * rp[:, 1]))
    return bool(dup.mean() >= frac)


#: flags that make an event unusable as an aircraft event by default
BAD_FLAGS = ("non-aircraft", "fit-failed", "one-sided", "fit-at-bound", "poor-fit")


def _label_and_select(events: list[Event], cfg: DetectConfig) -> None:
    for e in events:
        sweep = float(np.sum(asymptote_time(0.95, max(e.v, 1.0), max(e.d, 1.0), cfg.c)))
        if e.d_range.hi / max(e.d_range.lo, 1.0) > 8.0 or e.v_range.hi - e.v_range.lo > 0.8 * max(e.v, 1.0):
            e.quality_flags.append("poor-fit")
        if (e.d < cfg.min_aircraft_d or e.v < cfg.min_aircraft_v or sweep < cfg.min_sweep_s
                or "sweep-gap" in e.quality_flags):
            if "non-aircraft" not in e.quality_flags:
                e.quality_flags.append("non-aircraft")
    events.sort(key=lambda e: e.t_h)
    for i, e in enumerate(events, start=1):
        e.id = f"E{i:02d}"
    ok = sorted((e for e in events if not set(BAD_FLAGS) & set(e.quality_flags)), key=lambda e: -e.snr)
    chosen = {id(e) for e in ok[: cfg.n_select]}
    for e in events:
        e.selected = id(e) in chosen


# --- debug ----------------------------------------------------------------------------------------


def debug_plot(spec: Spectrogram, events: list[Event], path: str | Path, t_lo: float, t_hi: float,
               f_lo: float = 20.0, f_hi: float = 800.0, c: float = C_DEFAULT, title: str = "") -> Path:
    """PNG of Z in a window with fitted curves (h = 1..3), ridge points and t_h markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .fit import model_curve

    m = (spec.t >= t_lo) & (spec.t <= t_hi)
    fm = (spec.f >= f_lo) & (spec.f <= f_hi)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(spec.Z[m][:, fm].astype(np.float32).T, origin="lower", aspect="auto", cmap="magma",
              extent=(float(spec.t[m][0]), float(spec.t[m][-1]), float(spec.f[fm][0]), float(spec.f[fm][-1])),
              vmin=0, vmax=8)
    tt = np.linspace(t_lo, t_hi, 800)
    for e in events:
        if e.t_h < t_lo - 100 or e.t_h > t_hi + 100:
            continue
        col = "cyan" if "non-aircraft" not in e.quality_flags else "lime"
        ff = model_curve(e, tt, c=c)
        for h in (1, 2, 3):
            ax.plot(tt, h * ff, color=col, lw=0.8, alpha=0.7 if h == 1 else 0.4, ls="-" if h == 1 else "--")
        if e.ridge_points:
            rp = np.array(e.ridge_points)
            ax.plot(rp[:, 0], rp[:, 1], ".", color="white", ms=1.5, alpha=0.6)
        ax.axvline(e.t_h, color=col, lw=0.6, alpha=0.6)
        if not t_lo <= e.t_h <= t_hi:
            continue
        ax.text(e.t_h, f_hi * 0.97,
                f"{e.id} snr {e.snr:.1f}\nv {e.v:.0f} [{e.v_range.lo:.0f},{e.v_range.hi:.0f}]"
                f"\nd {e.d:.0f} [{e.d_range.lo:.0f},{e.d_range.hi:.0f}]\nf0 {e.f0:.1f}",
                color=col, fontsize=7, va="top", clip_on=True)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(f_lo, f_hi)
    ax.set_xlabel("audio time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(title)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p
