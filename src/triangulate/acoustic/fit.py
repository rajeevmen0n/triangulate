"""Curve fitting and uncertainty ranges for Doppler events (plan 3.2 "Curve fit and ranges"), WP08.

Replaces the video's slider tool and its MIN/BEST/MAX snapshots [N section 2 09:38].

1. Ridge points: in each 0.25-s frame (two STFT hops pooled), for every harmonic h, the peak of the
   whitened map Z within +-hw Hz of h * f_model(t) (accepted if Z >= 2.5 and not on the window
   edge), refined to sub-bin precision with a parabola on the raw dB spectrum. Three passes with
   hw = 12, 6, 3 Hz, refitting in between; harmonics with too few points are dropped
   (``harmonics_seen``).
2. Fit (ln f0, v, ln d, t_h) of the straight-line closed form (``physics.straight_line_freq``) by
   ``scipy.optimize.least_squares`` (soft_l1, f_scale 1 Hz). The residual of a harmonic-h point is
   f_obs - h f_model, so harmonics weigh h times more (they are h times more sensitive).
3. Ranges = union of
   * the 95 % Jacobian interval, with sigma estimated from the residuals (i.e. inflated by
     sqrt(chi2/dof)) and further inflated for serial correlation of the residuals,
   * the profile region: each parameter is stepped on a grid while the other three are refitted
     to the ridge points; the region is where the summed brightness of Z under the refitted curve
     stays within 15 % of its peak (above the local noise level),
   * minimum margins v +-10 %, d +-30 %, t_h +-3 s, f0 +-3 %.
   This deliberately reproduces the presenter's generous ranges (real curves are not straight-line
   passes [N section 2 10:00]).

Also: traced polylines (``LINE_POINTS``) -> ``parse_line_points`` and ``fit_trace``.

Public API:
    model_curve(event, t, c=343) -> fundamental frequency of the fitted curve at audio times t
    event_from_detection(spec, detection, cfg) -> core.types.Event
    fit_params(spec, f0, v, d, t_h, cfg) -> FitResult      (fit from any initial guess)
    parse_line_points(obj) -> (n, 2) array of (t, f)
    fit_trace(points, c=343, event_id="T01") -> core.types.Event (traced_line set, "traced" flag)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..core.types import Event, Range
from .physics import C_DEFAULT, asymptote_time, distance_from_slope, from_asymptotes, straight_line_factor
from .spectrogram import Spectrogram

__all__ = [
    "FitConfig", "RidgePoints", "FitResult", "model_curve", "extract_ridge", "fit_params", "sweep_coverage",
    "event_from_detection", "parse_line_points", "fit_trace",
]


@dataclass(frozen=True)
class FitConfig:
    c: float = C_DEFAULT
    pool: int = 2  # STFT frames pooled per ridge frame (0.25 s)
    hw_hz: tuple[float, ...] = (12.0, 6.0, 3.0)
    zmin: float = 2.5
    max_harmonic: int = 4
    half_min: float = 25.0
    half_max: float = 150.0
    min_points: int = 12
    min_harm_frac: float = 0.10  # of frames in the window, to count a harmonic as seen
    profile_keep: float = 0.85  # brightness within 15 % of peak
    margin_v: float = 0.10
    margin_d: float = 0.30
    margin_t: float = 3.0
    margin_f0: float = 0.03
    max_ridge_out: int = 600  # ridge points stored on the Event


@dataclass
class RidgePoints:
    t: np.ndarray
    f: np.ndarray  # observed frequency of harmonic h (Hz)
    h: np.ndarray  # harmonic number
    z: np.ndarray  # peak Z

    def __len__(self) -> int:
        return int(self.t.size)


@dataclass
class FitResult:
    f0: float
    v: float
    d: float
    t_h: float
    ok: bool
    points: RidgePoints
    harmonics: list[int]
    window: tuple[float, float]
    cov: np.ndarray | None = None  # of (ln f0, v, ln d, t_h), inflated
    rms_hz: float = float("nan")
    flags: list[str] = field(default_factory=list)

    @property
    def params(self) -> np.ndarray:
        return np.array([math.log(self.f0), self.v, math.log(self.d), self.t_h])


def model_curve(ev: Any, t, c: float = C_DEFAULT) -> np.ndarray:
    """Fundamental f(t) of an Event (or anything with f0, v, d, t_h attributes)."""
    return float(ev.f0) * straight_line_factor(np.asarray(t, dtype=float), ev.v, ev.d, ev.t_h, c)


def _f(p: np.ndarray, t: np.ndarray, c: float) -> np.ndarray:
    return math.exp(p[0]) * straight_line_factor(t, p[1], math.exp(p[2]), p[3], c)


# --- ridge ----------------------------------------------------------------------------------------


@dataclass(eq=False)
class _Block:
    t: np.ndarray  # pooled frame times
    Z: np.ndarray  # (n, F) float32
    db: np.ndarray  # (n, F) float32
    f: np.ndarray
    df: float


def _block(spec: Spectrogram, t_lo: float, t_hi: float, pool: int) -> _Block:
    i0 = int(np.searchsorted(spec.t, t_lo))
    i1 = int(np.searchsorted(spec.t, t_hi))
    n = (i1 - i0) // pool * pool
    if n <= 0:
        e = np.zeros((0, len(spec.f)), np.float32)
        return _Block(np.zeros(0), e, e, spec.f, spec.df)
    Z = spec.Z[i0 : i0 + n].astype(np.float32).reshape(n // pool, pool, -1).mean(1)
    db = spec.db[i0 : i0 + n].astype(np.float32).reshape(n // pool, pool, -1).mean(1)
    t = spec.t[i0 : i0 + n].reshape(n // pool, pool).mean(1)
    return _Block(t, Z, db, spec.f, spec.df)


def extract_ridge(blk: _Block, p: np.ndarray, harmonics, hw_hz: float, zmin: float, c: float) -> RidgePoints:
    out_t, out_f, out_h, out_z = [], [], [], []
    if blk.t.size == 0:
        return RidgePoints(np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0))
    fm = _f(p, blk.t, c)
    F = blk.f.size
    rows = np.arange(blk.t.size)
    for h in harmonics:
        hw = hw_hz * (1.0 + 0.5 * (h - 1))
        w = max(2, int(math.ceil(hw / blk.df)))
        fc = h * fm
        ok = (fc - hw > blk.f[0]) & (fc + hw < blk.f[-1])
        if not ok.any():
            continue
        kc = np.round((fc[ok] - blk.f[0]) / blk.df).astype(int)
        idx = kc[:, None] + np.arange(-w, w + 1)[None, :]
        idx = np.clip(idx, 0, F - 1)
        r = rows[ok]
        zz = blk.Z[r[:, None], idx]
        j = np.argmax(zz, axis=1)
        zp = zz[np.arange(len(j)), j]
        good = (zp >= zmin) & (j > 0) & (j < 2 * w)
        if not good.any():
            continue
        r, j, idx, zp = r[good], j[good], idx[good], zp[good]
        k = idx[np.arange(len(j)), j]
        a = blk.db[r, np.clip(k - 1, 0, F - 1)]
        b = blk.db[r, k]
        cc = blk.db[r, np.clip(k + 1, 0, F - 1)]
        den = a - 2 * b + cc
        off = np.where(den < -1e-6, 0.5 * (a - cc) / np.where(den < -1e-6, den, -1.0), 0.0)
        off = np.clip(off, -0.5, 0.5)
        out_t.append(blk.t[r])
        out_f.append(blk.f[0] + (k + off) * blk.df)
        out_h.append(np.full(len(r), h))
        out_z.append(zp)
    if not out_t:
        return RidgePoints(np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0))
    return RidgePoints(np.concatenate(out_t), np.concatenate(out_f), np.concatenate(out_h).astype(int),
                       np.concatenate(out_z))


# --- least squares --------------------------------------------------------------------------------

_LO = np.array([-np.inf, 3.0, math.log(5.0), -np.inf])
_HI = np.array([np.inf, 0.7 * C_DEFAULT, math.log(80000.0), np.inf])


def _resid(p, pts: RidgePoints, wts, c):
    return (pts.f - pts.h * _f(p, pts.t, c)) * wts


def _lsq(p0: np.ndarray, pts: RidgePoints, c: float, fixed: int | None = None, max_nfev: int = 200):
    with np.errstate(over="ignore", invalid="ignore"):
        return _lsq_inner(p0, pts, c, fixed, max_nfev)


def _lsq_inner(p0: np.ndarray, pts: RidgePoints, c: float, fixed: int | None, max_nfev: int):
    wts = np.sqrt(np.clip(pts.z / 8.0, 0.25, 1.0))
    lo, hi = _LO.copy(), _HI.copy()
    lo[3], hi[3] = p0[3] - 60.0, p0[3] + 60.0
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
    if fixed is None:
        return least_squares(_resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, x_scale="jac",
                             args=(pts, wts, c), max_nfev=max_nfev)
    free = [i for i in range(4) if i != fixed]

    def fun(q):
        p = p0.copy()
        p[free] = q
        return _resid(p, pts, wts, c)

    r = least_squares(fun, p0[free], bounds=(lo[free], hi[free]), loss="soft_l1", f_scale=1.0,
                      x_scale="jac", max_nfev=max_nfev)
    p = p0.copy()
    p[free] = r.x
    return p, float(np.sum(r.fun**2))


def _window(v: float, d: float, t_h: float, cfg: FitConfig) -> tuple[float, float]:
    w1, w2 = asymptote_time(0.95, max(v, 1.0), max(d, 1.0), cfg.c)
    w1 = float(np.clip(w1, cfg.half_min, cfg.half_max))
    w2 = float(np.clip(w2, cfg.half_min, cfg.half_max))
    return t_h - w1, t_h + w2


def fit_params(spec: Spectrogram, f0: float, v: float, d: float, t_h: float,
               cfg: FitConfig | None = None) -> FitResult:
    """Iterative ridge extraction + robust least squares from an initial (f0, v, d, t_h)."""
    cfg = cfg or FitConfig()
    p = np.array([math.log(f0), v, math.log(d), t_h], dtype=float)
    harmonics = list(range(1, cfg.max_harmonic + 1))
    # the window is fixed from the (generous) initial guess, widened for the larger of init/fit
    win = _window(max(v, 60.0), max(d, 1500.0), t_h, cfg)
    win = (max(win[0], float(spec.t[0])), min(win[1], float(spec.t[-1])))
    blk = _block(spec, *win, cfg.pool)
    pts = RidgePoints(np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0))
    res = None
    flags: list[str] = []
    for it, hw in enumerate(cfg.hw_hz):
        pts = extract_ridge(blk, p, harmonics, hw, cfg.zmin, cfg.c)
        if it == 0 and len(pts):
            n_fr = max(blk.t.size, 1)
            seen = [h for h in harmonics if np.sum(pts.h == h) >= max(6, cfg.min_harm_frac * n_fr)]
            harmonics = seen or [1]
            pts = _subset(pts, harmonics)
        if len(pts) < cfg.min_points:
            break
        res = _lsq(p, pts, cfg.c)
        p = res.x
    if res is None or len(pts) < cfg.min_points:
        return FitResult(f0=f0, v=v, d=d, t_h=t_h, ok=False, points=pts, harmonics=harmonics, window=win,
                         flags=["fit-failed"])
    # final harmonic census around the fitted curve
    allp = extract_ridge(blk, p, range(1, cfg.max_harmonic + 1), cfg.hw_hz[-1], cfg.zmin, cfg.c)
    n_fr = max(blk.t.size, 1)
    need = max(6, cfg.min_harm_frac * n_fr)
    seen = [h for h in range(1, cfg.max_harmonic + 1) if np.sum(allp.h == h) >= need]
    if seen and set(seen) != set(harmonics):
        pts2 = _subset(allp, seen)
        if len(pts2) >= cfg.min_points:
            res = _lsq(p, pts2, cfg.c)
            p, pts, harmonics = res.x, pts2, seen
    r = res.fun
    n = len(r)
    J = res.jac
    cov = None
    try:
        s2 = (1.4826 * np.median(np.abs(r))) ** 2 if n > 8 else float(np.sum(r**2) / max(n - 4, 1))
        s2 = max(s2, float(np.sum(r**2) / max(n - 4, 1)) * 0.25)
        cov = np.linalg.pinv(J.T @ J) * s2 * _serial_factor(pts, r)
    except np.linalg.LinAlgError:
        flags.append("singular-jacobian")
    tl, th = pts.t.min(), pts.t.max()
    if tl > p[3] - 5.0 or th < p[3] + 5.0:
        flags.append("one-sided")
    if win[0] <= spec.t[0] + 1 or win[1] >= spec.t[-1] - 1:
        flags.append("edge")
    resid_hz = pts.f - pts.h * _f(p, pts.t, cfg.c)
    cov_frac = sweep_coverage(pts, p, cfg.c)
    if cov_frac < 0.5:
        flags.append("sweep-gap")
    return FitResult(f0=math.exp(p[0]), v=float(p[1]), d=math.exp(p[2]), t_h=float(p[3]), ok=True, points=pts,
                     harmonics=sorted(harmonics), window=win, cov=cov,
                     rms_hz=float(np.sqrt(np.mean(resid_hz**2))), flags=flags)


def sweep_coverage(pts: RidgePoints, p: np.ndarray, c: float, nbin: int = 8) -> float:
    """Fraction of the central sweep (between 50 % of the approach and recede asymptotes, in log f)
    that is supported by ridge points. Low for hum lines that switch frequency abruptly."""
    v, d, th = float(p[1]), math.exp(p[2]), float(p[3])
    if len(pts) == 0 or v <= 0:
        return 0.0
    b, a = asymptote_time(0.5, v, d, c)
    fa, fb = _f(p, np.array([th - b, th + a]), c)
    lf = np.log(pts.f / pts.h)
    m = (pts.t >= th - 1.5 * b - 2) & (pts.t <= th + 1.5 * a + 2)
    edges = np.linspace(math.log(fb), math.log(fa), nbin + 1)
    hit = np.histogram(lf[m], bins=edges)[0] > 0
    return float(hit.mean())


def _subset(pts: RidgePoints, hs) -> RidgePoints:
    m = np.isin(pts.h, list(hs))
    return RidgePoints(pts.t[m], pts.f[m], pts.h[m], pts.z[m])


def _serial_factor(pts: RidgePoints, r: np.ndarray) -> float:
    """Variance inflation (1+rho)/(1-rho) from lag-1 autocorrelation of time-ordered residuals."""
    rho = []
    for h in np.unique(pts.h):
        m = pts.h == h
        if m.sum() < 8:
            continue
        o = np.argsort(pts.t[m])
        x = r[m][o]
        x = x - x.mean()
        den = float(np.sum(x * x))
        if den > 0:
            rho.append(float(np.sum(x[1:] * x[:-1]) / den))
    if not rho:
        return 1.0
    rr = float(np.clip(np.mean(rho), 0.0, 0.9))
    return (1 + rr) / (1 - rr)


# --- brightness profile ---------------------------------------------------------------------------


def _brightness(blk: _Block, p: np.ndarray, harmonics, c: float) -> float:
    if blk.t.size == 0:
        return 0.0
    fm = _f(p, blk.t, c)
    tot, wsum = 0.0, 0.0
    for h in harmonics:
        fh = h * fm
        ok = (fh > blk.f[0]) & (fh < blk.f[-1] - blk.df)
        if not ok.any():
            continue
        x = (fh[ok] - blk.f[0]) / blk.df
        k = np.floor(x).astype(int)
        a = x - k
        r = np.nonzero(ok)[0]
        z = blk.Z[r, k] * (1 - a) + blk.Z[r, k + 1] * a
        w = 1.0 / h
        tot += w * float(z.sum())
        wsum += w * blk.t.size
    return tot / max(wsum, 1e-9)


def _profile_ranges(blk: _Block, fr: FitResult, cfg: FitConfig) -> dict[int, tuple[float, float]]:
    p = fr.params
    pts = fr.points
    b_best = _brightness(blk, p, fr.harmonics, cfg.c)
    # local noise level: brightness of the same curve shifted off the ridge in log-frequency
    noise = np.median([_brightness(blk, p + np.array([s, 0, 0, 0]), fr.harmonics, cfg.c)
                       for s in (-0.12, -0.08, 0.08, 0.12)])
    keep = noise + cfg.profile_keep * (b_best - noise)
    grids = {
        0: p[0] + np.linspace(-0.06, 0.06, 13),
        1: p[1] * (1 + np.linspace(-0.4, 0.4, 17)),
        2: p[2] + np.linspace(-1.0, 1.0, 17),
        3: p[3] + np.linspace(-10.0, 10.0, 21),
    }
    out = {}
    for q, grid in grids.items():
        ok = np.zeros(grid.size, bool)
        for i, g in enumerate(grid):
            p0 = p.copy()
            p0[q] = g
            if q == 1 and not (_LO[1] < g < _HI[1]):
                continue
            try:
                pf, _ = _lsq(p0, pts, cfg.c, fixed=q, max_nfev=60)
            except ValueError:
                continue
            ok[i] = _brightness(blk, pf, fr.harmonics, cfg.c) >= keep
        c0 = int(np.argmin(np.abs(grid - p[q])))
        ok[c0] = True
        lo = hi = c0
        while lo - 1 >= 0 and ok[lo - 1]:
            lo -= 1
        while hi + 1 < grid.size and ok[hi + 1]:
            hi += 1
        out[q] = (float(grid[lo]), float(grid[hi]))
    return out


# --- events ---------------------------------------------------------------------------------------


def _ranges(fr: FitResult, prof: dict[int, tuple[float, float]] | None, cfg: FitConfig) -> dict[str, Range]:
    p = fr.params
    sd = np.sqrt(np.clip(np.diag(fr.cov), 0, None)) if fr.cov is not None else np.full(4, np.inf)
    sd = np.where(np.isfinite(sd), sd, 0.0)
    ci = [(p[i] - 1.96 * sd[i], p[i] + 1.96 * sd[i]) for i in range(4)]
    marg = [(p[0] + math.log(1 - cfg.margin_f0), p[0] + math.log(1 + cfg.margin_f0)),
            (p[1] * (1 - cfg.margin_v), p[1] * (1 + cfg.margin_v)),
            (p[2] + math.log(1 - cfg.margin_d), p[2] + math.log(1 + cfg.margin_d)),
            (p[3] - cfg.margin_t, p[3] + cfg.margin_t)]
    lo = [min(ci[i][0], marg[i][0], (prof or {}).get(i, (np.inf, 0))[0]) for i in range(4)]
    hi = [max(ci[i][1], marg[i][1], (prof or {}).get(i, (0, -np.inf))[1]) for i in range(4)]
    lo[1] = max(lo[1], 1.0)
    hi[1] = min(hi[1], 0.9 * cfg.c)
    # keep log-parameter intervals sane before exponentiating
    lo[0], hi[0] = max(lo[0], p[0] - 0.7), min(hi[0], p[0] + 0.7)
    lo[2], hi[2] = max(lo[2], p[2] - 3.0), min(hi[2], p[2] + 3.0)
    return {
        "f0": Range(lo=math.exp(lo[0]), hi=math.exp(hi[0])),
        "v": Range(lo=lo[1], hi=hi[1]),
        "d": Range(lo=math.exp(lo[2]), hi=math.exp(hi[2])),
        "t_h": Range(lo=lo[3], hi=hi[3]),
    }


def _ridge_out(pts: RidgePoints, n: int) -> list[tuple[float, float]]:
    o = np.argsort(pts.t, kind="stable")
    t, f = pts.t[o], (pts.f / pts.h)[o]
    if t.size > n:
        sel = np.linspace(0, t.size - 1, n).astype(int)
        t, f = t[sel], f[sel]
    return [(round(float(a), 3), round(float(b), 3)) for a, b in zip(t, f, strict=True)]


def event_from_detection(spec: Spectrogram, det: Any, cfg: FitConfig | None = None,
                         event_id: str = "E00") -> Event:
    """Fit a detection (anything with t_h, f0, v, d, snr) and build an Event with ranges."""
    cfg = cfg or FitConfig()
    fr = fit_params(spec, det.f0, det.v, det.d, det.t_h, cfg)
    flags = list(fr.flags)
    if getattr(det, "vehicle_template", False):
        flags.append("vehicle-template")
    if fr.ok and not (_LO[1] + 0.5 < fr.v < _HI[1] - 0.5 and math.log(6.0) < math.log(fr.d) < _HI[2] - 0.1):
        flags.append("fit-at-bound")
    prof = None
    if fr.ok and not {"one-sided", "sweep-gap", "fit-at-bound"} & set(flags):  # unusable: skip the profile
        blk = _block(spec, *fr.window, cfg.pool)
        prof = _profile_ranges(blk, fr, cfg)
    rg = _ranges(fr, prof, cfg)
    return Event(
        id=event_id, t_h=fr.t_h, t_h_range=rg["t_h"], f0=fr.f0, f0_range=rg["f0"], v=fr.v, v_range=rg["v"],
        d=fr.d, d_range=rg["d"], snr=float(getattr(det, "snr", 0.0)), harmonics_seen=fr.harmonics,
        ridge_points=_ridge_out(fr.points, cfg.max_ridge_out), quality_flags=flags,
    )


# --- traced polylines (LINE_POINTS) ---------------------------------------------------------------


def parse_line_points(obj: Any) -> np.ndarray:
    """Parse a traced polyline into an (n, 2) array of (audio time s, frequency Hz), sorted by time.

    Accepts: a path to / string of JSON; a dict with key LINE_POINTS / line_points / points /
    traced_line; a list of [t, f] pairs or of {"t": .., "f": ..} dicts; or "t,f" text lines.
    """
    if isinstance(obj, Path) or (isinstance(obj, str) and not obj.lstrip().startswith(("[", "{"))
                                 and Path(obj).suffix in (".json", ".txt", ".csv") and Path(obj).exists()):
        obj = Path(obj).read_text()
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith(("[", "{")):
            obj = json.loads(s)
        else:
            lines = [ln for ln in s.splitlines() if ln.strip() and not ln.startswith("#")]
            rows = [ln.replace(";", ",").split(",") for ln in lines]
            obj = [[float(a), float(b)] for a, b, *_ in rows]
    if isinstance(obj, dict):
        for key in ("LINE_POINTS", "line_points", "points", "traced_line"):
            if key in obj:
                return parse_line_points(obj[key])
        raise ValueError("no LINE_POINTS/points key")
    pts = []
    for it in obj:
        if isinstance(it, dict):
            t = it.get("t", it.get("time"))
            f = it.get("f", it.get("freq", it.get("frequency")))
            pts.append((float(t), float(f)))
        else:
            pts.append((float(it[0]), float(it[1])))
    a = np.array(pts, dtype=float).reshape(-1, 2)
    if a.shape[0] < 3:
        raise ValueError("need >= 3 traced points")
    if not np.all(np.isfinite(a)) or np.any(a[:, 1] <= 0):
        raise ValueError("bad traced points")
    return a[np.argsort(a[:, 0], kind="stable")]


def fit_trace(points: Any, c: float = C_DEFAULT, event_id: str = "T01",
              cfg: FitConfig | None = None) -> Event:
    """Fit the straight-line model to a traced polyline (densified to 0.25 s by linear interpolation)."""
    cfg = cfg or FitConfig(c=c)
    a = parse_line_points(points)
    t = np.arange(a[0, 0], a[-1, 0] + 1e-9, 0.25) if a[-1, 0] - a[0, 0] > 1.0 else a[:, 0]
    f = np.interp(t, a[:, 0], a[:, 1])
    fmax, fmin = float(f.max()), float(f.min())
    f0i, vi = from_asymptotes(fmax, fmin, c)
    f0i, vi = float(f0i), float(max(vi, 10.0)) * 1.1
    k = int(np.argmin(np.abs(np.log(f) - math.log(f0i))))
    th = float(t[k])
    sl = np.gradient(f, t)[k] if t.size > 2 else (f[-1] - f[0]) / (t[-1] - t[0])
    di = float(np.clip(distance_from_slope(f0i, vi, min(sl, -1e-3), c), 100.0, 30000.0))
    pts = RidgePoints(t=t, f=f, h=np.ones(t.size, int), z=np.full(t.size, 8.0))
    p = np.array([math.log(f0i), vi, math.log(di), th])
    best = None
    for dv in (1.0, 0.7, 1.4):  # a few starts; the trace may not reach the asymptotes
        r = _lsq(p * np.array([1, dv, 1, 1]), pts, c)
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    p = best.x
    r = best.fun
    s2 = float(np.sum(r**2) / max(r.size - 4, 1))
    fr = FitResult(f0=math.exp(p[0]), v=float(p[1]), d=math.exp(p[2]), t_h=float(p[3]), ok=True, points=pts,
                   harmonics=[1], window=(float(t[0]), float(t[-1])),
                   cov=np.linalg.pinv(best.jac.T @ best.jac) * max(s2, 0.25) * _serial_factor(pts, r))
    rg = _ranges(fr, None, cfg)
    flags = ["traced"]
    if not (t[0] < fr.t_h - 5 and t[-1] > fr.t_h + 5):
        flags.append("one-sided")
    return Event(id=event_id, t_h=fr.t_h, t_h_range=rg["t_h"], f0=fr.f0, f0_range=rg["f0"], v=fr.v,
                 v_range=rg["v"], d=fr.d, d_range=rg["d"], snr=0.0, harmonics_seen=[1],
                 ridge_points=[], traced_line=[(float(x), float(y)) for x, y in a], quality_flags=flags)

