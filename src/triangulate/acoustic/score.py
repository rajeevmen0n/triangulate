"""Level 5 of the acoustic search: locked forward-model scoring (plan 3.3.6, WP11).

For a hypothesis h (observer cell x, implied start o, one flight track P_k per selected event k):

1. Reception frames of event k: the *model* window i in [tau_k - W1_k, tau_k + W2_k] (audio-relative,
   0.25-s frames), W = time for |f - f0| to reach 90 % of the asymptote (``physics.asymptote_time``)
   from the event's fitted (v, d), clipped to [20, 120] s.
2. Emission times on the **real interpolated track** (``physics.nb_emission_time``, warm-started
   Newton, numba) for reception times o + t_i give g_k(t_i) = c / (c + v_r). Frames whose emission time
   falls outside the track segment are excluded (``coverage``).
3. Brightness score [N 2 17:56]: with T_s = o + dt the frame heard at absolute o + t_i sits at audio
   position t_i - dt, so

       S_k(dt, f0) = (1 / n_valid) sum_i H_k(t_i - dt, ln f0 + ln g_k(t_i))

   where H_k(t, u) = sum_h w_h Z(t, u + ln h) is the whitened, clipped SNR map Z resampled to
   u = ln f (du = 0.002, i.e. 0.2-0.3 Hz at 100-150 Hz, so f0 is a pure index shift) and pooled to
   0.25 s, harmonic-stacked over the event's ``harmonics_seen`` (w = 1, 0.6, 0.4, 0.3); harmonics
   above 1 kHz drop out. Before stacking, Z is made zero-mean across log-frequency (``ridge_filter``,
   running mean over +-12 u bins = +-2.4 % subtracted): the plain clipped Z has a broad ridge and a
   positive noise mean, so a curve 1-2 Hz off the ridge (a timing error traded against f0) scored
   almost as well as the right one; the filter roughly triples the dt contrast on synthetic passes.
4. **Locked** [N 2 18:27-20:12]: one shared dt (a correction to T_s) for all events, f0 free per event:
   ``S(h) = max_dt sum_k max_f0 S_k(dt, f0)``. With a calibration the sum is taken over z_k (so a
   traced and a brightness event lock on comparable scales); otherwise over raw S_k. dt is swept over
   +-5 s in 0.5-s steps, then +-0.25 s around the best, then refined by a parabola (sub-frame).
5. Traced / faint events (``Event.traced_line`` set) use the distance-to-line score instead
   [N 2 22:47]: ``S_k = -mean_t min(|f0 g(t + dt) - f_trace(t)| / sigma_f, 5)`` over the trace frames.

The unlocked score (each event at its own best dt) is reported as ``z_sum_unlocked`` for diagnostics.

Public API
    ScoreConfig, ScoreMap.from_spectrogram(spec, events, cfg), TrackBank.from_tracks(prepared)
    score_hypotheses(smap, bank, hyps, observers, calib=None, n_threads=None) -> polars DataFrame
    score_pairs(...) (low level: arrays in, (S, J, coverage) out), event_curve(...) for evidence plots
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np
import polars as pl

from ..core.types import Event
from .physics import C_DEFAULT, Track, asymptote_time, nb_emission_time, nb_track_state, solve_emission
from .spectrogram import Spectrogram, log_resample

if TYPE_CHECKING:
    from .calibrate import Calibration
    from .cpa import TrackSeg
    from .grid import ObserverCells

__all__ = [
    "ScoreConfig",
    "EventPlan",
    "ScoreMap",
    "TrackBank",
    "PairSet",
    "score_pairs",
    "score_hypotheses",
    "observer_lookup",
    "event_curve",
    "HARMONIC_WEIGHTS",
]

HARMONIC_WEIGHTS = (1.0, 0.6, 0.4, 0.3)  # w_h for h = 1..4 (plan 3.2)


@dataclass(frozen=True)
class ScoreConfig:
    du: float = 0.002  # log-frequency step of the scoring map (f0 sweep step = du * f0)
    frame_dt: float = 0.25  # time pooling of the scoring map (s); dt is swept in multiples of it
    fmin: float = 20.0
    fmax: float = 1000.0
    harmonics: str = "seen"  # "seen": the event's harmonics_seen; "all": h = 1..3
    ridge_half: int = 12  # zero-mean ridge filter half-width in u bins (+-2.4 % in f); 0 = plain Z
    dt_max: float = 5.0  # shared dt sweep +- (s) [N 2 06:37]
    dt_coarse: float = 0.5  # coarse dt step (s); then +-1 frame around the best
    frac: float = 0.9  # window half-widths: time to this fraction of the asymptote
    w_min: float = 20.0
    w_max: float = 120.0
    f0_margin: float = 0.005  # relative widening of the event's f0 range
    min_coverage: float = 0.3  # fraction of window frames with a valid emission time
    line_sigma_hz: float = 2.0  # traced-line score scale
    line_cap: float = 5.0
    z_missing: float = 0.0  # z of an event whose track does not cover the window
    c: float = C_DEFAULT

    @property
    def n_coarse(self) -> int:
        return 2 * int(round(self.dt_max / self.dt_coarse)) + 1

    @property
    def coarse_step(self) -> int:
        return max(1, int(round(self.dt_coarse / self.frame_dt)))


# --- scoring map ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class EventPlan:
    """How event k is scored: model frames [i0, i0 + n), f0 grid u-indices [j0, j0 + nj), map `hset`."""

    id: str
    kind: str  # "brightness" | "line"
    tau: float
    i0: int
    n: int
    j0: int
    nj: int
    hset: int
    w: tuple[float, float]  # (W1, W2) seconds
    tr_a0: int = 0  # line: first audio frame of the trace
    tr: np.ndarray | None = None  # line: trace frequency per audio frame (Hz)


def _stack(Zl: np.ndarray, du: float, weights: Sequence[float]) -> np.ndarray:
    """H(t, u) = sum_h w_h Z(t, u + ln h), the fractional shift ln h / du linearly interpolated."""
    H = np.zeros_like(Zl, dtype=np.float32)
    U = Zl.shape[1]
    for h, w in enumerate(weights, start=1):
        if w == 0.0:
            continue
        x = math.log(h) / du
        k = int(math.floor(x))
        a = np.float32(x - k)
        if k + 1 < U:
            H[:, : U - k - 1] += np.float32(w) * ((1 - a) * Zl[:, k : U - 1] + a * Zl[:, k + 1 :])
    return H


def ridge_filter(Zl: np.ndarray, half: int) -> np.ndarray:
    """Zero-mean across log-frequency: Z - (running mean over +-half u bins). Makes a curve that is
    off the ridge by ~1 Hz score below one on it (time/f0 trade-off sharper) and noise zero-mean."""
    if half <= 0:
        return Zl
    from scipy.ndimage import uniform_filter1d

    return (Zl - uniform_filter1d(Zl, size=2 * half + 1, axis=1, mode="nearest")).astype(np.float32)


def _harm_weights(ev: Event, mode: str) -> tuple[float, ...]:
    if mode == "all":
        return HARMONIC_WEIGHTS[:3]
    seen = {h for h in (ev.harmonics_seen or [1]) if 1 <= h <= len(HARMONIC_WEIGHTS)} or {1}
    return tuple(HARMONIC_WEIGHTS[h - 1] if h in seen else 0.0 for h in range(1, max(seen) + 1))


@dataclass(eq=False)
class ScoreMap:
    """Harmonic-stacked log-frequency maps H (n_sets, T, U) plus per-event scoring plans."""

    H: np.ndarray  # float32 (n_sets, T, U)
    t0: float  # audio time of frame 0 (s)
    fdt: float  # frame step (s)
    u0: float  # ln f of column 0
    du: float
    duration: float  # audio duration (s)
    events: list[Event]
    plans: list[EventPlan]
    cfg: ScoreConfig
    hsets: list[tuple[float, ...]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.H.shape[1])

    def frame_time(self, i) -> np.ndarray:
        return self.t0 + np.asarray(i) * self.fdt

    def f0_of(self, k: int, j) -> np.ndarray:
        return np.exp(self.u0 + (self.plans[k].j0 + np.asarray(j)) * self.du)

    @classmethod
    def from_spectrogram(
        cls,
        spec: Spectrogram,
        events: Sequence[Event],
        cfg: ScoreConfig | None = None,
    ) -> ScoreMap:
        cfg = cfg or ScoreConfig()
        ls = log_resample(spec, du=cfg.du, dt=cfg.frame_dt, fmin=cfg.fmin, fmax=cfg.fmax)
        duration = float(spec.t[-1] + (spec.t[1] - spec.t[0]) / 2) if len(spec.t) > 1 else 0.0
        return cls.from_log_map(
            ls.Z, float(ls.t[0]), float(ls.t[1] - ls.t[0]), float(ls.u[0]), cfg.du, events, cfg, duration
        )

    @classmethod
    def from_log_map(
        cls,
        Zl: np.ndarray,
        t0: float,
        fdt: float,
        u0: float,
        du: float,
        events: Sequence[Event],
        cfg: ScoreConfig,
        duration: float | None = None,
    ) -> ScoreMap:
        Zl = ridge_filter(np.ascontiguousarray(Zl, dtype=np.float32), cfg.ridge_half)
        evs = list(events)
        hsets: list[tuple[float, ...]] = []
        for ev in evs:
            w = _harm_weights(ev, cfg.harmonics)
            if ev.traced_line is None and w not in hsets:
                hsets.append(w)
        if not hsets:
            hsets = [(1.0,)]
        H = np.empty((len(hsets),) + Zl.shape, dtype=np.float32)
        for i, w in enumerate(hsets):
            H[i] = _stack(Zl, du, w)
        dur = float(duration) if duration is not None else t0 + Zl.shape[0] * fdt
        sm = cls(H=H, t0=t0, fdt=fdt, u0=u0, du=du, duration=dur, events=evs, plans=[], cfg=cfg, hsets=hsets)
        sm.plans = [sm._plan(ev) for ev in evs]
        return sm

    def _plan(self, ev: Event) -> EventPlan:
        cfg = self.cfg
        lo = ev.f0_range.lo * (1.0 - cfg.f0_margin)
        hi = ev.f0_range.hi * (1.0 + cfg.f0_margin)
        j0 = int(math.floor((math.log(max(lo, 1.0)) - self.u0) / self.du))
        j1 = int(math.ceil((math.log(max(hi, 1.0)) - self.u0) / self.du))
        nj = max(1, j1 - j0 + 1)
        if ev.traced_line is not None:
            a = np.asarray(ev.traced_line, dtype=float).reshape(-1, 2)
            a = a[np.argsort(a[:, 0], kind="stable")]
            ia = int(math.ceil((a[0, 0] - self.t0) / self.fdt))
            ib = int(math.floor((a[-1, 0] - self.t0) / self.fdt))
            if ib < ia:
                raise ValueError(f"event {ev.id}: traced line shorter than one frame")
            tt = self.t0 + np.arange(ia, ib + 1) * self.fdt
            tr = np.interp(tt, a[:, 0], a[:, 1]).astype(np.float64)
            return EventPlan(
                id=ev.id,
                kind="line",
                tau=float(ev.t_h),
                i0=ia,
                n=ib - ia + 1,
                j0=j0,
                nj=nj,
                hset=-1,
                w=(float(ev.t_h - a[0, 0]), float(a[-1, 0] - ev.t_h)),
                tr_a0=ia,
                tr=tr,
            )
        w1, w2 = asymptote_time(cfg.frac, max(ev.v, 1.0), max(ev.d, 1.0), cfg.c)
        w1 = float(np.clip(w1, cfg.w_min, cfg.w_max))
        w2 = float(np.clip(w2, cfg.w_min, cfg.w_max))
        i0 = int(round((ev.t_h - w1 - self.t0) / self.fdt))
        i1 = int(round((ev.t_h + w2 - self.t0) / self.fdt))
        return EventPlan(
            id=ev.id,
            kind="brightness",
            tau=float(ev.t_h),
            i0=i0,
            n=i1 - i0 + 1,
            j0=j0,
            nj=nj,
            hset=self.hsets.index(_harm_weights(ev, cfg.harmonics)),
            w=(w1, w2),
        )

    def shift_ok(self, k: int, shift: int, half: int) -> bool:
        """Is the audio window of event k fully inside the recording for shifts shift +- half frames?"""
        p = self.plans[k]
        a_lo = p.i0 - shift - half
        a_hi = p.i0 + p.n - 1 - shift + half
        if p.kind == "line":  # the trace is fixed in audio; the model moves
            return True
        return a_lo >= 0 and a_hi < self.n_frames


# --- tracks --------------------------------------------------------------------------------------


@dataclass(eq=False)
class TrackBank:
    """Concatenated track segments for the numba kernels. Times are relative to each segment start."""

    keys: list[tuple[str, int]]
    tt: np.ndarray  # (N,) float64, relative to tref of the segment
    pos: np.ndarray  # (N, 3)
    vel: np.ndarray  # (N, 3)
    off: np.ndarray  # (S,) int64 start of segment s
    length: np.ndarray  # (S,) int64
    tref: np.ndarray  # (S,) float64 absolute start time
    index: dict[tuple[str, int], int] = field(default_factory=dict)

    @classmethod
    def from_tracks(cls, tracks: pl.DataFrame | Sequence[Track] | Sequence[TrackSeg]) -> TrackBank:
        from .cpa import track_segments

        segs = track_segments(tracks, min_samples=2)
        if not segs:
            raise ValueError("no track segments")
        lens = np.array([sg.t.size for sg in segs], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        tref = np.array([float(sg.t[0]) for sg in segs])
        tt = np.concatenate([sg.t - sg.t[0] for sg in segs])
        pos = np.ascontiguousarray(np.concatenate([sg.pos for sg in segs]))
        vel = np.ascontiguousarray(np.concatenate([sg.vel for sg in segs]))
        keys = [(sg.icao, int(sg.seg)) for sg in segs]
        return cls(
            keys=keys,
            tt=tt,
            pos=pos,
            vel=vel,
            off=off,
            length=lens,
            tref=tref,
            index={k: i for i, k in enumerate(keys)},
        )

    def seg_index(self, icao: Sequence[str], seg: Sequence[int]) -> np.ndarray:
        try:
            return np.array(
                [self.index[(str(a), int(b))] for a, b in zip(icao, seg, strict=True)], dtype=np.int64
            )
        except KeyError as e:
            raise KeyError(f"track segment {e} not in the bank") from None

    def track(self, s: int) -> Track:
        a, n = int(self.off[s]), int(self.length[s])
        return Track(
            t=self.tt[a : a + n] + self.tref[s],
            pos=self.pos[a : a + n],
            vel=self.vel[a : a + n],
            label=self.keys[s][0],
        )


# --- numba kernels -------------------------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _solve_g(tt, pos, vel, ox, oy, oz, o_rel, t0, fdt, i0, n, c, g, valid):
    """g = c/(c+v_r) at reception times o_rel + t0 + (i0+i)*fdt (track-relative), warm-started."""
    hint = 0
    te = 0.0
    tr_prev = 0.0
    vr_prev = 0.0
    t_lo = tt[0]
    t_hi = tt[tt.shape[0] - 1]
    nv = 0
    for i in range(n):
        t_r = o_rel + t0 + (i0 + i) * fdt
        if i == 0:
            hint, px, py, pz, vx, vy, vz = nb_track_state(tt, pos, vel, t_r, 0)
            dx = px - ox
            dy = py - oy
            dz = pz - oz
            guess = t_r - math.sqrt(dx * dx + dy * dy + dz * dz) / c
        else:
            guess = te + (t_r - tr_prev) * c / (c + vr_prev)
        te, r, vr, hint, it = nb_emission_time(t_r, tt, pos, vel, ox, oy, oz, c, guess, hint, 1e-6, 30)
        tr_prev = t_r
        vr_prev = vr
        g[i] = c / (c + vr)
        ok = t_lo <= te <= t_hi
        valid[i] = ok
        if ok:
            nv += 1
    return nv


@nb.njit(cache=True, parallel=True)
def _bright_kernel(
    H,
    t0,
    fdt,
    du,
    tt_all,
    pos_all,
    vel_all,
    s_off,
    s_len,
    s_tref,
    p_seg,
    p_obs,
    p_o,
    p_ev,
    p_shift0,
    p_rev,
    e_i0,
    e_n,
    e_j0,
    e_nj,
    e_h,
    n_dt,
    step,
    c,
    min_cov,
    S,
    J,
    cov,
):
    T = H.shape[1]
    U = H.shape[2]
    half = n_dt // 2
    for p in nb.prange(p_seg.shape[0]):  # type: ignore[attr-defined]
        s = p_seg[p]
        a0 = s_off[s]
        ln = s_len[s]
        tt = tt_all[a0 : a0 + ln]
        pos = pos_all[a0 : a0 + ln]
        vel = vel_all[a0 : a0 + ln]
        e = p_ev[p]
        i0 = e_i0[e]
        n = e_n[e]
        j0 = e_j0[e]
        nj = e_nj[e]
        hs = e_h[e]
        g = np.empty(n)
        valid = np.empty(n, np.bool_)
        nv = _solve_g(
            tt,
            pos,
            vel,
            p_obs[p, 0],
            p_obs[p, 1],
            p_obs[p, 2],
            p_o[p] - s_tref[s],
            t0,
            fdt,
            i0,
            n,
            c,
            g,
            valid,
        )
        cov[p] = nv / n
        if nv < min_cov * n or nv == 0:
            for k in range(n_dt):
                S[p, k] = np.nan
                J[p, k] = -1
            continue
        off = np.empty(n, np.int64)
        for i in range(n):
            ii = n - 1 - i if p_rev[p] else i
            off[i] = int(math.floor(math.log(g[ii]) / du + 0.5))
        vv = np.empty(n, np.bool_)
        for i in range(n):
            vv[i] = valid[n - 1 - i] if p_rev[p] else valid[i]
        acc = np.empty(nj, np.float32)
        inv = 1.0 / nv
        for k in range(n_dt):
            d = p_shift0[p] + (k - half) * step
            acc[:] = 0.0
            for i in range(n):
                if not vv[i]:
                    continue
                a = i0 + i - d
                if a < 0 or a >= T:
                    continue
                base = j0 + off[i]
                jlo = 0 if base >= 0 else -base
                jhi = nj if base + nj <= U else U - base
                for j in range(jlo, jhi):
                    acc[j] += H[hs, a, base + j]
            best = 0
            bv = acc[0]
            for j in range(1, nj):
                if acc[j] > bv:
                    bv = acc[j]
                    best = j
            S[p, k] = bv * inv
            J[p, k] = best


@nb.njit(cache=True, parallel=True)
def _line_kernel(
    t0,
    fdt,
    u0,
    du,
    tt_all,
    pos_all,
    vel_all,
    s_off,
    s_len,
    s_tref,
    p_seg,
    p_obs,
    p_o,
    p_ev,
    p_shift0,
    p_rev,
    e_a0,
    e_n,
    e_j0,
    e_nj,
    tr_all,
    tr_off,
    n_dt,
    step,
    c,
    sigma,
    cap,
    min_cov,
    S,
    J,
    cov,
):
    half = n_dt // 2
    for p in nb.prange(p_seg.shape[0]):  # type: ignore[attr-defined]
        s = p_seg[p]
        a0 = s_off[s]
        ln = s_len[s]
        tt = tt_all[a0 : a0 + ln]
        pos = pos_all[a0 : a0 + ln]
        vel = vel_all[a0 : a0 + ln]
        e = p_ev[p]
        ta0 = e_a0[e]
        n = e_n[e]
        j0 = e_j0[e]
        nj = e_nj[e]
        tr = tr_all[tr_off[e] : tr_off[e] + n]
        dlo = p_shift0[p] - half * step
        m = n + 2 * half * step  # model frames ta0 + dlo ... ta0 + dlo + m - 1
        g = np.empty(m)
        valid = np.empty(m, np.bool_)
        _solve_g(
            tt,
            pos,
            vel,
            p_obs[p, 0],
            p_obs[p, 1],
            p_obs[p, 2],
            p_o[p] - s_tref[s],
            t0,
            fdt,
            ta0 + dlo,
            m,
            c,
            g,
            valid,
        )
        if p_rev[p]:
            g2 = g[::-1].copy()
            v2 = valid[::-1].copy()
            g = g2
            valid = v2
        # coverage at the central shift
        nv0 = 0
        for i in range(n):
            if valid[i + half * step]:
                nv0 += 1
        cov[p] = nv0 / n
        if nv0 < min_cov * n or nv0 == 0:
            for k in range(n_dt):
                S[p, k] = np.nan
                J[p, k] = -1
            continue
        acc = np.empty(nj)
        for k in range(n_dt):
            sh = k * step  # model index offset into g for trace frame 0
            acc[:] = 0.0
            cnt = 0
            for i in range(n):
                if not valid[sh + i]:
                    continue
                cnt += 1
                gi = g[sh + i]
                for j in range(nj):
                    f0 = math.exp(u0 + (j0 + j) * du)
                    x = abs(f0 * gi - tr[i]) / sigma
                    acc[j] += x if x < cap else cap
            if cnt < min_cov * n or cnt == 0:
                S[p, k] = np.nan
                J[p, k] = -1
                continue
            best = 0
            bv = acc[0]
            for j in range(1, nj):
                if acc[j] < bv:
                    bv = acc[j]
                    best = j
            S[p, k] = -bv / cnt
            J[p, k] = best


# --- pairs ---------------------------------------------------------------------------------------


@dataclass(eq=False)
class PairSet:
    """One (hypothesis, event) evaluation each: track segment, observer, implied start o."""

    seg: np.ndarray  # int64 (P,)
    obs: np.ndarray  # float64 (P, 3)
    o: np.ndarray  # float64 (P,) absolute start (epoch s, track time base)
    ev: np.ndarray  # int64 (P,) event index
    shift0: np.ndarray | None = None  # int64 (P,) centre of the dt sweep (frames)
    rev: np.ndarray | None = None  # bool (P,) time-reversed model curve (decoys)

    def __len__(self) -> int:
        return int(self.seg.size)

    def subset(self, sel) -> PairSet:
        return PairSet(
            self.seg[sel],
            self.obs[sel],
            self.o[sel],
            self.ev[sel],
            None if self.shift0 is None else self.shift0[sel],
            None if self.rev is None else self.rev[sel],
        )


def score_pairs(
    smap: ScoreMap,
    bank: TrackBank,
    pairs: PairSet,
    n_dt: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the kernels. Returns S (P, n_dt) float32 (NaN = not covered), J (P, n_dt) int32 best f0
    index into the event's f0 grid, coverage (P,) float32. Shift k is ``shift0 + (k - n_dt//2) * step``
    frames, i.e. dt = shift * fdt seconds (T_s = o + dt)."""
    P = len(pairs)
    S = np.full((P, n_dt), np.nan, dtype=np.float32)
    J = np.full((P, n_dt), -1, dtype=np.int32)
    cov = np.zeros(P, dtype=np.float32)
    if P == 0:
        return S, J, cov
    cfg = smap.cfg
    shift0 = pairs.shift0 if pairs.shift0 is not None else np.zeros(P, np.int64)
    rev = pairs.rev if pairs.rev is not None else np.zeros(P, np.bool_)
    kinds = np.array([pl_.kind == "line" for pl_ in smap.plans])
    is_line = kinds[pairs.ev]
    common = (bank.tt, bank.pos, bank.vel, bank.off, bank.length, bank.tref)
    b = np.flatnonzero(~is_line)
    if b.size:
        e_i0 = np.array([p.i0 for p in smap.plans], np.int64)
        e_n = np.array([p.n for p in smap.plans], np.int64)
        e_j0 = np.array([p.j0 for p in smap.plans], np.int64)
        e_nj = np.array([p.nj for p in smap.plans], np.int64)
        e_h = np.array([max(p.hset, 0) for p in smap.plans], np.int64)
        Sb = np.empty((b.size, n_dt), np.float32)
        Jb = np.empty((b.size, n_dt), np.int32)
        cb = np.empty(b.size, np.float32)
        _bright_kernel(
            smap.H,
            float(smap.t0),
            float(smap.fdt),
            float(smap.du),
            *common,
            pairs.seg[b],
            np.ascontiguousarray(pairs.obs[b]),
            pairs.o[b],
            pairs.ev[b],
            shift0[b].astype(np.int64),
            rev[b].astype(np.bool_),
            e_i0,
            e_n,
            e_j0,
            e_nj,
            e_h,
            int(n_dt),
            int(step),
            float(cfg.c),
            float(cfg.min_coverage),
            Sb,
            Jb,
            cb,
        )
        S[b], J[b], cov[b] = Sb, Jb, cb
    li = np.flatnonzero(is_line)
    if li.size:
        trs = [p.tr if p.tr is not None else np.zeros(0) for p in smap.plans]
        tr_off = np.concatenate([[0], np.cumsum([t.size for t in trs])[:-1]]).astype(np.int64)
        tr_all = np.concatenate(trs).astype(np.float64)
        e_a0 = np.array([p.tr_a0 for p in smap.plans], np.int64)
        e_n = np.array([p.n for p in smap.plans], np.int64)
        e_j0 = np.array([p.j0 for p in smap.plans], np.int64)
        e_nj = np.array([p.nj for p in smap.plans], np.int64)
        Sl = np.empty((li.size, n_dt), np.float32)
        Jl = np.empty((li.size, n_dt), np.int32)
        cl = np.empty(li.size, np.float32)
        _line_kernel(
            float(smap.t0),
            float(smap.fdt),
            float(smap.u0),
            float(smap.du),
            *common,
            pairs.seg[li],
            np.ascontiguousarray(pairs.obs[li]),
            pairs.o[li],
            pairs.ev[li],
            shift0[li].astype(np.int64),
            rev[li].astype(np.bool_),
            e_a0,
            e_n,
            e_j0,
            e_nj,
            tr_all,
            tr_off,
            int(n_dt),
            int(step),
            float(cfg.c),
            float(cfg.line_sigma_hz),
            float(cfg.line_cap),
            float(cfg.min_coverage),
            Sl,
            Jl,
            cl,
        )
        S[li], J[li], cov[li] = Sl, Jl, cl
    return S, J, cov


# --- hypotheses ----------------------------------------------------------------------------------


def observer_lookup(observers: ObserverCells, cells: np.ndarray) -> np.ndarray:
    """Observer ENU positions (n, 3) for packed cell keys (must all be present)."""
    keys = observers.cells.keys
    order = np.argsort(keys)
    sk = keys[order]
    pos = np.searchsorted(sk, cells)
    pos = np.clip(pos, 0, max(sk.size - 1, 0))
    if sk.size == 0 or np.any(sk[pos] != cells):
        raise KeyError("hypothesis cells missing from the observer set")
    return observers.xyz[order[pos]]


def _n_events(hyps: pl.DataFrame) -> int:
    return len([c for c in hyps.columns if c.startswith("icao")])


def _hyp_pairs(bank: TrackBank, hyps: pl.DataFrame, obs_xyz: np.ndarray, K: int) -> PairSet:
    H = hyps.height
    seg = (
        np.concatenate(
            [bank.seg_index(hyps[f"icao{k}"].to_list(), hyps[f"seg{k}"].to_list()) for k in range(K)]
        )
        if H
        else np.zeros(0, np.int64)
    )
    o = np.tile(hyps["o"].to_numpy().astype(np.float64), K)
    ev = np.repeat(np.arange(K, dtype=np.int64), H)
    return PairSet(seg=seg, obs=np.ascontiguousarray(np.tile(obs_xyz, (K, 1))), o=o, ev=ev)


def _parabola(ym: np.ndarray, y0: np.ndarray, yp: np.ndarray) -> np.ndarray:
    den = ym - 2.0 * y0 + yp
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(den < -1e-12, 0.5 * (ym - yp) / den, 0.0)
    return np.clip(np.nan_to_num(x), -0.5, 0.5)


def score_hypotheses(
    smap: ScoreMap,
    bank: TrackBank,
    hyps: pl.DataFrame,
    observers: ObserverCells | np.ndarray,
    calib: Calibration | None = None,
) -> pl.DataFrame:
    """Locked score for every hypothesis row (columns of ``matching.match_offsets``: hid, cell, o,
    icao_k, seg_k ...; event k of the hypothesis is ``smap.events[k]``).

    Adds: dt (s), t_s = o + dt, S (raw locked sum), S_k, f0_k, cov_k, and with a calibration z_k,
    z_sum (= locked sum of z) and z_sum_unlocked (each event at its own best dt).
    """
    cfg = smap.cfg
    K = _n_events(hyps)
    if K != len(smap.events):
        raise ValueError(f"hypotheses have {K} events, the score map {len(smap.events)}")
    Hn = hyps.height
    if Hn == 0:
        return hyps.with_columns(
            *(pl.lit(None, pl.Float64).alias(c) for c in ["dt", "t_s", "S", "z_sum", "z_sum_unlocked"])
        )
    xyz = (
        observers
        if isinstance(observers, np.ndarray)
        else observer_lookup(observers, hyps["cell"].to_numpy())
    )
    pairs = _hyp_pairs(bank, hyps, xyz, K)
    mu = np.zeros(K)
    sig = np.ones(K)
    if calib is not None:
        mu = np.array([c.mu for c in calib.events])
        sig = np.array([c.sigma for c in calib.events])

    def zed(S: np.ndarray) -> np.ndarray:  # (K, H, n) raw -> z, NaN -> z_missing
        z = (S - mu[:, None, None]) / sig[:, None, None] if calib is not None else S.astype(np.float64)
        return np.where(np.isfinite(z), z, cfg.z_missing if calib is not None else 0.0)

    # coarse sweep
    nc, st = cfg.n_coarse, cfg.coarse_step
    S, J, cov = score_pairs(smap, bank, pairs, nc, st)
    S3 = S.reshape(K, Hn, nc)
    Z3 = zed(S3)
    tot = Z3.sum(axis=0)
    kc = np.argmax(tot, axis=1)
    unlocked = Z3.max(axis=2).sum(axis=0)
    shift_c = (kc - nc // 2) * st
    # fine sweep (+-1 frame) around the coarse best, shared across events
    pairs.shift0 = np.tile(shift_c, K).astype(np.int64)
    Sf, Jf, _ = score_pairs(smap, bank, pairs, 3, 1)
    Sf3 = Sf.reshape(K, Hn, 3)
    Jf3 = Jf.reshape(K, Hn, 3)
    Zf3 = zed(Sf3)
    totf = Zf3.sum(axis=0)
    kf = np.argmax(totf, axis=1)
    rows = np.arange(Hn)
    frac = _parabola(totf[rows, np.maximum(kf - 1, 0)], totf[rows, kf], totf[rows, np.minimum(kf + 1, 2)])
    frac = np.where((kf == 0) | (kf == 2), 0.0, frac)
    shift = shift_c + (kf - 1)
    dt = np.clip((shift + frac) * smap.fdt, -cfg.dt_max - smap.fdt, cfg.dt_max + smap.fdt)
    Sk = Sf3[:, rows, kf]  # (K, H)
    Jk = Jf3[:, rows, kf]
    Zk = Zf3[:, rows, kf]
    cols: dict[str, Any] = {
        "dt": dt,
        "t_s": hyps["o"].to_numpy() + dt,
        "S": np.nansum(Sk, axis=0),
    }
    for k in range(K):
        cols[f"S_{k}"] = Sk[k].astype(np.float64)
        cols[f"f0_{k}"] = np.where(Jk[k] >= 0, smap.f0_of(k, np.maximum(Jk[k], 0)), np.nan)
        cols[f"cov_{k}"] = cov.reshape(K, Hn)[k].astype(np.float64)
        if calib is not None:
            cols[f"z_{k}"] = Zk[k]
    if calib is not None:
        cols["z_sum"] = Zk.sum(axis=0)
        cols["z_sum_unlocked"] = np.maximum(unlocked, Zk.sum(axis=0))
    else:
        cols["z_sum"] = np.full(Hn, np.nan)
        cols["z_sum_unlocked"] = np.full(Hn, np.nan)
    cols["S_unlocked"] = np.nansum(S3.max(axis=2), axis=0) if calib is None else np.full(Hn, np.nan)
    return hyps.with_columns(*(pl.Series(k, v) for k, v in cols.items()))


# --- evidence ------------------------------------------------------------------------------------


def event_curve(
    track: Track,
    obs,
    t_s: float,
    tau: np.ndarray,
    f0: float,
    c: float = C_DEFAULT,
) -> np.ndarray:
    """Modelled fundamental (Hz) at audio positions ``tau`` for a recording that starts at t_s
    (NaN where the emission time is outside the track), for overlays (WP20 evidence)."""
    em = solve_emission(np.asarray(tau, dtype=float) + t_s, track, obs, c)
    return np.where(em.valid, f0 * em.g, np.nan)
