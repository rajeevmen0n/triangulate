"""Doppler physics core (plan 3.1): closed-form straight-line curves and the retarded-time solver.

Conventions (shared by detection WP08, CPA WP09, scoring WP11 and synthesis):

* SI units throughout: metres, seconds, m/s, Hz. ``c`` is the speed of sound (default 343 m/s).
* Positions are Cartesian metres in one local frame (ENU from ``core.geo.geodetic_to_enu``, z up).
  The observer ``obs`` is a point in the same frame.
* ``v_r`` is the radial velocity of the source away from the observer, evaluated at the
  *emission* time: **v_r > 0 when receding** (frequency below f0), < 0 when approaching.
* Doppler (static observer, moving source): ``f = f0 * c / (c + v_r)``. ``g = c / (c + v_r)`` is
  the "Doppler factor"; ``ln f = ln f0 + ln g`` so f0 is a pure shift in log frequency.
* Time bases. ``t_r`` = reception (audio) time, ``t_e`` = emission time, both in the *same* base
  as the track's ``t`` (epoch seconds, or seconds relative to anything). They satisfy the retarded
  equation ``t_r = t_e + |P(t_e) - X| / c``. Audio position tau maps to ``t_r = T_s + tau``.
* Straight-line pass parameters: ``v`` speed, ``d`` slant CPA distance (> 0), ``t_c`` = emission
  time of closest approach, ``t_h = t_c + d / c`` = **heard** CPA time (where v_r = 0, f = f0).
  The closed-form helpers are parametrised by ``t_h`` (what the spectrogram shows); ``s = t_e - t_c``.

Closed form (derived from c(tau - s) = sqrt(d^2 + v^2 s^2) with tau = t_r - t_c, causal root):

    s   = [c^2 tau - sqrt(c^2 (v^2 tau^2 + d^2) - v^2 d^2)] / (c^2 - v^2)
    r   = sqrt(d^2 + v^2 s^2),   v_r = v^2 s / r
    df/dt_r = -f0 g^3 v^2 d^2 / (c r^3)

Slopes. At t_h the slope is ``-f0 v^2 / (c d)``; this is what the video's "max RoC" readout shows
(131.5*117^2/(343*4280) = 1.23 vs 1.222 on screen; 130*107^2/(343*3230) = 1.343 exactly). The
*true* maximum of |df/dt_r| is larger by (1 - v^2/c^2)^(-3/2) and occurs at ``t_r = t_c``
(d/c before t_h, where v_r = -v^2/c). ``cpa_slope`` returns the former, ``max_slope`` the latter.

Track-based solver: ``solve_emission`` solves the retarded equation per reception time on a sampled
track (positions + node velocities, cubic Hermite interpolation so velocity is continuous) with a
warm-started Newton iteration. F(t_e) = t_e + r(t_e)/c - t_r has F' = 1 + v_r/c in [1-v/c, 1+v/c],
so for v < 0.5c every Newton step contracts the error by at least 2/3 globally and quadratically
near the root; it converges to < 1e-9 s in 2-4 steps. Kernels are numba ``@njit`` functions; the
scalar kernel ``nb_emission_time`` can be called from other numba kernels (WP11 scoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba as nb
import numpy as np

__all__ = [
    "C_DEFAULT", "speed_of_sound",
    "doppler_factor", "doppler_shift", "radial_velocity", "heard_time", "emission_cpa_time",
    "straight_emission_offset", "straight_line_state", "straight_line_factor", "straight_line_freq",
    "straight_line_slope", "straight_time_of_vr",
    "cpa_slope", "max_slope", "asymptotes", "from_asymptotes", "distance_from_slope", "asymptote_time",
    "Track", "Emission", "Pass", "solve_emission", "track_passes",
    "nb_track_state", "nb_emission_time",
]

C_DEFAULT = 343.0  # m/s [N section 4]


def speed_of_sound(temperature_c: float | None = None) -> float:
    """c in m/s: 343 by default, else 331.3*sqrt(1 + T/273.15) for air at T degrees Celsius."""
    if temperature_c is None:
        return C_DEFAULT
    return 331.3 * math.sqrt(1.0 + float(temperature_c) / 273.15)


# --- Basic Doppler relations (numpy broadcasting) -----------------------------------------------


def doppler_factor(v_r, c: float = C_DEFAULT):
    """g = c / (c + v_r); v_r > 0 receding."""
    return c / (c + np.asarray(v_r, dtype=float))


def doppler_shift(f0, v_r, c: float = C_DEFAULT):
    """Received frequency f = f0 * c / (c + v_r)."""
    return np.asarray(f0, dtype=float) * doppler_factor(v_r, c)


def radial_velocity(pos, vel, obs):
    """v_r = vel . (pos - obs) / |pos - obs| over the last axis (size 3). Positive when receding."""
    dp = np.asarray(pos, dtype=float) - np.asarray(obs, dtype=float)
    r = np.linalg.norm(dp, axis=-1)
    return np.sum(np.asarray(vel, dtype=float) * dp, axis=-1) / r


def heard_time(t_c, d, c: float = C_DEFAULT):
    """Heard (retarded) CPA time t_h = t_c + d/c, valid for any track (v_r = 0 at the CPA)."""
    return np.asarray(t_c, dtype=float) + np.asarray(d, dtype=float) / c


def emission_cpa_time(t_h, d, c: float = C_DEFAULT):
    """Inverse of ``heard_time``: t_c = t_h - d/c."""
    return np.asarray(t_h, dtype=float) - np.asarray(d, dtype=float) / c


# --- Straight-line closed form --------------------------------------------------------------------


def straight_emission_offset(tau, v, d, c: float = C_DEFAULT):
    """s = t_e - t_c for a straight constant-speed pass, given tau = t_r - t_c (broadcasts).

    Uses the cancellation-free rationalised root for tau > 0 (near the CPA tau ~ d/c the textbook
    form subtracts two nearly equal numbers).
    """
    tau = np.asarray(tau, dtype=float)
    v = np.asarray(v, dtype=float)
    d = np.asarray(d, dtype=float)
    c2, v2 = c * c, v * v
    sq = np.sqrt(c2 * v2 * tau * tau + d * d * (c2 - v2))
    with np.errstate(divide="ignore", invalid="ignore"):
        s_pos = (c2 * tau * tau - d * d) / (c2 * tau + sq)  # tau > 0: denominator > 0
        s_neg = (c2 * tau - sq) / (c2 - v2)  # tau <= 0: no cancellation
    return np.where(tau > 0, s_pos, s_neg)


@dataclass(frozen=True)
class StraightState:
    s: np.ndarray  # t_e - t_c (s)
    r: np.ndarray  # source-observer distance at emission (m)
    v_r: np.ndarray  # radial velocity at emission (m/s, + receding)
    g: np.ndarray  # Doppler factor c/(c+v_r)


def straight_line_state(t_r, v, d, t_h, c: float = C_DEFAULT) -> StraightState:
    """Emission offset, range, radial velocity and Doppler factor of a straight pass (broadcasts)."""
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    tau = np.asarray(t_r, dtype=float) - (np.asarray(t_h, dtype=float) - d / c)
    s = straight_emission_offset(tau, v, d, c)
    r = np.sqrt(d * d + v * v * s * s)
    v_r = v * v * s / r
    return StraightState(s=s, r=r, v_r=v_r, g=c / (c + v_r))


def straight_line_factor(t_r, v, d, t_h, c: float = C_DEFAULT):
    """Doppler factor g(t_r) = f/f0 of a straight pass; ln g is the shape in log frequency."""
    return straight_line_state(t_r, v, d, t_h, c).g


def straight_line_freq(t_r, f0, v, d, t_h, c: float = C_DEFAULT):
    """f(t_r) = f0 * g(t_r) for a straight constant-speed pass heard at t_h."""
    return np.asarray(f0, dtype=float) * straight_line_factor(t_r, v, d, t_h, c)


def straight_line_slope(t_r, f0, v, d, t_h, c: float = C_DEFAULT):
    """Analytic df/dt_r (Hz/s, negative for a pass) = -f0 g^3 v^2 d^2 / (c r^3)."""
    st = straight_line_state(t_r, v, d, t_h, c)
    v = np.asarray(v, dtype=float)
    d = np.asarray(d, dtype=float)
    return -np.asarray(f0, dtype=float) * st.g**3 * v * v * d * d / (c * st.r**3)


def straight_time_of_vr(v_r, v, d, c: float = C_DEFAULT):
    """Reception time relative to t_h at which a straight pass has radial velocity v_r (|v_r| < v)."""
    v = np.asarray(v, dtype=float)
    d = np.asarray(d, dtype=float)
    beta = np.asarray(v_r, dtype=float) / v  # = v s / r
    r = d / np.sqrt(1.0 - beta * beta)
    s = beta * r / v
    return s + (r - d) / c


def cpa_slope(f0, v, d, c: float = C_DEFAULT):
    """Slope df/dt_r at the heard CPA t_h: -f0 v^2/(c d) (the video's "max RoC" readout, signed)."""
    return -np.asarray(f0, dtype=float) * np.asarray(v, dtype=float) ** 2 / (c * np.asarray(d, dtype=float))


def max_slope(f0, v, d, c: float = C_DEFAULT):
    """True max |df/dt_r| = f0 v^2 c^2 / (d (c^2 - v^2)^1.5), reached at t_r = t_c = t_h - d/c."""
    v = np.asarray(v, dtype=float)
    return np.asarray(f0, dtype=float) * v * v * c * c / (np.asarray(d, dtype=float) * (c * c - v * v) ** 1.5)


def distance_from_slope(f0, v, slope, c: float = C_DEFAULT):
    """Invert ``cpa_slope``: d = f0 v^2 / (c |slope at t_h|)."""
    return np.asarray(f0, dtype=float) * np.asarray(v, dtype=float) ** 2 / (c * np.abs(slope))


def asymptotes(f0, v, c: float = C_DEFAULT):
    """(f_pre, f_post) = (f0 c/(c - v), f0 c/(c + v)): far approach / far recede frequencies."""
    f0 = np.asarray(f0, dtype=float)
    v = np.asarray(v, dtype=float)
    return f0 * c / (c - v), f0 * c / (c + v)


def from_asymptotes(f_pre, f_post, c: float = C_DEFAULT):
    """Invert ``asymptotes``: f0 = harmonic mean of f_pre and f_post, v = c (R-1)/(R+1), R = f_pre/f_post.

    v does not depend on f0 (plan 3.1 identifiability).
    """
    f_pre = np.asarray(f_pre, dtype=float)
    f_post = np.asarray(f_post, dtype=float)
    ratio = f_pre / f_post
    return 2.0 * f_pre * f_post / (f_pre + f_post), c * (ratio - 1.0) / (ratio + 1.0)


def asymptote_time(frac, v, d, c: float = C_DEFAULT):
    """Durations (before, after) t_h at which |f - f0| reaches ``frac`` (0..1) of its asymptote.

    Both are positive seconds (in reception time; the two sides differ). Used for template
    half-widths (frac 0.95, WP08) and scoring windows W_k (frac 0.9, WP11).
    """
    q = np.asarray(frac, dtype=float)
    v = np.asarray(v, dtype=float)
    a = q * v / (c + v)  # receding: v_r/(c+v_r) = a
    vr_after = a * c / (1.0 - a)
    b = q * v / (c - v)  # approaching: w/(c-w) = b, v_r = -w
    vr_before = -b * c / (1.0 + b)
    return -straight_time_of_vr(vr_before, v, d, c), straight_time_of_vr(vr_after, v, d, c)


# --- Sampled tracks and numba kernels -------------------------------------------------------------


@nb.njit(cache=True, nogil=True)
def _locate(tt, t, hint):
    """Segment index i with tt[i] <= t < tt[i+1]; -1 before the start, n-1 at/after the end."""
    n = tt.shape[0]
    if t < tt[0]:
        return -1
    if t >= tt[n - 1]:
        return n - 1
    if 0 <= hint < n - 1 and tt[hint] <= t < tt[hint + 1]:
        return hint
    if 0 <= hint < n - 2 and tt[hint + 1] <= t < tt[hint + 2]:
        return hint + 1
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if tt[mid] <= t:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit(cache=True, nogil=True)
def nb_track_state(tt, pos, vel, t, hint):
    """Cubic-Hermite position and velocity at time t. Returns (i, px, py, pz, vx, vy, vz).

    Outside [tt[0], tt[-1]] the track is extrapolated at constant end velocity (i = -1 / n-1).
    ``tt`` (n,), ``pos``/``vel`` (n, 3) float64; ``hint`` is a segment index guess (any int).
    """
    n = tt.shape[0]
    i = _locate(tt, t, hint)
    if i < 0 or i >= n - 1:
        k = 0 if i < 0 else n - 1
        dt = t - tt[k]
        return (i, pos[k, 0] + vel[k, 0] * dt, pos[k, 1] + vel[k, 1] * dt, pos[k, 2] + vel[k, 2] * dt,
                vel[k, 0], vel[k, 1], vel[k, 2])
    h = tt[i + 1] - tt[i]
    u = (t - tt[i]) / h
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = (u3 - 2.0 * u2 + u) * h
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = (u3 - u2) * h
    d00 = (6.0 * u2 - 6.0 * u) / h
    d10 = 3.0 * u2 - 4.0 * u + 1.0
    d01 = (-6.0 * u2 + 6.0 * u) / h
    d11 = 3.0 * u2 - 2.0 * u
    px = h00 * pos[i, 0] + h10 * vel[i, 0] + h01 * pos[i + 1, 0] + h11 * vel[i + 1, 0]
    py = h00 * pos[i, 1] + h10 * vel[i, 1] + h01 * pos[i + 1, 1] + h11 * vel[i + 1, 1]
    pz = h00 * pos[i, 2] + h10 * vel[i, 2] + h01 * pos[i + 1, 2] + h11 * vel[i + 1, 2]
    vx = d00 * pos[i, 0] + d10 * vel[i, 0] + d01 * pos[i + 1, 0] + d11 * vel[i + 1, 0]
    vy = d00 * pos[i, 1] + d10 * vel[i, 1] + d01 * pos[i + 1, 1] + d11 * vel[i + 1, 1]
    vz = d00 * pos[i, 2] + d10 * vel[i, 2] + d01 * pos[i + 1, 2] + d11 * vel[i + 1, 2]
    return i, px, py, pz, vx, vy, vz


@nb.njit(cache=True, nogil=True)
def nb_emission_time(t_r, tt, pos, vel, ox, oy, oz, c, t_guess, hint, tol, max_iter):
    """Solve t_e + |P(t_e) - X|/c = t_r by safeguarded Newton from ``t_guess``.

    Returns (t_e, r, v_r, hint, n_iter). ``hint`` is the segment index for the next warm start.
    Precondition for guaranteed convergence: track speed < 0.5c (|F'| in [0.5, 1.5]).
    """
    t = t_guess
    it = 0
    while it < max_iter:
        it += 1
        hint, px, py, pz, vx, vy, vz = nb_track_state(tt, pos, vel, t, hint)
        dx = px - ox
        dy = py - oy
        dz = pz - oz
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        vr = (dx * vx + dy * vy + dz * vz) / r if r > 0.0 else 0.0
        fp = 1.0 + vr / c
        if fp < 0.05:  # only reachable for |v| ~ c
            fp = 0.05
        step = (t + r / c - t_r) / fp
        t -= step
        if abs(step) <= tol:
            break
    hint, px, py, pz, vx, vy, vz = nb_track_state(tt, pos, vel, t, hint)
    dx = px - ox
    dy = py - oy
    dz = pz - oz
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    vr = (dx * vx + dy * vy + dz * vz) / r if r > 0.0 else 0.0
    return t, r, vr, hint, it


@nb.njit(cache=True, nogil=True)
def _solve_series(t_r, tt, pos, vel, ox, oy, oz, c, tol, max_iter, t_e, r, vr, iters):
    """Warm-started solve for a series of reception times (any order; sorted is fastest)."""
    m = t_r.shape[0]
    hint = 0
    te_prev = 0.0
    tr_prev = 0.0
    vr_prev = 0.0
    for j in range(m):
        if j == 0:
            hint, px, py, pz, vx, vy, vz = nb_track_state(tt, pos, vel, t_r[0], 0)
            dx = px - ox
            dy = py - oy
            dz = pz - oz
            guess = t_r[0] - math.sqrt(dx * dx + dy * dy + dz * dz) / c  # one fixed-point step
        else:
            guess = te_prev + (t_r[j] - tr_prev) * c / (c + vr_prev)  # dt_e/dt_r = g
        te, rr, v, hint, k = nb_emission_time(t_r[j], tt, pos, vel, ox, oy, oz, c, guess, hint, tol, max_iter)
        t_e[j] = te
        r[j] = rr
        vr[j] = v
        iters[j] = k
        te_prev = te
        tr_prev = t_r[j]
        vr_prev = v


@nb.njit(cache=True, parallel=True)
def _solve_batch(t_r, tt, pos, vel, obs, c, tol, max_iter, t_e, r, vr, iters):
    for k in nb.prange(obs.shape[0]):  # type: ignore[attr-defined]
        _solve_series(t_r, tt, pos, vel, obs[k, 0], obs[k, 1], obs[k, 2], c, tol, max_iter,
                      t_e[k], r[k], vr[k], iters[k])


@nb.njit(cache=True)
def _states(tt, pos, vel, t, p_out, v_out):
    hint = 0
    for j in range(t.shape[0]):
        hint, px, py, pz, vx, vy, vz = nb_track_state(tt, pos, vel, t[j], hint)
        p_out[j, 0] = px
        p_out[j, 1] = py
        p_out[j, 2] = pz
        v_out[j, 0] = vx
        v_out[j, 1] = vy
        v_out[j, 2] = vz


def _f64c(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class Track:
    """A sampled source trajectory: times ``t`` (n,) strictly increasing (s, any base), positions
    ``pos`` (n, 3) and node velocities ``vel`` (n, 3) in local Cartesian metres (ENU, z up).

    Interpolation is cubic Hermite through (pos, vel), so position is C1 and velocity continuous;
    a constant-velocity line is reproduced exactly. Outside the span it extrapolates linearly.
    """

    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        t, pos, vel = _f64c(self.t), _f64c(self.pos), _f64c(self.vel)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("track needs >= 2 samples")
        if pos.shape != (t.size, 3) or vel.shape != (t.size, 3):
            raise ValueError("pos and vel must have shape (n, 3)")
        if not np.all(np.diff(t) > 0):
            raise ValueError("track times must be strictly increasing")
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise ValueError("non-finite track values")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "vel", vel)

    @classmethod
    def from_positions(cls, t, pos, vel=None, label: str = "") -> Track:
        """Build from sampled positions; node velocities default to 2nd-order finite differences."""
        t = _f64c(t)
        pos = _f64c(pos)
        if vel is None:
            vel = np.gradient(pos, t, axis=0, edge_order=2 if t.size > 2 else 1)
        return cls(t=t, pos=pos, vel=vel, label=label)

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def speed(self) -> np.ndarray:
        """|vel| at the nodes (n,)."""
        return np.linalg.norm(self.vel, axis=1)

    def state(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated (pos, vel), each shape t.shape + (3,)."""
        ta = np.asarray(t, dtype=np.float64)
        flat = np.ascontiguousarray(ta.reshape(-1))
        p = np.empty((flat.size, 3))
        v = np.empty((flat.size, 3))
        tref = self.t[0]
        _states(self.t - tref, self.pos, self.vel, flat - tref, p, v)
        return p.reshape(ta.shape + (3,)), v.reshape(ta.shape + (3,))


@dataclass(frozen=True, eq=False)
class Emission:
    """Solution of the retarded equation. Arrays have shape t_r.shape, or (k,) + t_r.shape for k
    observers. ``valid`` is False where t_e falls outside the track span (extrapolated)."""

    t_e: np.ndarray  # emission time, same base as t_r / track.t
    r: np.ndarray  # range at emission (m)
    v_r: np.ndarray  # radial velocity at emission (m/s, + receding)
    valid: np.ndarray  # bool
    c: float
    iterations: np.ndarray  # Newton steps per sample (diagnostic)

    @property
    def g(self) -> np.ndarray:
        """Doppler factor c / (c + v_r) = f / f0 = dt_e/dt_r."""
        return self.c / (self.c + self.v_r)

    def freq(self, f0: float, harmonic: int = 1) -> np.ndarray:
        return harmonic * f0 * self.g


def solve_emission(t_r, track: Track, obs, c: float = C_DEFAULT, tol: float = 1e-9,
                   max_iter: int = 50) -> Emission:
    """Emission times for reception times ``t_r`` at observer(s) ``obs`` on a sampled track.

    ``obs``: shape (3,) -> outputs shaped like t_r; shape (k, 3) -> outputs (k,) + t_r.shape
    (numba ``prange`` over observers). Times are shifted to the track start internally, so epoch
    seconds keep full precision. ``tol`` is the Newton step tolerance in seconds.
    """
    tr = np.asarray(t_r, dtype=np.float64)
    shape = tr.shape
    tref = track.t[0]
    trf = np.ascontiguousarray(tr.reshape(-1) - tref)
    tt = track.t - tref
    ob = np.asarray(obs, dtype=np.float64)
    m = trf.size
    if ob.shape == (3,):
        t_e, r, vr = np.empty(m), np.empty(m), np.empty(m)
        iters = np.empty(m, dtype=np.int32)
        _solve_series(trf, tt, track.pos, track.vel, ob[0], ob[1], ob[2], float(c), float(tol),
                      int(max_iter), t_e, r, vr, iters)
        out_shape = shape
    elif ob.ndim == 2 and ob.shape[1] == 3:
        k = ob.shape[0]
        t_e, r, vr = np.empty((k, m)), np.empty((k, m)), np.empty((k, m))
        iters = np.empty((k, m), dtype=np.int32)
        _solve_batch(trf, tt, track.pos, track.vel, np.ascontiguousarray(ob), float(c), float(tol),
                     int(max_iter), t_e, r, vr, iters)
        out_shape = (k,) + shape
    else:
        raise ValueError("obs must have shape (3,) or (k, 3)")
    valid = (t_e >= tt[0]) & (t_e <= tt[-1])
    return Emission(
        t_e=(t_e + tref).reshape(out_shape), r=r.reshape(out_shape), v_r=vr.reshape(out_shape),
        valid=valid.reshape(out_shape), c=float(c), iterations=iters.reshape(out_shape),
    )


# --- Reference CPA finder (truth for synthetic scenes; oracle for WP09) -------------------------


@dataclass(frozen=True)
class Pass:
    """One closest approach of a track to an observer (a local minimum of range over emission time)."""

    t_cpa: float  # emission time of the CPA (= t_c), track time base
    d: float  # slant CPA distance (m)
    v: float  # 3-D speed at the CPA (m/s)
    t_h: float  # heard CPA time t_cpa + d/c (where f = f0)


def track_passes(track: Track, obs, c: float = C_DEFAULT, r_max: float = math.inf,
                 step: float = 0.25) -> list[Pass]:
    """All interior local minima of |P(t) - obs| (r_min < r_max), refined to ~1e-9 s.

    A brute-force reference (dense sampling, then a root of v_r(t) = dr/dt by Brent's method), not
    the fast grid CPA. Minima at the very ends of the track are not passes (still approaching).
    """
    from scipy.optimize import brentq, minimize_scalar

    ob = np.asarray(obs, dtype=float)
    n = max(3, int(math.ceil((track.t_end - track.t_start) / step)) + 1)
    ts = np.linspace(track.t_start, track.t_end, n)
    p, _ = track.state(ts)
    rr = np.linalg.norm(p - ob, axis=1)
    idx = np.where((rr[1:-1] <= rr[:-2]) & (rr[1:-1] < rr[2:]))[0] + 1
    out: list[Pass] = []

    def rng(t: float) -> float:
        pp, _ = track.state(np.array([t]))
        return float(np.linalg.norm(pp[0] - ob))

    def vrad(t: float) -> float:
        pp, vv = track.state(np.array([t]))
        return float(radial_velocity(pp[0], vv[0], ob))

    for i in idx:
        if rr[i] >= r_max:
            continue
        a, b = ts[i - 1], ts[i + 1]
        if vrad(a) < 0.0 < vrad(b):
            tc = float(brentq(vrad, a, b, xtol=1e-10, rtol=4 * np.finfo(float).eps))
        else:  # degenerate bracket (e.g. a flat minimum); fall back to direct minimisation
            tc = float(minimize_scalar(rng, bounds=(a, b), method="bounded", options={"xatol": 1e-8}).x)
        pp, vv = track.state(np.array([tc]))
        d = float(np.linalg.norm(pp[0] - ob))
        out.append(Pass(t_cpa=tc, d=d, v=float(np.linalg.norm(vv[0])), t_h=tc + d / c))
    return out
