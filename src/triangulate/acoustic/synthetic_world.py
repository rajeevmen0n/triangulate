"""Fully synthetic acoustic world for the WP22 end-to-end, calibration and performance runs.

One seed gives one deterministic world (plan 5.2 WP22):

* a 60 x 60 km search region with an analytic fake DEM (`AnalyticDem`, orthometric) and a constant
  geoid undulation, so ground HAE = DEM + N everywhere (`World.ground_hae`);
* 30 GA-like flights (`n_flights`): piecewise straight / gently turning, climbing or level paths at
  45-85 m/s, plus a few jets that the jet filter must drop. They are sampled like ADS-B (irregular
  4-7 s, 25 ft altitude quantisation, a few metres of position noise) into a `TrackSet` and served
  by `InMemoryTrackProvider` (the `core.types.TrackProvider` protocol);
* an observer X heard by 2-3 of the flights ("heard"), at a known date but unknown time of day
  (T_s uniform over 03:00-20:00 UTC; the search window is the whole UTC day);
* a *silence decoy*: copies of the heard flights translated to a far point X_d (so a hypothesis at
  X_d scores exactly like the truth) plus one extra flight per gap between heard events that passes
  X_d closely while X_d should be silent. Only the silence filter can remove the decoy;
* every other flight (background, jets, decoy copies, silence flights) stays >= `quiet_m` from X
  during the recording, so it would be inaudible there and is not rendered.

Audio: the heard flights rendered with exact retarded Doppler (`synth.render_source`, on the exact
1-s track, not on the ADS-B samples), each scaled to a prescribed in-band SNR (20-1000 Hz over
t_h +- 60 s against the pink noise + clicks background), plus hum lines and clicks, optionally an MP3
round trip (`mp3_kbps`). Every random draw comes from ``numpy.random.default_rng(seed)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl

from ..core.geo import EARTH_R, enu_to_geodetic, geodetic_to_enu
from ..core.types import AIRCRAFT_SCHEMA, TRACK_SCHEMA, TrackQuery, TrackSet
from .physics import C_DEFAULT, Pass, Track, track_passes
from .synth import (
    FS_DEFAULT,
    SynthSource,
    band_power,
    hum_lines,
    mp3_roundtrip,
    pink_noise,
    render_source,
    transients,
)

__all__ = [
    "WorldConfig", "AnalyticDem", "InMemoryTrackProvider", "Flight", "World", "build_world",
    "ga_path", "trackset_from_flights", "SYNTH_LICENCE",
]

SYNTH_LICENCE = "synthetic tracks (WP22 test world, no licence)"
FT_M = 0.3048
KT_MS = 1852.0 / 3600.0
PROP_TYPES = ("C172", "PA28", "P28A", "SR22", "C182", "BE36", "DA40", "M20P", "C208")
JET_TYPES = ("B738",)


@dataclass(frozen=True)
class WorldConfig:
    center: tuple[float, float] = (39.80, -89.65)  # any place works: DEM and geoid are synthetic
    size_km: float = 60.0
    date: date = date(2026, 9, 8)
    n_flights: int = 30  # total, including heard, decoy copies, silence flights and jets
    n_jets: int = 2
    n_heard: tuple[int, ...] = (2, 2, 3)  # drawn uniformly per seed
    duration_s: float = 900.0
    ts_hours: tuple[float, float] = (3.0, 20.0)  # T_s (UTC hour of day) range
    h_obs: float = 5.0
    geoid_n: float = -30.0
    quiet_m: float = 7000.0  # unheard flights stay this far (3-D) from X during the recording
    heard_d: tuple[float, float] = (900.0, 3500.0)  # slant CPA distance of heard flights
    heard_h: tuple[float, float] = (300.0, 1400.0)  # CPA height above the observer
    heard_v: tuple[float, float] = (45.0, 85.0)
    heard_turn_dps: float = 0.4  # max turn rate of heard flights (deg/s): gentle, near-straight passes
    heard_p_turn: float = 0.35
    heard_p_climb: float = 0.15
    heard_snr_db: tuple[float, float] = (-1.0, 6.0)
    third_snr_db: tuple[float, float] = (-4.0, 2.0)  # the weaker extra heard flight
    min_gap_s: float = 260.0  # between heard events (> 2 x silence guard)
    edge_s: float = 160.0  # heard events stay this far from the recording ends
    silence_decoy: bool = True
    decoy_min_km: float = 15.0
    decoy_span_s: float = 300.0  # decoy copies exist for +- this around their pass at X_d
    silence_d_frac: tuple[float, float] = (0.3, 0.6)  # silence-flight CPA at X_d / min heard d
    noise_rms: float = 0.05
    hum_hz: tuple[float, ...] = (40.0, 60.0, 125.0, 180.0, 250.0, 360.0)
    hum_amp: float = 0.03
    click_rate_per_min: float = 4.0
    click_amp: float = 0.5
    mp3_kbps: int | None = None
    adsb_dt: tuple[float, float] = (4.0, 7.0)  # ADS-B sample spacing range (s)
    adsb_pos_sigma_m: float = 3.0
    c: float = C_DEFAULT

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        lat0, lon0 = self.center
        dlat = math.degrees(self.size_km * 500.0 / EARTH_R)
        dlon = dlat / math.cos(math.radians(lat0))
        return (lat0 - dlat, lon0 - dlon, lat0 + dlat, lon0 + dlon)


class AnalyticDem:
    """Smooth fake terrain (orthometric metres): a tilted plane plus two sinusoid ridges and a bump.

    Implements `core.types.DemProvider`. Relief is ~150 m over the region, so the observer height
    matters (as with a real DEM) but nothing needs a download."""

    def __init__(self, center: tuple[float, float], base_m: float = 220.0, geoid_n: float = -30.0):
        self.center = (float(center[0]), float(center[1]))
        self.base_m = float(base_m)
        self.geoid_n = float(geoid_n)

    def _en(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        n = np.radians(lat - self.center[0]) * EARTH_R
        e = np.radians(lon - self.center[1]) * EARTH_R * math.cos(math.radians(self.center[0]))
        return e, n

    def elevation(self, lat, lon) -> np.ndarray:
        e, n = self._en(lat, lon)
        return (self.base_m + 0.0012 * e - 0.0008 * n
                + 35.0 * np.sin(2 * np.pi * e / 17_000.0) * np.cos(2 * np.pi * n / 23_000.0)
                + 18.0 * np.sin(2 * np.pi * (e + n) / 9_000.0 + 0.7)
                + 60.0 * np.exp(-((e - 8000.0) ** 2 + (n + 5000.0) ** 2) / (2 * 6000.0**2)))

    def ground_hae(self, lat, lon) -> np.ndarray:
        """Ellipsoidal ground height (DEM + geoid N): the `acoustic.grid.GroundFn` contract."""
        return self.elevation(lat, lon) + self.geoid_n


class InMemoryTrackProvider:
    """`core.types.TrackProvider` over an in-memory `TrackSet` (bbox / time filtered on fetch)."""

    def __init__(self, ts: TrackSet, name: str = "synthetic", licence: str = SYNTH_LICENCE):
        self.ts = ts
        self.name = name
        self.licence = licence
        self.calls: list[TrackQuery] = []

    def days_available(self, t0: datetime, t1: datetime) -> list[date]:
        if self.ts.tracks.height == 0:
            return []
        tmin, tmax = (float(x) for x in (self.ts.tracks["t"].min(), self.ts.tracks["t"].max()))
        lo, hi = max(t0.timestamp(), tmin), min(t1.timestamp(), tmax)
        if lo > hi:
            return []
        d0 = datetime.fromtimestamp(lo, UTC).date()
        d1 = datetime.fromtimestamp(hi, UTC).date()
        return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    def fetch(self, q: TrackQuery, progress=None) -> TrackSet:
        self.calls.append(q)
        s, w, n, e = q.bbox
        t0, t1 = q.t0.timestamp(), q.t1.timestamp()
        tr = self.ts.tracks.filter(
            pl.col("lat").is_between(s, n) & pl.col("lon").is_between(w, e) & pl.col("t").is_between(t0, t1)
        )
        if not q.include_ground:
            tr = tr.filter(~pl.col("on_ground"))
        ac = self.ts.aircraft.filter(pl.col("icao").is_in(tr["icao"].unique().implode()))
        if progress is not None:
            progress(1.0, f"{ac.height} aircraft (in memory)")
        return TrackSet(tracks=tr, aircraft=ac, provider=self.name, licence=self.licence)


# --- flight paths ----------------------------------------------------------------------------------


def ga_path(rng: np.random.Generator, duration: float, v0: float, z0: float, *, heading0: float | None = None,
            max_turn_dps: float = 1.8, p_turn: float = 0.45, p_climb: float = 0.3,
            z_lim: tuple[float, float] = (500.0, 3200.0), dt: float = 1.0, sub: int = 10,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative GA path starting at the origin at t = 0: (t, pos (n,3), vel (n,3)), exact node
    velocities. Piecewise segments (40-300 s) of straight flight or a constant-rate turn, level or a
    +-3 m/s climb, with slow +-4 % speed wander. Positions integrate the velocity on a 0.1-s grid."""
    n_fine = int(math.ceil(duration / dt)) * sub + 1
    h = dt / sub
    tf = np.arange(n_fine) * h
    om = np.zeros(n_fine)
    cl = np.zeros(n_fine)
    t = 0.0
    while t < duration:
        L = float(rng.uniform(40.0, 300.0))
        m = (tf >= t) & (tf < t + L)
        if rng.random() < p_turn:
            om[m] = math.radians(rng.uniform(0.25 * max_turn_dps, max_turn_dps)) * rng.choice([-1.0, 1.0])
        if rng.random() < p_climb:
            cl[m] = rng.uniform(-3.0, 3.0)
        t += L
    psi0 = math.radians(rng.uniform(0, 360)) if heading0 is None else math.radians(heading0)
    psi = psi0 + np.concatenate([[0.0], np.cumsum(0.5 * (om[1:] + om[:-1]) * h)])
    amp, per, ph = rng.uniform(0.0, 0.04), rng.uniform(200.0, 600.0), rng.uniform(0, 2 * np.pi)
    v = v0 * (1.0 + amp * np.sin(2 * np.pi * tf / per + ph))
    z = z0 + np.concatenate([[0.0], np.cumsum(0.5 * (cl[1:] + cl[:-1]) * h)])
    zc = np.clip(z, *z_lim)
    vz = np.where(zc != z, 0.0, cl)
    vel = np.column_stack([v * np.sin(psi), v * np.cos(psi), vz])
    pos = np.zeros((n_fine, 3))
    pos[1:, :2] = np.cumsum(0.5 * (vel[1:, :2] + vel[:-1, :2]) * h, axis=0)
    pos[:, 2] = zc
    return tf[::sub].copy(), pos[::sub].copy(), vel[::sub].copy()


@dataclass(eq=False)
class Flight:
    icao: str
    typecode: str
    role: str  # heard | background | jet | decoy | silence
    track: Track  # exact 1-s path, ENU about the world origin, z = HAE, absolute epoch times
    f0: float = 0.0
    snr_db: float | None = None
    harmonics: tuple[float, ...] = (1.0, 0.6, 0.4)
    copy_of: str | None = None  # decoy: icao of the heard flight it copies
    pass_x: Pass | None = None  # heard: the CPA at X that defines the event


@dataclass(eq=False)
class World:
    seed: int
    cfg: WorldConfig
    origin: tuple[float, float]  # lat, lon of the ENU frame (= the bbox centre)
    bbox: tuple[float, float, float, float]
    dem: AnalyticDem
    X: np.ndarray  # observer ENU (ground HAE + h_obs)
    truth_latlon: tuple[float, float]
    t_s: float  # true UTC epoch of audio sample 0
    flights: list[Flight]
    trackset: TrackSet
    audio: np.ndarray
    fs: int = FS_DEFAULT
    decoy_latlon: tuple[float, float] | None = None
    decoy_X: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    @property
    def provider(self) -> InMemoryTrackProvider:
        return InMemoryTrackProvider(self.trackset)

    @property
    def ground_hae(self):
        return self.dem.ground_hae

    @property
    def day_window(self) -> tuple[float, float]:
        """T_s window for "known date, unknown time": the whole UTC day (minus the recording)."""
        d0 = datetime(self.cfg.date.year, self.cfg.date.month, self.cfg.date.day, tzinfo=UTC).timestamp()
        return d0, d0 + 86400.0 - self.cfg.duration_s

    def heard(self) -> list[Flight]:
        return sorted((f for f in self.flights if f.role == "heard"), key=lambda f: f.pass_x.t_h)

    def heard_taus(self) -> list[float]:
        return [f.pass_x.t_h - self.t_s for f in self.heard()]

    def by_icao(self, icao: str) -> Flight:
        return next(f for f in self.flights if f.icao == icao)


def _icao(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        s = f"{int(rng.integers(0xA00000, 0xADFFFF)):06x}"
        if s not in used:
            used.add(s)
            return s


def _min_dist(tr: Track, X: np.ndarray, t_lo: float, t_hi: float) -> float:
    m = (tr.t >= t_lo) & (tr.t <= t_hi)
    if not m.any():
        return math.inf
    return float(np.linalg.norm(tr.pos[m] - X, axis=1).min())


def _local_enu(dem: AnalyticDem, origin, e: float, n: float, h_above: float) -> np.ndarray:
    lat, lon, _ = enu_to_geodetic(np.array([e]), np.array([n]), np.array([0.0]), *origin)
    g = float(dem.ground_hae(lat, lon)[0])
    ee, nn, uu = geodetic_to_enu(lat, lon, np.array([g + h_above]), *origin)
    return np.array([float(ee[0]), float(nn[0]), float(uu[0])])


def _heard_track(rng, cfg: WorldConfig, X: np.ndarray, t_h: float) -> tuple[Track, Pass]:
    """A GA path whose closest pass to X is heard at t_h, with slant d and height drawn from cfg."""
    for _ in range(200):
        d = float(np.exp(rng.uniform(*np.log(cfg.heard_d))))
        hh = float(rng.uniform(cfg.heard_h[0], min(cfg.heard_h[1], 0.95 * d)))
        v0 = float(rng.uniform(*cfg.heard_v))
        half = float(rng.uniform(450.0, 750.0))
        t, pos, vel = ga_path(rng, 2 * half, v0, 1000.0, max_turn_dps=cfg.heard_turn_dps,
                            p_turn=cfg.heard_p_turn, p_climb=cfg.heard_p_climb)
        i = int(round(half))
        vh = vel[i, :2] / np.linalg.norm(vel[i, :2])
        right = np.array([vh[1], -vh[0]])
        side = rng.choice([-1.0, 1.0])
        p_c = X + np.array([*(side * math.sqrt(d * d - hh * hh) * right), hh])
        pos = pos - pos[i] + p_c
        tr = Track(t=t, pos=pos, vel=vel)
        ps = track_passes(tr, X, cfg.c)
        if not ps:
            continue
        p = min(ps, key=lambda q: q.d)
        if not (cfg.heard_d[0] * 0.8 <= p.d <= cfg.heard_d[1] * 1.2) or abs(p.t_cpa - half) > 120:
            continue
        sh = t_h - p.t_h
        tr = Track(t=tr.t + sh, pos=tr.pos, vel=tr.vel)
        p = Pass(t_cpa=p.t_cpa + sh, d=p.d, v=p.v, t_h=p.t_h + sh)
        return tr, p
    raise RuntimeError("could not place a heard track")


def _random_track(rng, cfg: WorldConfig, half_m: float, t_lo: float, t_hi: float, jet: bool = False) -> Track:
    dur = float(rng.uniform(900.0, 2700.0))
    if jet:
        t, pos, vel = ga_path(rng, dur, float(rng.uniform(150, 230)), float(rng.uniform(4000, 9000)),
                              max_turn_dps=0.5, p_climb=0.2, z_lim=(3000.0, 11000.0))
    else:
        t, pos, vel = ga_path(rng, dur, float(rng.uniform(42, 85)), float(rng.uniform(700, 2800)))
    start = np.array([rng.uniform(-half_m, half_m), rng.uniform(-half_m, half_m), 0.0])
    pos = pos + start
    t0 = float(rng.uniform(t_lo, t_hi))
    return Track(t=t + t0, pos=pos, vel=vel)


def _translate(tr: Track, shift: np.ndarray, t_lo: float = -math.inf, t_hi: float = math.inf) -> Track:
    """The track moved by `shift` (same times), optionally cut to [t_lo, t_hi]."""
    m = (tr.t >= t_lo) & (tr.t <= t_hi)
    return Track(t=tr.t[m].copy(), pos=tr.pos[m] + shift, vel=tr.vel[m].copy())


def _silence_track(rng, cfg: WorldConfig, Xd: np.ndarray, t_h: float, d: float) -> Track:
    hh = min(0.8 * d, 600.0)
    t, pos, vel = ga_path(rng, 1200.0, float(rng.uniform(45, 75)), 1500.0, p_turn=0.2, p_climb=0.0)
    i = 600
    vh = vel[i, :2] / np.linalg.norm(vel[i, :2])
    p_c = Xd + np.array([*(math.sqrt(d * d - hh * hh) * np.array([vh[1], -vh[0]])), hh])
    tr = Track(t=t, pos=pos - pos[i] + p_c, vel=vel)
    p = min(track_passes(tr, Xd, cfg.c), key=lambda q: q.d)
    return Track(t=tr.t + (t_h - p.t_h), pos=tr.pos, vel=tr.vel)


def trackset_from_flights(flights: Sequence[Flight], origin, rng: np.random.Generator, cfg: WorldConfig,
                          ) -> TrackSet:
    """ADS-B-like sampling of the exact tracks: irregular spacing, position noise, 25-ft altitudes."""
    rows = []
    ac = []
    for f in flights:
        tr = f.track
        ts = [tr.t_start + float(rng.uniform(0, cfg.adsb_dt[1]))]
        while ts[-1] < tr.t_end:
            ts.append(ts[-1] + float(rng.uniform(*cfg.adsb_dt)))
        ts = np.array([x for x in ts if x <= tr.t_end])
        p, v = tr.state(ts)
        p = p + np.column_stack([rng.normal(0, cfg.adsb_pos_sigma_m, (ts.size, 2)), np.zeros(ts.size)])
        lat, lon, hae = enu_to_geodetic(p[:, 0], p[:, 1], p[:, 2], *origin)
        gs = np.hypot(v[:, 0], v[:, 1])
        geom_ft = np.round(hae / FT_M / 25.0) * 25.0
        baro_ft = np.round((hae - cfg.geoid_n - 20.0) / FT_M / 25.0) * 25.0  # "pressure" altitude
        rows.append(pl.DataFrame({
            "icao": f.icao, "reg": "", "typecode": f.typecode, "category": "A3" if f.role == "jet" else "A1",
            "t": ts, "lat": lat, "lon": lon, "alt_baro_ft": baro_ft, "alt_geom_ft": geom_ft,
            "gs_kt": np.round(gs / KT_MS, 1), "track_deg": np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360,
            "vrate_fpm": np.round(v[:, 2] / FT_M * 60.0 / 64.0) * 64.0, "on_ground": False,
            "source": "adsb_icao",
        }, schema=TRACK_SCHEMA))
        ac.append({"icao": f.icao, "reg": "", "typecode": f.typecode, "desc": "", "category":
                   "A3" if f.role == "jet" else "A1", "db_flags": 0, "owner_op": "", "year": ""})
    tracks = pl.concat(rows).sort("icao", "t")
    return TrackSet(tracks=tracks, aircraft=pl.DataFrame(ac, schema=AIRCRAFT_SCHEMA),
                    provider="synthetic", licence=SYNTH_LICENCE)


def _event_times(rng, cfg: WorldConfig, k: int) -> list[float]:
    """k heard times in [edge, duration - edge], at least min_gap apart: the spare time is split
    over the k + 1 intervals with a flat Dirichlet draw (always feasible when the scene fits)."""
    slack = cfg.duration_s - 2 * cfg.edge_s - (k - 1) * cfg.min_gap_s
    if slack < 0:
        raise RuntimeError("cannot space the heard events; lengthen duration_s")
    sp = slack * rng.dirichlet(np.ones(k + 1))
    return [float(cfg.edge_s + sp[: i + 1].sum() + i * cfg.min_gap_s) for i in range(k)]


def _render(world_flights: Sequence[Flight], X: np.ndarray, cfg: WorldConfig, t_s: float,
            rng: np.random.Generator) -> np.ndarray:
    fs = FS_DEFAULT
    n = int(round(cfg.duration_s * fs))
    t_r = t_s + np.arange(n) / fs
    bg = pink_noise(n, fs, cfg.noise_rms, rng) + transients(n, fs, cfg.click_rate_per_min, cfg.click_amp, rng)
    sig = np.zeros(n)
    for f in world_flights:
        src = SynthSource(f.track, f0=f.f0, harmonics=f.harmonics,
                          phases=tuple(rng.uniform(0, 2 * np.pi, len(f.harmonics))), label=f.icao)
        x = render_source(src, X, t_r, cfg.c)
        tau = f.pass_x.t_h - t_s
        a, b = int(max(tau - 60, 0) * fs), int(min(tau + 60, cfg.duration_s) * fs)
        ps, pb = band_power(x[a:b], fs, 20, 1000), band_power(bg[a:b], fs, 20, 1000)
        sig += x * math.sqrt(pb / ps * 10 ** (f.snr_db / 10))
    if cfg.hum_hz:
        sig += hum_lines(n, fs, [(f, cfg.hum_amp) for f in cfg.hum_hz], rng, am_depth=0.2)
    mix = (sig + bg).astype(np.float32)
    if cfg.mp3_kbps:
        mix = mp3_roundtrip(mix, fs, cfg.mp3_kbps)
    return mix


def build_world(seed: int, cfg: WorldConfig | None = None) -> World:
    """Deterministic synthetic world for one seed (see the module docstring)."""
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng([22, seed])
    bbox = cfg.bbox
    origin = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    dem = AnalyticDem(origin, geoid_n=cfg.geoid_n)
    half = cfg.size_km * 500.0
    d0 = datetime(cfg.date.year, cfg.date.month, cfg.date.day, tzinfo=UTC).timestamp()
    t_s = d0 + float(rng.uniform(*cfg.ts_hours)) * 3600.0
    t_s = round(t_s, 3)
    xe, xn = rng.uniform(-0.45 * half, 0.45 * half, 2)
    X = _local_enu(dem, origin, float(xe), float(xn), cfg.h_obs)
    lat, lon, _ = enu_to_geodetic(np.array([X[0]]), np.array([X[1]]), np.array([X[2]]), *origin)
    truth = (float(lat[0]), float(lon[0]))
    used: set[str] = set()
    k = int(rng.choice(cfg.n_heard))
    taus = _event_times(rng, cfg, k)
    order = rng.permutation(k)  # which events are the two strong ones
    flights: list[Flight] = []
    for j, tau in enumerate(taus):
        tr, p = _heard_track(rng, cfg, X, t_s + tau)
        strong = j in order[:2]
        snr = float(rng.uniform(*(cfg.heard_snr_db if strong else cfg.third_snr_db)))
        harm = (1.0, float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.0, 0.5)))
        flights.append(Flight(_icao(rng, used), str(rng.choice(PROP_TYPES)), "heard", tr,
                              f0=float(rng.uniform(70.0, 200.0)), snr_db=snr, harmonics=harm, pass_x=p))
    rec = (t_s - 300.0, t_s + cfg.duration_s + 120.0)

    def quiet(tr: Track) -> bool:
        return _min_dist(tr, X, *rec) >= cfg.quiet_m

    decoy_X = decoy_ll = None
    if cfg.silence_decoy:
        for _ in range(2000):
            de, dn = rng.uniform(-0.45 * half, 0.45 * half, 2)
            if math.hypot(de - X[0], dn - X[1]) < cfg.decoy_min_km * 1000:
                continue
            Xd = _local_enu(dem, origin, float(de), float(dn), cfg.h_obs)
            shift = Xd - X
            # copies are cut to +-decoy_span_s around their pass (an ADS-B coverage window)
            copies = [_translate(f.track, shift, f.pass_x.t_cpa - cfg.decoy_span_s,
                                 f.pass_x.t_cpa + cfg.decoy_span_s) for f in flights]
            if all(quiet(c) for c in copies):
                break
        else:
            raise RuntimeError("no decoy site")
        decoy_X = Xd
        la, lo, _ = enu_to_geodetic(np.array([Xd[0]]), np.array([Xd[1]]), np.array([Xd[2]]), *origin)
        decoy_ll = (float(la[0]), float(lo[0]))
        for f, c in zip(list(flights), copies, strict=True):
            flights.append(Flight(_icao(rng, used), f.typecode, "decoy", c, f0=f.f0, copy_of=f.icao))
        dmin = min(f.pass_x.d for f in flights if f.role == "heard")
        for a, b in zip(taus[:-1], taus[1:], strict=True):
            for _ in range(200):
                d_s = float(rng.uniform(*cfg.silence_d_frac)) * dmin
                sl = _silence_track(rng, cfg, Xd, t_s + 0.5 * (a + b), d_s)
                if quiet(sl):
                    break
            else:
                raise RuntimeError("no quiet silence flight")
            flights.append(Flight(_icao(rng, used), str(rng.choice(PROP_TYPES)), "silence", sl))
    n_bg = cfg.n_flights - len(flights)
    n_jet = min(cfg.n_jets, max(n_bg, 0))
    while len(flights) < cfg.n_flights:
        jet = sum(f.role == "jet" for f in flights) < n_jet
        tr = _random_track(rng, cfg, 0.9 * half, t_s - 2400.0, t_s + cfg.duration_s + 600.0, jet=jet)
        if quiet(tr):
            flights.append(Flight(_icao(rng, used), str(rng.choice(JET_TYPES if jet else PROP_TYPES)),
                                  "jet" if jet else "background", tr))
    ts = trackset_from_flights(flights, origin, rng, cfg)
    audio = _render([f for f in flights if f.role == "heard"], X, cfg, t_s, rng)
    return World(seed=seed, cfg=cfg, origin=origin, bbox=bbox, dem=dem, X=X, truth_latlon=truth, t_s=t_s,
                 flights=flights, trackset=ts, audio=audio, decoy_latlon=decoy_ll, decoy_X=decoy_X,
                 notes={"taus": taus, "strong": [int(i) for i in order[:2]]})
