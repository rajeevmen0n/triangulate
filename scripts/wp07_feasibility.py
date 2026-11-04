"""WP07 feasibility gate: does adsb.lol contain the aircraft behind the Oyster Bay demo?

Reproducible: `python scripts/wp07_feasibility.py [--json PATH]`. Uses the cached 2026-09-08 index
(builds it from the network if missing). Prints no registrations or owner names.

The video used Flightradar24 ids (not mappable to ICAO hex), so the aircraft are found by search:
brute-force CPA of every prepared track to the ground-truth point, then match each event's speed,
slant distance and heard time (plan 5.2 WP07, notes 2 10:33).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triangulate.acoustic.physics import C_DEFAULT, Track, solve_emission, track_passes  # noqa: E402
from triangulate.adsb.adsblol import AdsbLolProvider  # noqa: E402
from triangulate.adsb.aircraft_types import load_types  # noqa: E402
from triangulate.adsb.tracks import prepare_trackset  # noqa: E402
from triangulate.core.types import TrackQuery  # noqa: E402

GT = (40.876722, -73.526444)
H_OBS = 5.0
BBOX = (40.50, -74.00, 41.20, -73.00)
T0 = datetime(2026, 9, 8, 19, 45, tzinfo=UTC)
T1 = datetime(2026, 9, 8, 22, 0, tzinfo=UTC)
AUDIO_START = datetime(2026, 9, 8, 20, 15, 44, tzinfo=UTC)
EVENTS = {
    1: dict(tau=2152.0, v=(130.0, 154.0), d=(4300.0, 7800.0)),
    2: dict(tau=3073.0, v=(88.0, 110.0), d=(1390.0, 4690.0)),
}
GAP_VIDEO_S = 920.0
# plan 5.2 WP07 acceptance (looser): time +-5 min, d <= 9/6 km, v 120-165 / 80-120
PLAN = {1: dict(v=(120.0, 165.0), dmax=9000.0), 2: dict(v=(80.0, 120.0), dmax=6000.0)}
TIME_TOL_STRICT = 120.0
TIME_TOL_PLAN = 300.0
DAY = T0.date()
OUT_JSON = ROOT / "data/fixtures/feasibility_2026-09-08.json"


def heard_expected(ev: int) -> float:
    return AUDIO_START.timestamp() + EVENTS[ev]["tau"]


def observer_height_hae() -> tuple[float, str]:
    """Observer ellipsoidal height: DEM + geoid N + h_obs; falls back to geoid-only if no DEM."""
    from triangulate.dem.geoid import geoid_N
    try:
        from triangulate.dem.provider import elevation
        dem = float(np.asarray(elevation(GT[0], GT[1])).ravel()[0])
        if not np.isfinite(dem):
            raise RuntimeError("nan dem")
        src = "dem+geoid"
    except Exception as e:  # noqa: BLE001
        dem, src = 0.0, f"no-dem({type(e).__name__}); 0 m"
    n = float(geoid_N(GT[0], GT[1], warn=False))
    return dem + n + H_OBS, f"{src} (dem={dem:.1f} N={n:.1f})"


def callsigns(icao: str) -> list[str]:
    """Callsigns (detail.flight) seen in the cached raw member; TRACK_SCHEMA does not carry them."""
    import gzip
    f = AdsbLolProvider().member_path(DAY, icao)
    if not f.exists():
        return []
    doc = json.loads(gzip.decompress(f.read_bytes()))
    out = {(r[8] or {}).get("flight", "").strip() for r in doc.get("trace", []) if isinstance(r[8], dict)}
    return sorted(c for c in out if c)


def coverage(raw: pl.DataFrame, icao: str, t_cpa: float, half: float = 120.0) -> dict:
    d = raw.filter((pl.col("icao") == icao) & pl.col("t").is_between(t_cpa - half, t_cpa + half)).sort("t")
    full = raw.filter(pl.col("icao") == icao).sort("t")
    ts = d["t"].to_numpy()
    out: dict = {"n_raw_in_pm120s": int(len(ts))}
    if len(ts) >= 2:
        dt = np.diff(ts)
        out.update(median_dt_s=float(np.median(dt)), max_gap_s=float(dt.max()))
    before = full.filter(pl.col("t") <= t_cpa)["t"].to_numpy()
    after = full.filter(pl.col("t") >= t_cpa)["t"].to_numpy()
    out["nearest_sample_before_s"] = float(t_cpa - before.max()) if len(before) else None
    out["nearest_sample_after_s"] = float(after.min() - t_cpa) if len(after) else None
    if d.height:
        out["alt_geom_frac"] = float(d["alt_geom_ft"].is_not_null().mean())
        out["sources"] = sorted(set(d["source"].drop_nulls().to_list()))
    return out


def run(progress=None) -> dict:
    prov = AdsbLolProvider()
    ts = prov.fetch(TrackQuery(bbox=BBOX, t0=T0, t1=T1))
    types = load_types()
    prepared, cls = prepare_trackset(ts, GT, types, exclude_jets_=False)
    hobs, hsrc = observer_height_hae()
    obs = np.array([0.0, 0.0, hobs])
    kind = {r["icao"]: r for r in cls.iter_rows(named=True)}
    passes = []
    for (icao, seg), g in prepared.group_by(["icao", "seg"], maintain_order=True):
        g = g.sort("t")
        if g.height < 10:
            continue
        pos = g.select("e", "n", "u").to_numpy()
        tr = Track.from_positions(g["t"].to_numpy(), pos, label=icao)
        for p in track_passes(tr, obs, r_max=20000.0, step=1.0):
            if not (T0.timestamp() <= p.t_h <= T1.timestamp()):
                continue
            # cross-check with the retarded-time solver: emission time of the sound heard at t_h
            em = solve_emission(np.array([p.t_h]), tr, obs, C_DEFAULT)
            te = float(np.asarray(em.t_e).ravel()[0])
            a = kind[icao]
            passes.append(dict(
                icao=icao, seg=int(seg), typecode=a["typecode"], desc_code=a["desc_code"],
                keep=bool(a["keep"]), reason=a["reason"], d_m=p.d, v_ms=p.v, t_h=p.t_h,
                t_h_utc=datetime.fromtimestamp(p.t_h, UTC).strftime("%H:%M:%S"),
                t_cpa=p.t_cpa, solve_emission_err_s=te - p.t_cpa,
                alt_src=g["alt_src"][g.height // 2], seg_span=(float(g["t"][0]), float(g["t"][-1])),
            ))
    raw = ts.tracks
    res = dict(ground_truth=GT, observer_hae_m=hobs, observer_height_source=hsrc,
               window=[T0.isoformat(), T1.isoformat()], n_aircraft=ts.aircraft.height,
               n_tracks_rows=raw.height, n_jets_excluded=int((~cls["keep"]).sum()),
               n_kept=int(cls["keep"].sum()), n_passes_within_20km=len(passes))
    matches: dict = {}
    for ev, spec in EVENTS.items():
        th = heard_expected(ev)
        strict, plan, near = [], [], []
        for p in passes:
            dt = p["t_h"] - th
            if abs(dt) <= TIME_TOL_PLAN:
                near.append(p)
            vs_ok = spec["v"][0] <= p["v_ms"] <= spec["v"][1]
            d_ok = spec["d"][0] <= p["d_m"] <= spec["d"][1]
            if p["keep"] and abs(dt) <= TIME_TOL_STRICT and vs_ok and d_ok:
                strict.append(p)
            pl_ = PLAN[ev]
            if (p["keep"] and abs(dt) <= TIME_TOL_PLAN and pl_["v"][0] <= p["v_ms"] <= pl_["v"][1]
                    and p["d_m"] <= pl_["dmax"]):
                plan.append(p)
        for lst in (strict, plan, near):
            for p in lst:
                p["dt_vs_expected_s"] = p["t_h"] - th
                p["coverage"] = coverage(raw, p["icao"], p["t_cpa"])
                p["callsigns"] = callsigns(p["icao"])
        matches[ev] = dict(expected_heard_utc=datetime.fromtimestamp(th, UTC).strftime("%H:%M:%S"),
                           strict=strict, plan_criteria=plan, all_in_time_window_pm5min=near)
    res["events"] = matches
    pairs = []
    for a in matches[1]["strict"]:
        for b in matches[2]["strict"]:
            if a["icao"] != b["icao"]:
                pairs.append(dict(ev1=a["icao"], ev2=b["icao"], gap_s=b["t_h"] - a["t_h"]))
    res["strict_pairs"] = pairs
    res["pairs_within_10s_of_921"] = [p for p in pairs if abs(p["gap_s"] - 921.0) <= 10.0]
    res["go"] = bool(matches[1]["plan_criteria"] and matches[2]["plan_criteria"]
                     and any(abs(b["t_h"] - a["t_h"] - 921.0) <= 10.0
                             for a in matches[1]["plan_criteria"] for b in matches[2]["plan_criteria"]
                             if a["icao"] != b["icao"]))
    res["go_strict"] = bool(res["pairs_within_10s_of_921"])
    return res


def _fmt(p: dict) -> str:
    c = p["coverage"]
    cs = ",".join(p.get("callsigns", []))
    return (f"{p['icao']} {cs} {p['typecode']}({p['desc_code']}) d={p['d_m']:.0f}m v={p['v_ms']:.1f}m/s "
            f"heard={p['t_h_utc']} dt={p['dt_vs_expected_s']:+.0f}s alt={p['alt_src']} "
            f"samp={c.get('median_dt_s')}/{c.get('max_gap_s')}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    a = ap.parse_args()
    res = run()
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1, default=str))
    print(f"observer HAE {res['observer_hae_m']:.1f} m [{res['observer_height_source']}]; "
          f"{res['n_aircraft']} aircraft, {res['n_jets_excluded']} jets excluded, "
          f"{res['n_passes_within_20km']} passes <20km")
    for ev, m in res["events"].items():
        print(f"\nEvent {ev} (expected heard {m['expected_heard_utc']}):")
        for k in ("strict", "plan_criteria"):
            print(f"  {k}: {len(m[k])}")
            for p in m[k]:
                print("    " + _fmt(p))
        print(f"  all passes within +-5min (any type): {len(m['all_in_time_window_pm5min'])}")
        for p in m["all_in_time_window_pm5min"]:
            print(f"    keep={p['keep']} " + _fmt(p))
    print("\nstrict pairs:", res["strict_pairs"])
    print("GO (plan criteria):", res["go"], "| GO (strict video ranges):", res["go_strict"])
    return 0 if res["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
