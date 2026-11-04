"""WP22: synthetic end-to-end calibration of lambda and the acceptance table (plan 5.2, 3.3.7).

    nix develop --command python scripts/bench_synthetic_e2e.py [--seeds 40] [--mp3-seeds 40]
        [--single-seeds 5] [--workers 4] [--out data/calibration]

Runs `triangulate.acoustic.synthetic_e2e.run_calibration`: 40 deterministic synthetic worlds
(30 GA tracks over 60 x 60 km, fake DEM, 2-3 heard flights, known date / unknown time, a silence
decoy) through world -> in-memory provider -> prepare -> synth audio -> find_events ->
acoustic_search -> run_fusion; fits lambda for 90 % HPD coverage; then the MP3-32k and the
single-event known-time variants at that lambda. Writes <out>/lambda.json (summary, gitignored),
<out>/runs.json (per-seed records) and the cached hypotheses (<out>/hyps/), and prints the table.
Commit the fitted numbers in `src/triangulate/acoustic/calibration_defaults.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _log(r: dict) -> None:
    b = r.get("best") or {}
    tag = "mp3" if r.get("mp3_kbps") else ("single" if r.get("single_event") else "clean")
    print(f"[{time.strftime('%H:%M:%S')}] {tag:6s} seed {r['seed']:3d} ok={r.get('ok')} "
          f"best={b.get('dist_m', float('nan')):.0f} m tuple_ok={b.get('correct_tuple')} "
          f"decoy={r.get('decoy', {}).get('removed')} {r.get('error', '')}", flush=True)


def table(out: dict) -> list[tuple[str, str, str, bool]]:
    c, m, s = out["clean"], out.get("mp3_32k"), out.get("single_event")
    rows = [
        ("top-1 within 100 m (clean)", ">= 0.90", f"{c['top1_within_100m']:.3f}",
         c["top1_within_100m"] >= 0.9),
        ("silence decoy removed", "1.00", f"{c['decoy_removed']:.3f} of {c['decoy_applicable']}",
         c["decoy_removed"] == 1.0),
        ("90 % HPD coverage", "[0.80, 0.97]", f"{c['hpd90_coverage']:.3f}",
         0.8 <= c["hpd90_coverage"] <= 0.97),
    ]
    if m:
        rows.append(("MP3 32k top-3 within 250 m", ">= 0.80", f"{m['top3_within_250m']:.3f}",
                     m["top3_within_250m"] >= 0.8))
    if s and s["band"]:
        n = len(s["band"])
        k = sum(b["contains_truth"] for b in s["band"])
        rows.append(("single-event band contains truth", "all", f"{k}/{n}", k == n))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--mp3-seeds", type=int, default=40)
    ap.add_argument("--single-seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "data" / "calibration"))
    a = ap.parse_args()
    from triangulate.acoustic.synthetic_e2e import run_calibration

    out = run_calibration(a.seeds, mp3_seeds=a.mp3_seeds, single_seeds=a.single_seeds, workers=a.workers,
                          out_dir=a.out, log=_log)
    print(f"\nlambda* = {out['lam']} (coverage {out['coverage_at_lam']:.3f}); "
          f"wall {out['wall_s'] / 60:.1f} min")
    print("coverage curve:", ", ".join(f"{lam:g}:{c:.2f}" for lam, c in out["coverage_curve"]))
    rows = table(out)
    for name, target, got, ok in rows:
        print(f"  {name:36s} target {target:14s} measured {got:16s} {'PASS' if ok else 'FAIL'}")
    tm = out["clean"]["timing_median_s"]
    print("median per-seed timing (s):", {k: round(v, 1) for k, v in tm.items() if "." not in k})
    print(f"wrote {Path(a.out) / 'lambda.json'}")
    return 0 if all(r[3] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
