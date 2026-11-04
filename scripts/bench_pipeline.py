"""WP22 performance benchmark of the acoustic stages against plan 3.3.8 (no network).

    nix develop --command python scripts/bench_pipeline.py [--skip-decode] [--skip-detect] [--out PATH]

Stages and targets (4 cores, no GPU):
- decode + STFT + whitening of 1 h of audio <= 15 s (a 1-h 44.1 kHz MP3 written to data/tmp/bench/
  from a synthetic scene, decoded by `ingest.media.decode_audio` -> `compute_spectrogram`);
- detection (432 templates) + fitting on 1 h <= 3 min (`detect.find_events`);
- CPA stage, region mode (100 x 100 km, 3 h, 500 flights) <= 60 s (`scripts/bench_cpa.py` scene);
- matching + silence <= 20 s (DuckDB offset match + silence filter on that CPA table);
- scoring 50 k hypotheses x 2 events <= 60 s (`scripts/bench_score.py`).
The end-to-end per-seed timings of the synthetic world come from `scripts/bench_synthetic_e2e.py`.
Results are printed and written as JSON (default data/calibration/bench_pipeline.json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TARGETS = {"decode_stft_whiten_1h": 15.0, "detect_fit_1h": 180.0, "cpa_region": 60.0,
           "match_silence": 20.0, "score_50k_x2": 60.0}


def one_hour_audio(seed: int = 0) -> np.ndarray:
    """1 h synthetic scene at 4 kHz: 4 straight GA passes + pink noise, hum lines and clicks."""
    from triangulate.acoustic.synth import DEMO_HUM_HZ, SynthSource, straight_pass_track, synthesize

    rng = np.random.default_rng(seed)
    obs = np.zeros(3)
    srcs = []
    for k, t_h in enumerate((500.0, 1400.0, 2152.0, 3073.0)):
        tr = straight_pass_track(obs, t_h, float(rng.uniform(50, 140)), float(rng.uniform(1000, 6000)),
                                 heading_deg=float(rng.uniform(0, 360)), half_span_s=400.0)
        srcs.append(SynthSource(tr, f0=float(rng.uniform(80, 180)), label=f"s{k}"))
    syn = synthesize(3600.0, srcs, obs, noise_rms=0.3, hum=[(f, 0.03) for f in DEMO_HUM_HZ],
                     click_rate_per_min=4.0, click_amp=0.4, seed=seed)
    return syn.audio


def bench_decode(audio: np.ndarray, work: Path) -> dict:
    from triangulate.acoustic.spectrogram import compute_spectrogram
    from triangulate.ingest.media import decode_audio

    work.mkdir(parents=True, exist_ok=True)
    raw, mp3 = work / "bench_1h.f32", work / "bench_1h.mp3"
    audio.astype("<f4").tofile(raw)
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar",
                    "4000", "-ac", "1", "-i", str(raw), "-af", "aresample=44100:resampler=soxr",
                    "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)], check=True)
    raw.unlink()
    t = time.perf_counter()
    x = decode_audio(mp3)
    t_dec = time.perf_counter() - t
    t = time.perf_counter()
    spec = compute_spectrogram(x)
    t_spec = time.perf_counter() - t
    mp3.unlink()
    return {"decode_s": t_dec, "stft_whiten_s": t_spec, "wall_s": t_dec + t_spec,
            "frames": int(spec.Z.shape[0]), "spec": spec}


def bench_detect(spec) -> dict:
    from triangulate.acoustic.detect import find_events

    t = time.perf_counter()
    res = find_events(spec)
    wall = time.perf_counter() - t
    return {"wall_s": wall, "events": len(res.events), "selected": sum(e.selected for e in res.events),
            "timings": res.timings}


def bench_cpa_match_silence(threads: int | None = None) -> tuple[dict, dict]:
    import polars as pl
    from bench_cpa import T0, make_scene, region_bbox, video_gates

    from triangulate.acoustic.cpa import cpa_search
    from triangulate.acoustic.matching import match_offsets
    from triangulate.acoustic.silence import silence_filter
    from triangulate.fusion.grid import GridSpec

    bbox = region_bbox(100.0)
    spec = GridSpec.for_bbox(bbox)
    origin = (spec.lat0, spec.lon0)
    tracks = make_scene(500, 100.0, 3.0, 0)
    flat = lambda la, lo: np.zeros_like(la)  # noqa: E731
    t_s = (T0, T0 + 3 * 3600.0)
    gates = video_gates(t_s)
    warm = tracks.filter(pl.col("icao") == "000000")
    cpa_search(warm, origin, spec, gates, ground_hae=flat)  # numba warm-up
    t = time.perf_counter()
    res = cpa_search(tracks, origin, spec, gates, ground_hae=flat, n_threads=threads)
    t_cpa = time.perf_counter() - t
    taus = [2152.0, 3073.0]
    t = time.perf_counter()
    hyps = match_offsets(res.table, taus, 5.0, t_s)
    t_match = time.perf_counter() - t
    t = time.perf_counter()
    sr = silence_filter(hyps, tracks, res.observers, taus, n_threads=threads) if hyps.height else None
    t_sil = time.perf_counter() - t
    cpa = {"wall_s": t_cpa, "flights": 500, "km": 100, "hours": 3, "cpa_rows": res.table.height,
           "cells_candidates": res.stats.get("cells_candidates")}
    ms = {"wall_s": t_match + t_sil, "match_s": t_match, "silence_s": t_sil, "hyps_matched": hyps.height,
          "hyps_after_silence": sr.hyps.height if sr else 0}
    return cpa, ms


def bench_score() -> dict:
    import bench_score as bs

    r = bs.main(50_000)
    return {"wall_s": r["score_s"], "calibrate_s": r["calibrate_s"], "n_hyps": r["n_hyps"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "calibration" / "bench_pipeline.json"))
    a = ap.parse_args()
    out: dict = {}
    spec = None
    if not (a.skip_decode and a.skip_detect):
        audio = one_hour_audio()
        if not a.skip_decode:
            d = bench_decode(audio, ROOT / "data" / "tmp" / "bench")
            spec = d.pop("spec")
            out["decode_stft_whiten_1h"] = d
        if not a.skip_detect:
            if spec is None:
                from triangulate.acoustic.spectrogram import compute_spectrogram

                spec = compute_spectrogram(audio)
            out["detect_fit_1h"] = bench_detect(spec)
    out["cpa_region"], out["match_silence"] = bench_cpa_match_silence()
    out["score_50k_x2"] = bench_score()
    for k, v in out.items():
        v["target_s"] = TARGETS[k]
        v["ok"] = bool(v["wall_s"] <= TARGETS[k])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        verdict = "PASS" if v["ok"] else "FAIL"
        print(f"{k:24s} {v['wall_s']:8.1f} s  (target <= {v['target_s']:.0f} s)  {verdict}")
    return 0 if all(v["ok"] for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
