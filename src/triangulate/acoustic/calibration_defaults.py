"""Committed results of the WP22 lambda calibration (plan 3.3.7, 5.2 WP22).

ell_A = LAMBDA_FIT * sum_k z_k. Fitted by `scripts/bench_synthetic_e2e.py` (=
`acoustic.synthetic_e2e.run_calibration`) as the largest lambda on `synthetic_e2e.LAMBDA_GRID`
whose 90 % HPD region (fused posterior, `fusion.candidates.credible_level` of the truth's leaf)
covers the truth in >= 90 % of 40 deterministic synthetic worlds (seeds 0-39,
`synthetic_world.WorldConfig()` defaults). The per-seed output is written to the gitignored
data/calibration/{lambda,runs}.json; the numbers below are copied from that run (2026-11-04).
Coverage is flat at 0.95 for lambda <= 0.37 (the misses are seeds whose truth leaf has no
hypothesis or a wrong-by-100-m peak), 0.925 at 0.54-0.77, and falls quickly above 1.
"""

LAMBDA_FIT = 0.77385  # was 0.5 (WP11 provisional)
LAMBDA_PROVISIONAL = 0.5

CALIBRATION = {
    "date": "2026-11-04",
    "seeds": "0-39 (clean), 0-39 (MP3 32 kbps), 0-4 (single event, known T_s +- 60 s)",
    "lambda_grid": "geomspace(0.02, 10, 18)",
    "target_coverage": 0.9,
    "coverage_at_fit": 0.925,
    "coverage_curve": {
        0.02: 0.95, 0.12441: 0.95, 0.37251: 0.95, 0.5369: 0.925, 0.77385: 0.925, 1.11538: 0.80,
        1.60763: 0.70, 2.31712: 0.65, 3.33973: 0.55, 4.81365: 0.35, 6.93804: 0.20, 10.0: 0.125,
    },
    "acceptance_at_fit": {
        "top1_within_100m": 0.85,  # target >= 0.90: NOT met (misses at 107-192 m, one at 366 m)
        "median_top1_m": 28.2,
        "decoy_removed": 1.0,  # 40 / 40
        "hpd90_coverage": 0.925,  # target [0.80, 0.97]
        "mp3_32k_top3_within_250m": 1.0,  # target >= 0.80
        "mp3_32k_top1_within_100m": 0.85,
        "single_event_band_contains_truth": 1.0,  # 5 / 5 (fused 90 % HPD covers 3 / 5)
        "correct_tuple": 1.0,
    },
}
