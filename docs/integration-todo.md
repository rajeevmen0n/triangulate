# Integration TODO (collected from work-package reports)

Status after WP25 (pipeline integration, `src/triangulate/jobs/stages.py`). `[x]` = closed, `[ ]` = open.

## Job workspace artifact contract (needed by the web UI, WP21)
Written by the real stages into `data/jobs/<id>/`, served via `/api/jobs/{id}/artifacts/{name}`:
- [x] `results.json`: schema `triangulate.results/1`, defined in the `jobs/stages.py` module docstring and checked by `stages.validate_results`. Keys: `status`, `candidates[]` (rank, lat, lon, prob_mass, mass_radius_m, hpd_level, margin_to_runner_up in nats, `t_s` ISO UTC, `flights` as ICAO hex ids, per-module `contributions`), `best`, `margin {nats, acoustic_z}`, `T_s {best, window, prior_components}`, `flights {icao: {reg, typecode, category}}`, `modules {acoustic|visual|metadata: {status, weight, message}}`, `module_status`, `events {source, n_total, n_used, ids_used}`, `hpd[]`, `warnings[]` (single-event, low margin, skipped/degraded modules), `notes[]`, `attribution[]`, `cost_usd`. `web/static/js/results.js` reads it.
- [x] `candidates.geojson`, `fusion.json`, `posterior.tif`/`.npz` (stage `fusion`)
- [x] `heatmap.png` + `heatmap.json` (stage `render`, bounds = the search bbox)
- [x] `spectrogram.png` + `spectrogram.json` (stage `events`, full recording, with fitted events)
- [x] `events.json`: full `Event` list (detected, or the user-supplied `JobParams.events`)
- [x] `metadata.json`: `MetadataReport` (only when metadata is enabled); results.js renders gps/times/make/model/encoder
- [x] `clues.json`: written by the visual stage (only with consent and key)
- Also: `audio_4k.f32`, `spec.npz`, `tracks.parquet`, `flights.json`, `layers/` (cache dirs), `scores.parquet`, `acoustic.json`.

## Stage registration
- [x] `register_real_stages()` is called by the CLI `run` and `serve`; `pipeline.reset_registry()` restores stubs (tests). `create_app()` without `stages=` uses whatever is registered.
- [x] Real metadata, spectrogram/events, adsb, dem, score (acoustic_search), visual (WP15 `VisualStage`), fusion, render. `cpa` and `match` are no-op stages: the work happens inside `score` (one `acoustic_search` call), so the progress bar jumps at `score`.
- [x] WP14 `TimestampHint` and WP13 time hints become a tz-mixture prior on T_s (`time_components`), applied as a soft late re-rank of the candidates (`time_prior` contribution, alpha = 0.6). They never narrow the search window or the ADS-B fetch.
- [x] "Trust GPS": `JobParams.trust_gps` / `--trust-gps` pins the file GPS as candidate #0. GPS otherwise enters as the WP12 mixture layer (only if inside the bbox).
- [ ] "Promote time hint to hard constraint": use `constraints.known_audio_time` (the UI has no one-click button yet; the API form has no `trust_gps` / `use_metadata` fields yet).
- [x] Visual: frames/OCR are local; Claude steps only with `visual` + `visual_consent` + `ANTHROPIC_API_KEY` (checked by WP15). Visual layers (WP16) and the sun re-rank (WP17, `sun_clue_from_clues`) run in the fusion stage.
- Test seam: `jobs.stages.SERVICES` (track-provider factory, ground-height function); `TRIANGULATE_SKIP_METADATA=1` or `--no-metadata` / `JobParams.use_metadata=False` never reads container metadata (used for audio.mp3).

## Data/physics notes
- [ ] WP06: use `cpa_slope` (the slope at heard CPA), not `max_slope`, when deriving d from a slope (plan §3.1 is wrong here). Not touched by WP25.
- WP07: the true audio.mp3 start is ≈ 20:17:35Z (the demo aircraft are heard ≈110 s later than the notes' 20:15:44Z anchor). Use this for any WP23 time-tolerance assertion.
- [ ] WP03: `TRACK_SCHEMA` has no callsign; the UI shows ICAO hex (plus registration/type from `flights` when known).
- [ ] WP03: the ICAO type table is a bundled, model-transcribed Doc 8643 subset. Spot-check it.
- [x] WP10 silence scope: single-event jobs use `scope="full", exempt="aircraft"`; multi-event jobs use the default "between". Revisit "full" for multi-event once `exempt_events` is validated (WP22/23).
- WP10 real data: implied audio.mp3 start = 20:17:34.5Z (confirms the WP07 estimate).

## WP22 calibration findings
- lambda fitted to 0.774 (`acoustic/calibration_defaults.py`). 90 % HPD coverage is 0.925.
- [ ] Top-1 within 100 m is 0.85 against the 0.90 target (median 28 m). The misses are in the acoustic argmax, not in lambda.
  - [ ] WP03: interpolate sparse ADS-B tracks using gs/track (exact 1 s tracks fix 2 of 4 probed misses).
  - [ ] WP08: the slant distance d is under-estimated (0.6–0.8×) on turning tracks (up to 1.2°/s), so the truth falls outside the CPA distance gate. Synthetic heard flights are limited to ≤ 0.4°/s for now.
  - [ ] WP11: after refinement the best hypothesis can still be a 200 m cell.
- [ ] Single-event mode is overconfident at the fitted lambda: the fused 90 % HPD covers the truth in only 3/5 seeds.
- The MP3 32k synthetic test is weak: the codec is nearly transparent below 1 kHz.

## WP23 acceptance (Oyster Bay, `tests/e2e/test_acceptance_oyster_bay.py`)
- A1 PASS: best 157 m, truth at rank 1. A2 PASS: 141 m, margin z 21.8. A3 PASS: T_s* 20:17:29.7Z (+106 s vs the notes' anchor, −2 s vs 20:17:32Z).
- [ ] A4 (report-only) FAIL: single event at 1276 s. Top-1 is a069fe, 36 km off. The plan's known clip time of 20:37Z ±90 s probably excludes the true clip time (true start + 1276 s ≈ 20:38:48Z). Re-run with 20:38:48 ±90 s before tuning the single-event path.
- A5 (full-day, `TRIANGULATE_ACCEPT_A5=1`) has not been run.
- Fixture: `scripts/build_acceptance_fixture.py` → `data/cache/fixtures/adsb_2026-09-08_LI_1930-2300.parquet` (+ manifest).

## Unverified
- [ ] WP15 has never called the real Claude API (no key). Run the `llm`-marked test once `ANTHROPIC_API_KEY` is set. The visual-layer and sun-rerank branches of the fusion stage are untested end to end for the same reason.
- [x] WP21 JS exercised in headless Chromium (`nix develop .#browser --command python -m pytest -m browser tests/browser -s`): full upload → SSE → results flow, 0 console/page errors, 0 external requests. Fixed: results never rendered (fitBounds on a detached map → NaN LatLng), nested-array flattening in `el()`, a literal "null" in the detail card, status stuck at "queued", typed bbox not shown on the pick-map, form grid alignment.

## UI follow-ups (from the browser check)
- [ ] Results map fits the whole search bbox, so the posterior (50 % region ≈ 0.01 km²) is a dot. Fit to the 90 % HPD (with a minimum extent) and stretch the heatmap.
- [ ] Copernicus attribution appears twice in the results "Data and attribution" card (once from results.json and once from config).
- [ ] Candidate "probability" is mass within 300 m, so overlapping candidates sum past 100 % (100 %, 49 %, 36 % …). Label it as "mass within 300 m" or de-overlap.
- [ ] Events table and edit-events table list all 35 detected events. Collapse non-aircraft/flagged ones by default.
- [ ] Sidebar job list polls every 4 s and lags the main panel ("queued" while done).
- [ ] Time zone field is slightly out of line when its hint wraps.
