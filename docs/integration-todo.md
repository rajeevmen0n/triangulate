# Integration TODO (collected from work-package reports)

Gaps between the independently built packages. These must be closed when the real stages are wired into the pipeline (WP18 stubs → real stages), at the latest in WP22.

## Job workspace artifact contract (needed by the web UI, WP21)
The UI reads these artifacts from `data/jobs/<id>/` via `/api/jobs/{id}/artifacts/{name}`:
- `results.json`: final job result. Its shape is not yet specified; define it and update `web/static/js/results.js`.
- `candidates.geojson`, `fusion.json`, `posterior.tif`/`.npz` (WP12 `write_outputs`)
- `heatmap.png` + `heatmap.json` (WP20 `write_heatmap`)
- `spectrogram.png` + `spectrogram.json` (WP20 `render_spectrogram_window`, format `triangulate.spectrogram_overlay/1`)
- `events.json`: the full `Event` list (WP08 output), required by the UI event editor and reruns
- `metadata.json`: the WP13 `MetadataReport` (the UI currently guesses its shape)
- `clues.json`: the WP15 `VisualClues` (written by `run_visual_llm`)

## Stage registration
- `triangulate.visual.stage.register()` (WP15) is not called automatically. The pipeline must register the real visual stage.
- Register the real metadata (WP13), frames/OCR (WP14), acoustic (WP08–11), geocode (WP16), sun (WP17) and fusion (WP12) stages in place of the WP18 stubs.
- WP14 `TimestampHint` → tz-mixture time prior (WP13 `time_components` style) in the pipeline.
- WP13 "trust GPS" / "promote to hard constraint" actions are left to WP18 integration.

## Data/physics notes
- WP06: use `cpa_slope` (the slope at heard CPA), not `max_slope`, when deriving d from a slope (plan §3.1 is wrong here).
- WP07: the true audio.mp3 start is ≈ 20:17:35Z (the demo aircraft are heard ≈110 s later than the notes' 20:15:44Z anchor). Use this for any WP23 time-tolerance assertion.
- WP03: `TRACK_SCHEMA` has no callsign; the UI's "flight IDs" show ICAO hex unless a callsign column is added.
- WP03: the ICAO type table is a bundled, model-transcribed Doc 8643 subset (the tar1090-db licence is unclear). Spot-check it.

- WP10: the silence filter's "full" scope ([0, duration]) rejects every Oyster Bay hypothesis, because other audible aircraft have no exemptions. The default "between" scope works (204 cells, ground truth kept). Revisit "full" once WP08's detected-event list can feed `exempt_events`. Single-event recordings need `scope="full"`.
- WP10 real data: implied audio.mp3 start = 20:17:34.5Z (confirms the WP07 estimate).

## Unverified
- WP15 has never called the real Claude API (no key). Run the `llm`-marked test once `ANTHROPIC_API_KEY` is set.
- WP21 JS has not been exercised in a real browser.
