# Video notes: "I discovered a sound that leaks your exact location (and it's everywhere)"

## Evidence and provenance

Tags used throughout:

| Tag | Source |
|---|---|
| **[F mm:ss]** | Frame I looked at, from the user-supplied local copy `video.mp4`. On-screen text was read from frames at 512 px or 1024 px width. The source video is only 640x360, so very small text is approximate. |
| **[TR mm:ss]** | Transcript: the user-supplied `sub.srt` (subtitles, 840 cues, 00:00 to 29:03). Its provenance is unknown and it was not machine-verified. It contains a few likely mis-hearings, e.g. "closest point of arrival" for *approach*. |
| **[D]** | Uploader's YouTube description, fetched from the public watch page. |
| **[TH]** | YouTube thumbnail. |
| **[I]** | My inference or standard physics. Not confirmed by the video. |

How the evidence was gathered:
- YouTube blocked yt-dlp ("Sign in to confirm you're not a bot"), so no cookies were used and analysis switched to the local file.
- **Local watch run** (balanced detail, 100 scene-change frames over the full video): `watch-runs/watch-1snv_0c6/`
- **High-res cue reruns** (`--detail transcript --resolution 1024 --timestamps ...`):
  - `watch-runs/watch-girkc249/`: 32 cues taken from on-screen UIs, plots and text
  - `watch-runs/watch-kqw5bg2g/`: 22 cues taken from "look at this" moments in the transcript
  - `watch-runs/watch-gbxb33_b/`: 3 cues (the Google Maps coordinate panel, and the ADS-B track fields)
- Limited audio cross-reference: segment spectrograms of `audio.mp3` in `watch-runs/audio-xref/`
- The local file has no embedded subtitle stream, and the watch skill produced no transcript (whisper backend = none). The only transcript is `sub.srt`.

---

## 1. Metadata and summary

| Field | Value |
|---|---|
| Title | I discovered a sound that leaks your exact location (and it's everywhere) |
| URL | https://www.youtube.com/watch?v=UytYrhYLOKI |
| Channel | colsto (https://www.youtube.com/@colsto), about 89.8K subscribers |
| Published | Nov 2, 2026. About 2.28M views when fetched. |
| Duration | **29:12** (1751.5 s), from the local file (640x360 h264 + AAC) |
| Companion video | "Unknown location - raw audio": https://www.youtube.com/watch?v=4CxVxuMVkMc [D] |
| Code released? | No [D] |

**Summary.** The presenter geolocates a **security-camera video** that shows only a blurry, pixelated tree line against the sky. The only givens are "filmed **September 8**, somewhere in the **continental US**" [TR 00:19–00:31]. The recording contains two faint, effectively inaudible **propeller-aircraft Doppler curves** whose closest approaches are **920 s apart** [TR 01:25–02:58, F 02:51]. He took Flightradar24 tracks for every flight over the US that day, removed known jet types, and placed 200 m grids around the flight paths, using a **DEM** for ground elevation [F 06:22]. He then filtered candidate points through several stages:
1. "heard a flight, then another one 920 s later", using the **delayed (retarded) CPA time**
2. aircraft speed and CPA slant distance, measured by fitting each Doppler curve
3. a **silence filter**: no other flight was heard between the two events

He then forward-modelled the exact Doppler curve each surviving point would have heard, taking emission-time radial velocity from the real flight geometry. Each modelled curve was scored by **summing spectrogram brightness under the curve**, using a "**locked**" two-event score [TR 18:12–20:12].

The winning point is about **40.876 N, -73.525 W**, which is **Oyster Bay, New York**. It won by a large margin (combined score 20.48 against 15.25 for the runner-up) [F 21:42]. A separate test used one 1-minute clip with a known time (20:37 UTC) and a US-wide search. It ranked the same spot first [TR 22:47–24:13, F 23:30]. He confirmed the location by matching the tree-line silhouette in **Google Street View** near **19 Bayview Ave, Oyster Bay**, and found the security camera on a pole [F 25:45, 26:06; TR 25:20–26:09].

The video ends with countermeasure experiments in Audacity:
- a high-pass filter plus MP3 export does **not** remove the curve
- a loud JBL speaker does not drown it out
- the curve **survives phone-call compression**

The mitigations he suggests are making cuts in the audio and adding noise cancellation [TR 26:09–28:10].

### Corrections to the earlier description-only notes
- **The location is in New York, not California.** The thumbnail's "…, CA 90011" address is decoration or clickbait. The real result is Oyster Bay, NY [F 21:42, 25:28, 25:45].
- **The ADS-B source is named: Flightradar24.** The presenter shows the Flightradar24 web map [F 03:04] and the **`pyfr24`** GitHub repo, "A Python client for the Flightradar24 API with CLI support. Fetch, plot and analyse flight data" [F 03:04, low-res frame]. He says "they have an API you can pay for; there's also a Python package to scrape the data" [TR 03:02–03:08]. My earlier guesses (OpenSky, adsb.lol, ADSBx) were wrong.
- **Prop/jet classification works by exclusion.** He uses the ICAO DOC 8643 "Aircraft Type Designators" table [F 04:01]. Because that lookup has gaps, he **filtered out known jet types** rather than keeping known props [TR 03:55–04:15].
- **The time of day was not known in the main demo.** Only the date and "continental US" were known. The 1-min test is the one with a known time, 8:37 pm UTC [TR 23:04].
- **Grid sizes.** The 200 m grid covers a 10 km radius around each flight path [TR 05:13–05:20]. It is later **densified to 100 m** for scoring, because the true point can lie up to 141 m from a 200 m grid node and then score badly [TR 20:12–20:45].
- **Scoring was not a ridge least-squares fit.** It sums the spectrogram brightness that each modelled curve covers. For the faint 1-min clip, he hand-traced the curve and scored each candidate by its distance to the traced line instead [TR 18:12–18:27, 23:32–23:45].
- **The DEM's resolution is about 1 km.** The QGIS layer is named `dem_mosaic_1km` [F 06:22]. The transcript only says "country-wide DEMs" [TR 02:08].

---

## 2. Timestamped walkthrough

| Time | What is shown and said | Source |
|---|---|---|
| 00:00–00:15 | Intro montage from earlier geolocation work (rainforest structures, a deep-sea dive clip showing "水深651mに着底"). "Single most powerful geolocation clue I've come across." | F 00:05, TR 00:00–00:15 |
| 00:15–00:35 | The target: a static, blurry security-cam clip of a tree line. Givens: Sept 8, continental US. | F 00:18, TR 00:15–00:35 |
| 00:41–01:02 | Compares with his earlier thunder-based geolocation video (about 5 km precision). Cartoon of thunder and a bird behind a brick wall: low frequencies travel far and penetrate walls. Shows a final QGIS heatmap as a teaser. | F 00:45, 00:56, 01:03; TR 00:41–01:01 |
| 01:06–01:57 | Plays two audio snippets; "you probably couldn't tell". Opens a **spectrogram viewer**, zooms to the right band and raises the resolution, and a Doppler curve appears. Viewer settings: window size 2048, Hamming, zero-padding factor 1, linear frequency scale, display range 5–800 Hz, "Rose" palette, gain 20 dB. | F 01:44 (settings panel), 01:57; TR 01:06–01:57 |
| 01:57–02:12 | Scale of the project: "months to build, over 5,000 lines of code, 50 GB parquet files, 3D physics modelling, country-wide DEMs". | TR 01:57–02:12 |
| 02:12–02:58 | Finds a second, louder curve with **harmonics on top**. These two events are what he uses, and their CPAs are **920 s** apart. Introduces ADS-B (position, speed, altitude). | F 02:28, 02:51; TR 02:12–02:58 |
| 03:01–03:30 | Data acquisition: Flightradar24 (paid API) and the `pyfr24` scraper, run "for a few hours" to get every flight over the continental US on Sept 8. Full flight paths are downloaded, so some tracks reach Hawaii and abroad. QGIS shows a yellow mass of US tracks. | F 03:04, 03:36 (yellow tracks); TR 03:01–03:29 |
| 03:35–04:15 | Plays the louder event with frequencies boosted: it is clearly a prop plane. ICAO DOC 8643 type-designator table (columns Manufacturer, Model, Type Designator, Description, Engine Type, Engine Count, WTC). Known jets are removed, giving an orange "prop/unknown" US track map. | F 04:01, 04:14; TR 03:35–04:15 |
| 04:16–05:25 | **Time-offset filter on a small test area.** QGIS Identify panel on a pink flight line, layer `flight_<id>_track_1sec`, fields: `fid, timestamp (epoch), utc_time, lat, lon, alt_ft (4600), vs_fpm, track_deg (260.3), gs_kt (~116), valid`. A spreadsheet of flights has columns `flightid, registration, typecode (B78X, F2TH, M20P, PA44, B39M, COL4, EC35, B752, P28A, C162, C172, CL35, GALX, C150…), callsign, first_seen_utc, last_seen_utc, max_alt, sample_lat, sample_lon`. A 200 m point grid is laid within 10 km of the flight. CPA is defined, and a Doppler curve lasts "almost 2 minutes", so the CPA time sits at its middle. | F 04:26, 04:38, 04:50, 05:03; TR 04:16–05:25 |
| 05:03 | The spectrogram viewer's "Selection analysis" panel: Start/End, RMS level, peak level, dominant frequency (96 Hz), crest, clipping, noise floor (-63.2 dBFS), centroid, ZCR, plus a band-energy breakdown (0–250 Hz ≈ 88 %). | F 05:03 |
| 05:24–06:37 | **Delayed CPA time.** Layer `flight_<id>_candidate_gridpoints` has fields `retarded_cpa_time, cpa_distance_m (3364.8), ground_elevation_m (257.5)`. The QGIS measure tool shows about 3105 m. Distance must be the 3D **slant** distance. ADS-B altitude is above sea level, so the ground elevation of each grid point comes from the DEM (`dem_mosaic_1km`, continental US). Worked example: CPA at 22:46:34, heard at 22:46:46. | F 05:24, 06:02 (triangle cartoon), 06:22, 06:50; TR 05:27–06:37 |
| 06:37–07:47 | Every grid point stores a list of (flight ID, delayed CPA time). A "CPA Timeline" tool slides a **920 s window** looking for a pair of CPAs at that offset, allowing about **±5 s** of slack. | F 07:04, 07:15, 07:28; TR 06:37–07:47 |
| 07:49–08:27 | About **35,000 points** survive in the test area alone, which is under 0.05 % of the continental US land area. A football-field analogy for the size of the US. | F 07:31, 07:48, 08:06, 08:13, 08:18; TR 07:49–08:27 |
| 08:28–09:38 | "DOPPLER CURVES" section. Two clips: a bigger frequency drop means a faster source. For linear constant-velocity motion, **velocity sets the size of the drop and CPA distance sets how fast the drop happens**. A two-channel spectrogram of a fast passing source, and a side-by-side comparison of a small drop and a large drop. | F 08:26, 08:43, 08:57; TR 08:28–09:38 |
| 09:38–10:37 | **Interactive "Doppler Curve Fit" tool** (Plotly, window 01:50:30–01:52:15). Sliders: `f0 (fundamental freq)`, `v (plane speed)`, `d (slant distance)`, `t0 (CPA time)`. Readouts: `f_pre (approach)`, `f_post (recede)`, `max RoC` (Hz/s). There is a snapshot log ("MIN/MAX auto-calculated, BEST = midpoint"). Snapshots seen are in the table after this one. The left side of the curve does not fit an ideal model, so he takes **generous ranges**. "Decent fit at 100 m/s… pretty sure it wasn't going 70." The CPA time is uncertain by a couple of seconds (1:51:15 to 1:51:13). | F 08:57, 10:00, 10:08, 14:02, 14:08, 18:22; TR 09:38–10:37 |
| 10:33 | **Measured ranges**: event 1 (left) **130–154 m/s, 4300–7800 m away at CPA**. Event 2 (right) **88–110 m/s, 1390–4690 m away at CPA**. Gap 920 s. | F 10:33, 12:19, 12:28 |
| 10:37–11:34 | Speed filter on the US map: green = flight segments matching event 1's speed, pink = event 2's. Candidate points are placed only at the measured slant-distance range from those segments, then offset-filtered, giving about **73,000** points (later called "roughly 70,000"). Zooming in shows they cluster at **airports**. | F 10:40, 10:49, 10:50, 11:12, 11:19–11:24; TR 10:37–11:34 |
| 11:34–12:57 | **Silence filter.** About 15 min with no plane heard is itself a signal. On-screen rule: disqualify any candidate that "heard" another flight between event 1 and event 2. The **audible threshold distance is max(event1 CPA, event2 CPA)**. A 1.5x or 2x multiplier would be reasonable, but none is used, to stay conservative. This takes 70,000 down to **15,000**. | F 12:04, 12:09, 12:12, 12:51; TR 11:34–12:57 |
| 12:57–14:09 | "Thanks for watching" fake-out. Real planes change heading, speed and altitude, and that information is encoded in the curve's deviations from the ideal shape. | TR 12:57–14:09 |
| 14:09–15:35 | Plan: forward-model, for each candidate and flight, the exact Doppler curve that the real geometry would produce. A radial-velocity demo: plane at 90 m/s, heading 47°, radial component 21.6 m/s at 76.1° off-nose. Formula: v_r = v · cos(angle off nose). | F 15:30 (17:00 cue), 14:34, 14:50; TR 14:09–15:35 |
| 15:35–17:56 | "Building the Doppler Curve, Step by Step" tool (Position and Delayed modes). Example: speed 140 m/s, CPA distance 980 m, altitude 1020 m, f0 130 Hz, c = 343 m/s. **Doppler equation f = f0 · c / (c + v_r)**. Naive mode uses the aircraft position at reception time (the TR example gives 62.7 m/s, which yields 109.9 Hz). Correct "delayed" mode solves **t_e = t_r − r(t_e)/c** for the emission time. At t_r = 25 s this gives t_e = 16.910 s, r = 2775 m, v_r = 121.16 m/s, f = 96.07 Hz. The wrong method does not just shift the curve in time; it changes its shape. "Right vs Wrong Model – Real Data (1:35:52, flightid 1100069709)" compares the two on the real spectrogram. | F 16:24, 17:30, 18:03, 18:41, 19:04; TR 15:35–17:56 |
| 17:56–18:27 | The model is run for all candidates and flights. A grid of many modelled curves over spectrogram thumbnails, each labelled with FlightID and coordinates. **Score = sum of spectrogram brightness under the modelled curve.** | F 18:03; TR 17:56–18:27 |
| 18:27–20:12 | **Locked two-event scoring.** Event 1 is swept over a few seconds of time offset and over f0, and the best combination is kept. Event 2 is swept over f0 only; its time position is fixed relative to event 1 by flight geometry. A dashboard shows one shared `dt` (0.74 s) that shifts both modelled curves: event1 (1:35:52) score 8.89, event2 (1:51:13) score 8.43. | F 19:41; TR 18:27–20:12 |
| 20:12–20:58 | 200 m spacing is too coarse because a point can be 141 m off, so the grid is densified to 100 m. That gives roughly 30–40k candidates to score, which is very expensive, and is why the earlier filters matter. | TR 20:12–20:58 |
| 21:01–22:23 | Results. Scatter of event-1 score against event-2 score (about 29k points). The caption names script `1_2b_v7.py` and function `curve_match_scores_locked_spectrogram`. Points are grouped by flight pair and the best of each pair kept. **Top 7 flight-pair combos** are listed after this table. | F 21:13, 21:42, 21:44, 22:01–22:20; TR 21:01–22:23 |
| 22:23–22:47 | QGIS, with the winning flight pair rescored at finer resolution. The event-1-only heatmap is a **diagonal band**. The event-2 heatmap is another band, and the combined heatmap is a **compact blob** on the harbor shore. | F 22:33, 22:37, 22:41 (cues), 22:41; TR 22:23–22:47 |
| 22:47–24:13 | **Single-event test**: a 1-minute clip at elapsed 1:21:16, with the time given as **8:37 pm UTC**. Flight segments are filtered to that time and the speed range. Histogram "Best Score Per Flight – 1:21:16 (253 flights)": n = 253, min -63.48, max -1.74, mean -26.53. The curve was too faint for brightness scoring, so it was **hand-traced** ("traced composite line (LINE_POINTS)") and scored by distance to that line. Results: **top1 flight 1100029523 at (40.877922, -73.531941), score -1.74, dt +0.45 s**; top2 (41.044589, -77.739976) -3.68; top3 (43.692244, -79.325785) -4.66; top4 (32.072902, -102.151974) -4.70. This curve is about 500 Hz falling to about 310 Hz. The same spot ranks first, by a margin, under both methods with different flights. | F 23:05, 23:17, 23:30, 23:47, 23:58, 24:03; TR 22:47–24:13 |
| 24:13–25:18 | Generalisation slide. Example 1: spatial = full US, temporal = 1 day, 2 flights (~30 min). Example 2: full US, 1 hour, 1 flight (~60 s). "You need one of these to be good." Knowing the city (about 0.01 % of land area) relaxes the time constraint. "Within a couple hundred feet from one minute of audio." | F 25:05; TR 24:13–25:18 |
| 24:36–24:52 | Zooms out from the continental US to the San Francisco Bay area and California. The purpose is not explained in the transcript. | F 24:36–24:52 |
| 25:20–26:09 | **Visual verification.** He takes the heatmap centre, which is in the water, so the target is on the bank. He Googles the coordinate "40.8767282,-73.5264340" and gets a map pin on Oyster Bay harbor. In **Google Street View** ("19 Bayview Ave, Oyster Bay, New York", imagery from Sep 2025) he looks for the tree-line silhouette: "V-shaped tree followed by a point and then this big rounded part". Side-by-side match, then he looks up to find "the exact security camera" on a utility pole. | F 25:28, 25:29, 25:34, 25:45, 26:00, 26:06; TR 25:20–26:09 |
| 26:09–26:45 | **Countermeasure 1: high-pass filter (Audacity).** Doppler frequencies overlap the human voice. After a high-pass and MP3 export the curve looks gone, but boosting all frequencies brings it back. | F 26:09, 26:31, 26:35, 26:40; TR 26:09–26:45 |
| 26:45–27:11 | **Countermeasure 2: drowning it out.** A JBL speaker across the room played a Doppler curve under his speech, and the curve is visible beneath the speech in Audacity at about 60–100 Hz. Only **cuts** in the audio defeat the time-offset method. | F 26:57; TR 26:45–27:11 |
| 27:11–28:05 | **Countermeasure 3: phone call.** Calls have exact timestamps, which suits the single-event method. A speaker in another room played a Doppler curve, with the phone under squish balls. The curve survives compression and noise reduction. A gap appears while the call is muted, and the curve returns when he talks again. | F 27:11, 27:33, 27:40, 27:56; TR 27:11–28:05 |
| 28:05–29:03 | Mitigations: cut the audio, or add noise cancellation. Odds of misuse are slim. Use cases: investigations, or videos online with "literally zero visual clues". Closing Google Earth 3D views of the harbor and a globe. | F 28:10, 28:26, 28:38, 28:50; TR 28:05–29:03 |

**Doppler Curve Fit snapshots** (09:38–10:37 row). Fit window 01:50:30–01:52:15:

| f0 (Hz) | v (m/s) | d (m) | t0 | f_pre / f_post (Hz) | max RoC (Hz/s) |
|---|---|---|---|---|---|
| 131.5 | 117 | 4280 | 1:51:15.90 | 199.58 / 98.05 | 1.222 |
| 133.5 | 112 | 3460 | 1:51:12.90 | 198.23 / 100.64 | 1.411 |
| 130 | 107 | 3230 | 1:51:14.70 | 188.94 / 99.09 | 1.343 |
| 130.5 | 100 | 2965 | 1:51:15.40 | 184.20 / 101.04 | 1.286 |

Spectrogram colour bar: 0 to -80 dB.

**Top 7 flight-pair combos** (21:01–22:23 row):

| Rank | Flights | Location | Combined | Event 1 | Event 2 |
|---|---|---|---|---|---|
| 1 | 1100069709 + 1100048665 | **(40.876466, -73.524951)** | **20.48** | 10.64 | 9.84 (LOCKED) |
| 2 | 1100088546 + 1100072793 | (45.699498, -73.643756) | 15.25 | 9.52 | 5.73 |
| 3 | 1100029918 + 1100060374 | (36.454080, -94.456016) | 15.04 | 7.90 | n/a |
| 4 | 1100065108 + 1100061857 | (42.012829, -81.426671) | 14.87 | 8.33 | n/a |
| 5 | 1100069287 + 1100070751 | (43.437551, -79.358276) | 14.85 | 5.75 | n/a |

Some runner-up coordinates (near Montreal, and in Lake Ontario) lie outside the continental US, so the candidate set was not clipped to CONUS land [I].

**Event clock times.** Event labels use **elapsed recording time**. 1:35:52 = 5752 s and 1:51:13 = 6673 s, which is a gap of 921 s, roughly the "920 s" [F 19:41, I]. So the source recording is more than 1.5 h long. The recording's total length is not stated.

---

## 3. Clue types used

| Clue | How it narrowed the location | Source |
|---|---|---|
| **Prior: date and region** | "Sept 8, continental US" defines the ADS-B query (1 day, all CONUS flights). | TR 00:26–00:31 |
| **Aircraft Doppler curves (tonal prop noise and harmonics)** | The core signal. Each curve gives speed, CPA slant distance, CPA time and f0. | TR 01:50–02:35, F 10:33 |
| **Aircraft type** | Prop sound means jets are excluded via ICAO DOC 8643 type designators. | TR 03:35–04:15, F 04:01 |
| **Relative timing of events (920 s)** | Candidate points must hear two qualifying flights about 920 ± 5 s apart, measured in delayed CPA time. | TR 04:16–07:47 |
| **Speed and CPA distance per event** | Restricts which flight segments count and where along them candidates may sit. | TR 09:11–11:15, F 10:33 |
| **Silence (negative evidence)** | Reject candidates that would have heard another flight within max(CPA1, CPA2) between the events. This removes airport clusters. 70k becomes 15k. | TR 11:34–12:57, F 12:09 |
| **Fine curve shape (non-linear flight)** | Real heading, speed and altitude changes in the ADS-B track make each candidate's predicted curve unique. Scored against the spectrogram. | TR 12:57–20:12 |
| **Terrain elevation** | Ground height for slant range and propagation delay. | TR 05:58–06:25, F 06:22 |
| **Absolute time (1-min test only)** | 20:37 UTC limits segments to one time slice, so a single event is enough. | TR 23:04–23:18 |
| **Visual: tree-line silhouette** | Final verification in Street View (V-shaped tree, a point, then a rounded crown). Then the camera on the pole is identified. | TR 25:33–26:09, F 26:00, 26:06 |
| Signage, text, architecture, vehicles, plates, sun/shadows, EXIF | **Not used.** The video was chosen to have no visual clues. | TR 28:30–28:37 |

---

## 4. Tools, data and techniques

| Item | Use | Source |
|---|---|---|
| **Flightradar24** (web map; paid API) | Source of ADS-B flight tracks | F 03:04, TR 03:02 |
| **`pyfr24`** (GitHub, "Python client for the Flightradar24 API with CLI support") | Bulk download of every CONUS flight on Sept 8. Ran for hours, producing about 50 GB of parquet. | F 03:04 (low-res), TR 02:06, 03:04–03:12 |
| **Parquet** | Storage for the flight data (50 GB files) | TR 02:06 |
| **ICAO DOC 8643 Aircraft Type Designators** (icao.int lookup table) | Typecode to description, engine type and count. Used to remove jets. | F 04:01, TR 03:59–04:15 |
| Spreadsheet (Excel-style) | Inspecting the flight list (`flightid … sample_lon`) | F 04:38 |
| **QGIS** with satellite basemaps (layer names include google, carto, bing, norway) | All spatial work: tracks, grids, Identify panel, measure tool, heatmaps. Layer names seen: `flight_<id>_track_1sec`, `flight_<id>_candidate_gridpoints`, `dem_mosaic_1km`, `post silence + offset…`, `event scores`, `vs_locked_…`, `1:35:52_candid…`, `1:21:16_…` | F 01:03–26:00 (many) |
| **DEM**: a CONUS mosaic, layer `dem_mosaic_1km` (product not named) | Ground elevation per grid point | F 06:22, TR 02:10, 06:18 |
| **Spectrogram viewer** (dark UI with multi-view and waveform, selection analysis, settings panel). Appears to run inside a VS Code-style editor; the status bar reads "Claude Code". [I: probably a custom or extension-based tool; not named] | Finding and measuring Doppler curves | F 01:44, 01:57, 05:03, 08:43; crop of the status bar |
| **Custom Plotly HTML dashboards**: "Doppler Curve Fit", "Building the Doppler Curve, Step by Step", radial-velocity demo, "CPA Timeline", "Right vs Wrong Model", locked dual-event viewer, score scatter, "Top 7 flight-pair combos", "Single-Event: Top 8 Flights", "Best Score Per Flight" histogram | Manual curve fitting, explaining the method, inspecting results | F 08:57–23:58 |
| **Python pipeline**: more than 5,000 LOC, script `1_2b_v7.py`, function `curve_match_scores_locked_spectrogram` | Grid generation, retarded CPA, filters, forward model, scoring | TR 02:03–02:05, F 21:13 |
| **Google Search / Google Maps / Street View / Google Earth 3D** | Viewing the coordinate, street-level tree-line verification, finding the camera | F 25:28, 25:45, 26:06, 28:10 |
| **Audacity** (spectrogram view, high-pass filter, MP3 export, call recording) | Countermeasure experiments | F 26:31–27:56 |
| JBL speaker, phone call | Physical tests of masking and phone-codec survival | TR 26:53–27:46 |

Physics and constants shown on screen [F 16:24–18:41]:
- c = 343 m/s
- Doppler shift: `f = f0 · c / (c + v_r)`, with v_r positive when receding
- Radial velocity: `v_r = v · cos(angle off nose)`
- Emission time: solve `t_e = t_r − r(t_e)/c`
- The "max RoC" readout matches `f0·v²/(c·d)`. Checked against one snapshot: 131.5·117²/(343·4280) ≈ 1.23 Hz/s, and the screen shows 1.222 [I].

---

## 5. Decision logic (generate, then narrow, then verify)

1. **Observe** [TR 01:25–02:35]: spectrogram (window 2048, Hamming, 5–800 Hz). Find Doppler events by eye. Pick two, at elapsed times 1:35:52 and 1:51:13.
2. **Measure** [TR 09:38–10:37, F 10:33]: fit each curve manually with the slider tool. Record MIN and MAX plausible values, with BEST = midpoint, for v, d, t0 and f0. Keep the ranges generous.
3. **Generate** [TR 03:01–05:25]: all Sept 8 CONUS flights, minus known jets. Lay a 200 m grid within 10 km of each segment and compute the **retarded CPA time** with DEM elevation and slant range.
4. **Narrow** [TR 07:05–12:57]:
   1. 920 ± 5 s offset between two qualifying flights' delayed CPAs
   2. per-event speed range and slant-distance range, giving about 73k (~70k) points
   3. silence filter (threshold distance = max of the two CPA distances, no multiplier), giving 15k points
5. **Score** [TR 14:09–20:58]:
   1. densify to 100 m, giving about 30–40k points
   2. forward-model each candidate's curve from the interpolated real track using emission time
   3. score = summed brightness under the curve
   4. event 1 is swept over ±a few seconds and over f0; event 2 is swept over f0 only, with its time locked to event 1
   5. combined score = event1 + event2 (20.48 = 10.64 + 9.84)
6. **Rank** [TR 21:01–22:23]: keep the best point per flight pair. The winner is clearly separated: 20.48 against 15.25.
7. **Refine** [TR 22:23–22:47]: rescore only the winning flights on a finer grid. Each single-event heatmap is an elongated band (the distance ambiguity along the track); their intersection is a tight blob.
8. **Cross-validate** [TR 22:47–24:13]: an independent single-event, known-time test across the full US gives the same spot as top-1.
9. **Verify visually** [TR 25:20–26:09]: Street View silhouette matching of the tree line, then finding the camera.

---

## 6. Automation opportunities for the "upload a video, find the location" app

The video's method is an **audio module**. Its last step, **visual verification**, maps directly onto the app's visual pipeline. The presenter did several steps by hand: event picking, slider fitting, line tracing and Street View matching. Those are the main automation targets.

### 6.1 Ingest and metadata
- **Automate**: ffprobe/exiftool for creation time, GPS atoms, device and timezone. Let the user enter priors (date, region, approximate time). These are exactly the givens the presenter relied on.
- **Hard parts**: platform re-encodes strip metadata. Security cameras burn timestamps into the image, so **OCR of on-screen timestamps** is worth adding [I]. The demo's "time unknown, date known" case costs far more compute than a known time.

### 6.2 Audio to spectrogram
- **Automate**: ffmpeg mono extract at the native rate. STFT with about a 2048-sample Hamming window (as shown). Analyse 5–800 Hz: the event curves were about 100–200 Hz and the 1-min clip about 300–500 Hz. Apply per-bin median normalisation or whitening and harmonic stacking.
- **Hard parts**: the curves are faint and inaudible. The presenter had to "zoom into the right frequency and increase resolution". Codec loss and wind noise. However, he showed that MP3 and phone codecs did not erase the curves.

### 6.3 Event detection and curve fitting (manual in the video)
- **Automate**:
  1. Find ridges with a ridge or Viterbi tracker, or a small CNN trained on synthetic Doppler curves.
  2. Fit `f(t) = f0·c/(c + v_r(t))` for a straight-line pass by nonlinear least squares to get f0, v, d and t0.
  3. Produce **confidence intervals** in place of the presenter's MIN/MAX snapshots.
  4. For faint curves, do what he did by hand: extract a polyline ("LINE_POINTS") and score by distance to it.
- **Hard parts**: real curves deviate from the straight-line model. The left side of event 2 did not fit, which is why he used generous ranges. Overlapping sources. Separating aircraft from cars and trains (much sharper drops). Choosing which events to use.

### 6.4 Flight data
- **Automate**: ADS-B for the date and region. Options:
  - Flightradar24 API, which the presenter used; it is paid, and `pyfr24` wraps it
  - [I] OpenSky historical data or adsb.lol open dumps as cheaper or open alternatives
  
  Store tracks as parquet. The presenter's single CONUS day was about 50 GB. Join typecode against ICAO DOC 8643 and **exclude known jets** rather than requiring known props, to cope with lookup gaps.
- **Hard parts**: cost and licensing, coverage of low general-aviation traffic, and data volume. Tracks must be interpolated to 1 s for emission-time maths (the layer is named `track_1sec`).

### 6.5 Candidate generation and filtering
- **Automate**: vectorised geometry over track segments, using a DEM for elevation. Ground resolution should be finer than 1 km if possible: the presenter used a 1 km mosaic, and I'd suggest Copernicus GLO-30 or USGS 3DEP [I].
  - Coarse grid of 200 m within 10 km of each segment
  - Speed and CPA-distance gating per event
  - Retarded-CPA offset matching with ±5 s tolerance
  - Silence filter with radius = max(CPA1, CPA2)
  - Densify to 100 m before scoring
- **Hard parts**: computational cost. Without the cheap filters, a full-day CONUS search is infeasible ("you just could not do this on the full US for the full day"). The silence rule depends on loudness and mic sensitivity, which is why he kept the threshold conservative. Speed of sound varies with temperature, and wind is ignored [I].

### 6.6 Forward model and scoring
- **Automate**: for each candidate and flight, solve the emission time per reception sample, compute v_r from the interpolated track, and synthesise f(t), including harmonics as an [I] extension. Score:
  - by brightness integrated along the curve, or
  - by distance to an extracted ridge when SNR is low
  
  Use locked multi-event scoring: one shared clock offset dt, per-event f0 sweeps, and fixed inter-event spacing. Group by flight combination and keep the best point. Report the score margin over the runner-up as confidence (20.48 against 15.25 in the demo).
- **Hard parts**: a GPU or JAX-style batch would be needed for tens of thousands of candidates multiplied by sweeps. Score calibration across recordings. A single event gives an elongated band and needs time or other constraints to collapse it.

### 6.7 Output and verification (the visual module)
- **Automate**: heatmap output, reverse geocoding, and fetching Street View, Mapillary or KartaView imagery around the top-k points. Match the video's static scene against it: skyline and tree-line silhouette matching (edge/horizon profile correlation), or VLM comparison. Also allow for the observer standing on the shore when the heatmap peak is in water.
- **Hard parts**: the target frame is blurry and pixelated. Street View dates differ from the recording date (here Sep 2025 imagery). Camera height and field of view are unknown.

### 6.8 Other app modules (not shown in this video)
OCR of signs and timestamps, VLM clue extraction, OpenStreetMap/Overpass queries, sun and shadow analysis, other acoustic cues (thunder, which he covered in a previous video at about 5 km precision; sirens; ENF), and reverse image search. These need other sources.

### 6.9 Ethics
The presenter withheld the code [D]. He notes that a security camera's audio was enough, and that phone calls carry exact timestamps [TR 27:11–27:27]. An app that does this can locate private individuals, so build in consent, rate limits and coarsened outputs.

---

## 7. Dev test fixture

Dev test fixture: audio.mp3; user-provided ground truth 40°52'36.2"N 73°31'35.2"W (≈40.876722, -73.526444).

Date and time as stated in the video: filmed **Sept 8** [TR 00:26–00:29]. The only time of day given is for the 1-minute single-event test clip: "8:37 p.m. UTC" [TR 23:04–23:07]. No full timestamp is shown for the main two-event recording. The presenter searched that whole day [F 25:05].

**Ground-truth cross-check.**
- The video's final Google Maps panel reads **40°52'36.2"N 73°31'35.2"W / 40.876728, -73.526434, "VFGF+MCW Oyster Bay, New York"** [F 25:31, 1024 px cue]. That is about 1 m from the user's coordinate, so the two agree.
- The pipeline's best two-event grid point (40.876466, -73.524951) is about 129 m away. The best single-event point (40.877922, -73.531941) is about 481 m away. These are haversine distances I computed [I].

**Cross-reference to the audio (limited check).**
- `audio.mp3` is 3600.3 s long, matching the viewer's "60:00.000" [F 01:44].
- Times on the video's Plotly dashboards ("elapsed recording time") are **3600 s ahead** of positions in this file:
  - event 1 (1:35:52) is at about 2152 s in the file
  - event 2 (1:51:13) is at about 3073 s
  - the single-event clip (1:21:16) is at about 1276 s (the viewer cursor reads "1276.198s" [F 28:26])
- Segment spectrograms in `watch-runs/audio-xref/` show the expected descending tones at those positions:
  - event 1: about 175 to 100 Hz with 2x and 3x harmonics
  - event 2: about 175 to 110 Hz
  - single event: about 500 to 330 Hz, faint

---

## 8. Remaining gaps
- A balanced run of 100 frames over 29 minutes is sparse. Cue reruns covered the transcript's "look here" moments, but some brief UI states may have been missed, such as the numbers in the full Top 8 single-event grid and the event 2 heatmap on its own.
- Event 2's measured speed range on the slide (88–110 m/s) differs from the fit snapshots shown for the 01:50:30–01:52:15 window (100–117 m/s). The final ranges actually used are not fully explained.
- Not shown or not named: the spectrogram app, the DEM product, the audio sample rate, the exact sweep sizes for dt and f0, how brightness is normalised, and how the 30k+ candidates were run (hardware and runtime).
- Why the video zooms to the San Francisco Bay and California at 24:36–24:52 is unclear from the transcript.
- The transcript is user-provided and unverified. The numbers quoted from it (35k, 73k, 15k, 30–40k, ±5 s, 10 km, 141 m) are spoken figures.
