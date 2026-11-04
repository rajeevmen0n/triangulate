# WP07 feasibility gate: adsb.lol vs the Oyster Bay demo (2026-09-08)

**Verdict: GO.** adsb.lol (prod release, no staging/mlatonly needed) contains one turboprop for each
demo event whose geometry matches the video's measured ranges, and the two are 924 s apart
(video: 920 s). The OpenSky fallback is not needed.

Reproduce: `nix develop --command python scripts/wp07_feasibility.py` (writes
`data/fixtures/feasibility_2026-09-08.json`); gate test: `tests/e2e/test_wp07_feasibility.py`
(`network`, `slow`).

## Method
The notes cannot identify the aircraft by ICAO hex (the video used Flightradar24 ids, e.g.
1100069709, which do not map to hex), so this is a search. Query bbox 40.50/-74.00/41.20/-73.00,
19:45-22:00 UTC; 692 aircraft, 497 excluded as jets, 195 kept. All tracks (jets included, for
comparison) prepared to 1-s ENU about the ground truth (40.876722, -73.526444); observer at
DEM + geoid + 5 m (HAE -26.1 m). `track_passes` brute-force CPA (slant 3-D distance, 3-D speed,
heard time = t_cpa + d/c, c = 343 m/s); `solve_emission` agrees with the CPA emission time to
<1e-6 s. Expected heard times: audio start 20:15:44 + 2152 s / 3073 s.

## Matches (strict: video speed and distance ranges, heard time within +-2 min of expected)
| Event | ICAO / callsign / type | CPA slant | Speed | Heard CPA | Video range |
|---|---|---|---|---|---|
| 1 | a1dfeb / CST1219 / PC12 (single turboprop) | 6152 m | 136.9 m/s | 20:53:25 (+109 s vs 20:51:36) | 130-154 m/s, 4300-7800 m |
| 2 | ace391 / EGE93 / B350 (twin turboprop) | 3530 m | 103.4 m/s | 21:08:49 (+112 s vs 21:06:57) | 88-110 m/s, 1390-4690 m |

Each is the **only** non-jet pass in its window satisfying the ranges (the plan's looser criteria,
+-5 min, d <= 9/6 km, v 120-165/80-120, also give exactly these two). Gap: 924.2 s (video 920/921 s;
within the 10 s tolerance). Both are offset by about +110 s from the heard time derived from the
notes' "8:37 pm" clip, consistent with that anchor being rounded to the minute: audio start is
more likely about 20:17:35 UTC. Test A3 (+-120 s) is therefore only just inside tolerance; the
notes' anchor should not be used tighter than +-2 min.

## Coverage quality
- Both tracks are ADS-B (`adsb_icao`), 100 % geometric altitude, so altitude sigma is about 10 m. No MLAT/TIS-B.
- CST1219: raw sampling about 20 s (12 samples within +-120 s of CPA), nearest samples 7 s before / 13 s
  after CPA. No gaps larger than 20 s; the prepared
  segment is continuous through CPA. Sparse but adequate; the
  PCHIP 1-s resample carries interpolation error, mainly in position along-track.
- EGE93: about 4 s sampling (41 samples), max gap 20 s, nearest samples 4 s / 3 s from CPA.
- Neither is removed by jet exclusion (PC12 L1T, B350 L2T are kept). Jets in the same windows
  (e.g. A339, A333) would be excluded as in the video.

## Notes and caveats
- 352 passes within 20 km in the window, so the search is not trivially unique; the pair
  discrimination relies on speed + distance + time + the 921 s gap. The silence filter and
  Doppler scoring (later WPs) provide the rest.
- Event 1's speed 136.9 m/s (266 kt) is fast for a PC-12 but within
  its cruise-descent envelope; geometric altitude/speed come from ADS-B, not the fit.
- The ground truth is the video's result, so agreement is evidence the data are compatible, not an independent proof.
