# Responsible use

triangulate estimates where an audio or video recording was made, from aircraft heard in it.

**Purpose.** OSINT verification, journalism, research, and checking your own recordings. It is not a
tool for locating private individuals, and it does not identify people. The visual prompts forbid
naming or describing people or reading faces, and plate numbers are never stored.

**No bulk use.** One file per job. There is no batch or crawler API and no URL ingestion of third-party
social media. The API allows at most 20 new jobs per rolling 24 hours (`TRIANGULATE_MAX_JOBS_PER_DAY`,
0 disables).

**Read the uncertainty.** Results are probabilistic. Read the probability region (HPD), the
runner-up margin and the per-module contributions. A single pin is never shown without its region.
Single-event and low-margin results are flagged. Metadata hints are labelled untrusted.

**Local by default.** The server binds to 127.0.0.1. Audio, ADS-B, DEM and OCR processing stay on
the machine. Third-party calls are listed at `/api/health`: adsb.lol/GitHub, Copernicus S3, OSM
services, the optional basemap (`TRIANGULATE_BASEMAP_ENABLED`), and the Claude API only for jobs
where you opted in to the visual module and gave consent for that job.

**Retention.** Job workspaces, uploads and job rows older than `TRIANGULATE_RETENTION_DAYS` (default
7; 0 disables) are deleted when the server starts and by `triangulate purge [--days N] [--dry-run]`.
Deleting a job (`DELETE /api/jobs/{id}`) removes it at once. The purge only removes entries inside
`data/jobs` and `data/tmp`. It never follows or removes symlinks and never touches `data/cache`.

**Attribution.** Results and exports carry these notices:

- Flight data © adsb.lol contributors (ODbL 1.0)
- © OpenStreetMap contributors (ODbL 1.0)
- Contains Copernicus DEM (GLO-30) © DLR e.V. 2020 and © Airbus Defence and Space GmbH 2020, provided under COPERNICUS by the European Union and ESA; all rights reserved
- Map: Leaflet 1.9.4 © Vladimir Agafonkin and contributors (BSD 2-Clause)
- Aircraft type designators: ICAO Doc 8643 (bundled subset)

The canonical list is `ATTRIBUTION` in `src/triangulate/api/app.py`, served at `/api/config`.
