# Vendored assets

No CDN is used at runtime. Everything under `vendor/` was downloaded once.

| Asset | Version | Source | Licence |
|---|---|---|---|
| Leaflet (`vendor/leaflet/leaflet.js`, `leaflet.css`, `images/`) | 1.9.4 | https://unpkg.com/leaflet@1.9.4/dist/ | BSD-2-Clause (`vendor/leaflet/LICENSE`) |

sha256:

- `vendor/leaflet/leaflet.js`  db49d009c841f5ca34a888c96511ae936fd9f5533e90d8b2c4d57596f4e5641a
- `vendor/leaflet/leaflet.css` a7837102824184820dfa198d1ebcd109ff6d0ff9a2672a074b9a1b4d147d04c6
- `vendor/leaflet/LICENSE`     53e8dc25862014e4324741ca18fbe3611e11d42ef69f59f86ea8c5389647d4cb

The only runtime network request the page itself can make to a third party is the optional
basemap tile URL from `/api/config` (`basemap_url`), and only when `basemap_enabled` is true.
