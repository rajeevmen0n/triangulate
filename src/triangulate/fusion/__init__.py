"""Fusion (WP12): common multi-resolution grid, log posterior, candidates, exports.

Submodules (import them directly; this package imports nothing so `fusion.grid` stays light):

- `grid`       cell ids (level, ix, iy), packing, coordinates, tiles; `LogField`, `Layer` (stable API)
- `combine`    the layer contract, `fuse` -> `Posterior`, layer builders (acoustic/clue/GPS)
- `candidates` NMS candidates (`FusionCandidate`), HPD regions, late `rerank`
- `export`     `run_fusion` -> `FusionResult`, GeoTIFF / GeoJSON / JSON / Web-Mercator outputs

Read the docstring of `combine` for how a module submits a layer.
"""
