"""Geoid undulation N (EGM2008) via the PROJ grid us_nga_egm08_25.tif (downloaded on demand)."""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)
_lock = threading.Lock()
_transformer = None
_warned = False


def _get_transformer():
    global _transformer
    with _lock:
        if _transformer is None:
            from pyproj import Transformer, network

            network.set_network_enabled(True)
            _transformer = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
        return _transformer


def geoid_available() -> bool:
    """True if the EGM2008 grid can be used (downloads it on first use)."""
    try:
        n = geoid_N(np.array([40.877]), np.array([-73.526]), warn=False)
        return bool(np.isfinite(n).all() and abs(float(n[0])) > 1e-6)
    except Exception:  # noqa: BLE001
        return False


def geoid_N(lat, lon, warn: bool = True):
    """Geoid height N (m): h_ellipsoidal = h_orthometric + N. Falls back to 0 with a warning
    when the PROJ grid is unavailable. Returns float for scalar input, else ndarray."""
    global _warned
    scalar = np.ndim(lat) == 0 and np.ndim(lon) == 0
    la, lo = np.broadcast_arrays(np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float)))
    try:
        _, _, z = _get_transformer().transform(lo, la, np.zeros_like(la))
        n = -np.asarray(z, float)
        if not np.isfinite(n).all() or np.all(n == 0):
            raise RuntimeError("geoid grid not applied")
    except Exception as e:  # noqa: BLE001
        if warn and not _warned:
            log.warning("EGM2008 geoid grid unavailable (%s); using N = 0 m", e)
            _warned = True
        n = np.zeros_like(la)
    return float(n[0]) if scalar else n
