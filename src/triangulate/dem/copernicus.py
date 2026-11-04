"""Copernicus GLO-30 / GLO-90 COG tiles, named by floor(lat)/floor(lon).

Strategy per tile: use the local cache copy if present; else a windowed read through /vsicurl/;
if that fails, download the whole tile into data/cache/dem/. A missing tile (open ocean, HTTP 404)
yields 0 m.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import httpx
import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)

_BASE = {
    30: ("copernicus-dem-30m", "10"),
    90: ("copernicus-dem-90m", "30"),
}


def tile_name(lat: float, lon: float, resolution_m: int = 30) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    code = _BASE[resolution_m][1]
    return (f"Copernicus_DSM_COG_{code}_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM")


def tile_url(name: str, resolution_m: int = 30) -> str:
    return f"https://{_BASE[resolution_m][0]}.s3.amazonaws.com/{name}/{name}.tif"


class CopernicusDem:
    def __init__(self, resolution_m: int = 30, cache_dir: Path | None = None, allow_network: bool = True):
        if resolution_m not in _BASE:
            raise ValueError("resolution_m must be 30 or 90")
        self.resolution_m = resolution_m
        self.cache_dir = Path(cache_dir) if cache_dir else get_settings().cache_subdir("dem")
        self.allow_network = allow_network
        self._lock = threading.Lock()

    # -- tile access -------------------------------------------------------------------------
    def local_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.tif"

    def _missing_marker(self, name: str) -> Path:
        return self.cache_dir / f"{name}.missing"

    def _download(self, name: str) -> Path | None:
        dest = self.local_path(name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with httpx.stream("GET", tile_url(name, self.resolution_m), timeout=60, follow_redirects=True,
                          headers={"User-Agent": get_settings().user_agent}) as r:
            if r.status_code in (403, 404):
                self._missing_marker(name).write_text("")
                return None
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
        return dest

    def _sample(self, src, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Bilinear sample (pixel centres) from an open rasterio dataset via one windowed read."""
        from rasterio.windows import Window

        T = src.transform
        col = (lon - T.c) / T.a - 0.5
        row = (lat - T.f) / T.e - 0.5
        c0 = int(max(math.floor(col.min()), 0))
        r0 = int(max(math.floor(row.min()), 0))
        c1 = int(min(math.floor(col.max()) + 2, src.width))
        r1 = int(min(math.floor(row.max()) + 2, src.height))
        arr = src.read(1, window=Window(c0, r0, max(c1 - c0, 1), max(r1 - r0, 1)), masked=True)
        data = np.ma.filled(arr.astype("float64"), 0.0)
        h, w = data.shape
        cc, rr = col - c0, row - r0
        ci = np.clip(np.floor(cc).astype(int), 0, w - 1)
        ri = np.clip(np.floor(rr).astype(int), 0, h - 1)
        c2, r2 = np.minimum(ci + 1, w - 1), np.minimum(ri + 1, h - 1)
        fc, fr = np.clip(cc - ci, 0, 1), np.clip(rr - ri, 0, 1)
        return (data[ri, ci] * (1 - fc) * (1 - fr) + data[ri, c2] * fc * (1 - fr)
                + data[r2, ci] * (1 - fc) * fr + data[r2, c2] * fc * fr)

    def _tile_values(self, name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        import rasterio

        with self._lock:
            local = self.local_path(name)
            if not local.exists() and self._missing_marker(name).exists():
                return np.zeros_like(lat)
            if not local.exists():
                if not self.allow_network:
                    return np.zeros_like(lat)
                try:  # 1) windowed read over HTTP
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="2"):
                        with rasterio.open("/vsicurl/" + tile_url(name, self.resolution_m)) as src:
                            return self._sample(src, lat, lon)
                except Exception as e:  # noqa: BLE001
                    log.info("vsicurl read of %s failed (%s); downloading whole tile", name, e)
                if self._download(name) is None:
                    return np.zeros_like(lat)  # open ocean
            with rasterio.open(local) as src:
                return self._sample(src, lat, lon)

    # -- public ------------------------------------------------------------------------------
    def elevation(self, lat, lon) -> np.ndarray:
        """Orthometric elevation (m, EGM2008) at the given points; 0 where no tile exists."""
        la, lo = np.broadcast_arrays(np.atleast_1d(np.asarray(lat, float)),
                                     np.atleast_1d(np.asarray(lon, float)))
        shape = la.shape
        la, lo = la.ravel(), lo.ravel()
        out = np.zeros(la.shape, float)
        keys = np.stack([np.floor(la), np.floor(lo)], axis=1)
        for k in np.unique(keys, axis=0):
            m = (keys[:, 0] == k[0]) & (keys[:, 1] == k[1])
            name = tile_name(k[0], k[1], self.resolution_m)
            out[m] = self._tile_values(name, la[m], lo[m])
        return out.reshape(shape)
