import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from triangulate.dem.copernicus import CopernicusDem, tile_name, tile_url
from triangulate.dem.geoid import geoid_N


def _make_tile(path, lat0=40, lon0=-74, n=100):
    # elevation = 100*(lat-lat0) + 10*(lon-lon0)... planar ramp, exactly bilinear-reproducible
    res = 1.0 / n
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    lat_c = lat0 + 1 - (rows + 0.5) * res
    lon_c = lon0 + (cols + 0.5) * res
    data = (100 * (lat_c - lat0) + 10 * (lon_c - lon0)).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(lon0, lat0 + 1, res, res)) as dst:
        dst.write(data, 1)


def test_tile_naming():
    assert tile_name(40.877, -73.526) == "Copernicus_DSM_COG_10_N40_00_W074_00_DEM"
    assert tile_name(-0.5, 0.5, 90) == "Copernicus_DSM_COG_30_S01_00_E000_00_DEM"
    assert "copernicus-dem-90m" in tile_url(tile_name(1, 1, 90), 90)


def test_synthetic_tile_bilinear_and_missing(tmp_path):
    name = tile_name(40.5, -73.5)
    _make_tile(tmp_path / f"{name}.tif")
    dem = CopernicusDem(30, cache_dir=tmp_path, allow_network=False)
    lat = np.array([40.5, 40.123, 40.9])
    lon = np.array([-73.5, -73.77, -73.2])
    e = dem.elevation(lat, lon)
    np.testing.assert_allclose(e, 100 * (lat - 40) + 10 * (lon + 74), atol=1e-3)
    # tile absent (ocean / offline) -> 0 m, scalar input works
    assert dem.elevation(10.5, 10.5).tolist() == [0.0]
    # mixed tiles
    mixed = dem.elevation(np.array([40.5, 10.5]), np.array([-73.5, 10.5]))
    assert mixed[1] == 0.0 and mixed[0] > 0


def test_ocean_marker(tmp_path):
    dem = CopernicusDem(30, cache_dir=tmp_path, allow_network=True)
    (tmp_path / f"{tile_name(0.5, -30.5)}.missing").write_text("")
    assert dem.elevation(0.5, -30.5)[0] == 0.0


@pytest.mark.network
def test_oyster_bay_elevation_and_geoid():
    from triangulate.dem.provider import get_dem

    e = get_dem().elevation(np.array([40.8767]), np.array([-73.5264]))[0]
    print("elevation", e)
    assert -5 <= e <= 40
    assert get_dem().elevation(np.array([40.0]), np.array([-71.0]))[0] == 0.0  # open ocean
    n = geoid_N(40.8767, -73.5264)
    if n == 0.0:
        pytest.skip("PROJ EGM2008 grid unavailable (warning logged)")
    assert -35 <= n <= -28
