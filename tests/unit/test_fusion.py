"""WP12 fusion: grid ids, multi-resolution layers, posterior, candidates, HPD, exports."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from triangulate.core.types import GridSpec, ModuleResult
from triangulate.fusion.candidates import compute_hpd, find_candidates, pinned_candidate, rerank
from triangulate.fusion.combine import (
    acoustic_layer,
    domain_area_m2,
    evidence_at,
    fuse,
    gps_mixture_layer,
    kernel_layer,
    membership_loglik,
    polygon_layer,
)
from triangulate.fusion.export import run_fusion, web_mercator_raster, write_outputs
from triangulate.fusion.grid import (
    TILE,
    Layer,
    LogField,
    cell_centers_latlon,
    cell_centers_xy,
    latlon_to_cells,
    latlon_to_xy,
    level_for_size,
    logmeanexp_downsample,
    pack_cell,
    pack_cells,
    tile_of,
    to_level,
    unpack_cell,
    unpack_cells,
    xy_to_cells,
)

SMALL_BBOX = (40.75, -73.70, 41.05, -73.30)  # ~33 x 34 km
MID_BBOX = (40.60, -73.80, 41.00, -73.27)  # ~44 x 45 km


def small_spec(max_cells=40_000, bbox=SMALL_BBOX) -> GridSpec:
    return GridSpec.for_bbox(bbox, max_dense_cells=max_cells)


def mass_at_level(post, level):
    a = post.at_level(level)
    s = GridSpec.cell_size(level)
    return float(np.exp(logsumexp(a[np.isfinite(a)]) + 2 * math.log(s)))


# --- part 1: cell ids ---------------------------------------------------------------------------


def test_pack_unpack_and_levels():
    lv = np.array([0, 3, 6, 6])
    ix = np.array([0, -5, 123456, -(2**27)])
    iy = np.array([7, 9, -1, 2**27])
    keys = pack_cells(lv, ix, iy)
    assert keys.dtype == np.int64 and (keys > 0).all() and len(set(keys.tolist())) == 4
    for a, b in zip(unpack_cells(keys), (lv, ix, iy), strict=True):
        np.testing.assert_array_equal(a, b)
    assert unpack_cell(pack_cell((5, -3, 4))) == (5, -3, 4)
    assert level_for_size(200) == 3 and level_for_size(25) == 6
    with pytest.raises(ValueError):
        level_for_size(300)
    # parents of negative cells floor correctly; children are the SW corner
    px, py = to_level(6, np.array([-1, 7, 8]), np.array([0, -9, 15]), 3)
    np.testing.assert_array_equal(px, [-1, 0, 1])
    np.testing.assert_array_equal(py, [0, -2, 1])
    cx, cy = to_level(3, np.array([1]), np.array([-2]), 5)
    assert (cx[0], cy[0]) == (4, -8)
    tx, ty = tile_of(np.array([-1, 255, 256]), np.array([0, 0, 513]))
    np.testing.assert_array_equal(tx, [-1, 0, 1])
    np.testing.assert_array_equal(ty, [0, 0, 2])


def test_coordinates_roundtrip():
    spec = small_spec()
    lat, lon = 40.876722, -73.526444
    for lv in range(7):
        ix, iy = latlon_to_cells(spec, lat, lon, lv)
        clat, clon = cell_centers_latlon(spec, lv, ix, iy)
        x, y = latlon_to_xy(spec, lat, lon)
        cx, cy = cell_centers_xy(spec, lv, ix, iy)
        s = GridSpec.cell_size(lv)
        assert abs(cx - x) <= s / 2 and abs(cy - y) <= s / 2
        assert xy_to_cells(spec, cx, cy, lv) == (ix, iy)
        assert abs(clat - lat) < s / 1e5 and abs(clon - lon) < s / 7e4
        # nesting: the 25 m cell's ancestor is this cell
        fx, fy = latlon_to_cells(spec, lat, lon, 6)
        assert to_level(6, fx, fy, lv) == (ix, iy)


# --- part 2: fields -----------------------------------------------------------------------------


def test_logfield_render_nan_and_aggregation():
    spec = small_spec()
    f = LogField(spec, fill=-1.0)
    f.add_patch(1, 2, 3, np.array([[1.0, np.nan], [3.0, 4.0]]))  # 800 m cells (2..3, 3..4)
    # level-2 render: each level-1 cell -> 2x2 children; NaN shows the fill
    r = f.render(2, 4, 6, 4, 4)
    np.testing.assert_array_equal(r[:2, :2], 1.0)
    np.testing.assert_array_equal(r[:2, 2:], -1.0)
    np.testing.assert_array_equal(r[2:, :2], 3.0)
    # a finer patch overrides and aggregates with log-mean-exp
    f.add_patch(2, 4, 6, np.array([[5.0, np.nan], [np.nan, np.nan]]))
    agg = f.render(1, 2, 3, 2, 2)
    assert agg[0, 0] == pytest.approx(math.log((math.exp(5) + 3 * math.exp(1)) / 4))
    assert agg[1, 1] == 4.0 and agg[0, 1] == -1.0
    # point queries agree with renders
    x, y = cell_centers_xy(spec, 2, np.arange(4, 8), np.full(4, 6))
    np.testing.assert_array_equal(f.value_at_xy(x, y), f.render(2, 4, 6, 4, 1)[0])
    # LME conserves mass
    a = np.random.default_rng(0).normal(size=(8, 8))
    assert logsumexp(logmeanexp_downsample(a, 4)) + math.log(16) == pytest.approx(logsumexp(a))


def test_add_cells_reduces_duplicates():
    spec = small_spec()
    lay = Layer.from_cells(spec, 6, [10, 10, 300], [5, 5, 5], [1.0, 2.0, 7.0], "a", "acoustic", fill=-3)
    assert len(lay.patches) == 2  # two different 256-tiles
    v = lay.value_at_cells(6, np.array([10, 300, 11]), np.array([5, 5, 5]))
    np.testing.assert_allclose(v, [np.logaddexp(1, 2), 7.0, -3.0])
    mx = Layer.from_cells(spec, 6, [10, 10], [5, 5], [1.0, 2.0], "a", "acoustic", reduce="max")
    assert mx.value_at_cells(6, 10, 5) == 2.0


# --- posterior: acceptance criteria -------------------------------------------------------------


def _coarse_and_fine(spec, seed=1):
    rng = np.random.default_rng(seed)
    nx0, ny0 = spec.shape(0)
    coarse = Layer.from_dense(spec, rng.normal(0, 2, (ny0, nx0)), "coarse", "visual", level=0)
    b = spec.base_level
    nxb, nyb = spec.shape(b)
    dense = Layer.from_dense(spec, rng.normal(0, 1, (nyb, nxb)), "dense", "metadata")
    fine = Layer(spec=spec, name="fine", module="acoustic", fill=-2.3)
    # two clusters of 25 m cells, one straddling a tile boundary, plus a 50 m block
    for cx, cy in ((TILE * 2 - 20, 300), (900, 700)):
        ii, jj = np.meshgrid(np.arange(cx, cx + 60), np.arange(cy, cy + 40))
        fine.add_cells(6, ii.ravel(), jj.ravel(), rng.normal(3, 2, ii.size))
    fine.add_patch(5, 100, 100, rng.normal(1, 1, (30, 30)))
    return coarse, dense, fine


def test_mass_conserved_across_levels_and_no_seams():
    spec = small_spec()
    coarse, dense, fine = _coarse_and_fine(spec)
    post = fuse(spec, [coarse, dense, fine], domain=None)
    assert post.fine_level == 6 and post.base_level == 3 and post.tiles
    assert post.total_mass() == pytest.approx(1.0, abs=1e-6)
    for lv in range(post.base_level + 1):
        assert mass_at_level(post, lv) == pytest.approx(1.0, abs=1e-6)

    # Brute force: render every layer at 25 m over the base extent -> same normaliser.
    w = post.weights
    nxb, nyb = spec.shape(post.base_level)
    k = 1 << (6 - post.base_level)
    nxf, nyf = nxb * k, nyb * k
    tot = (w["visual"] * coarse.render(6, 0, 0, nxf, nyf) + w["metadata"] * dense.render(6, 0, 0, nxf, nyf)
           + w["acoustic"] * fine.render(6, 0, 0, nxf, nyf))
    assert float(logsumexp(tot) + 2 * math.log(25)) == pytest.approx(post.log_norm, abs=1e-9)

    # Point values equal the weighted layer sum everywhere (base leaves and tile leaves).
    rng = np.random.default_rng(5)
    x0, y0 = spec.x0, spec.y0
    xs = x0 + rng.uniform(0, nxb * 200, 4000)
    ys = y0 + rng.uniform(0, nyb * 200, 4000)
    expect = (w["visual"] * coarse.value_at_xy(xs, ys) + w["metadata"] * dense.value_at_xy(xs, ys)
              + w["acoustic"] * fine.value_at_xy(xs, ys) - post.log_norm)
    np.testing.assert_allclose(post.value_at_xy(xs, ys), expect, atol=1e-9)

    # No seams: a flat fine layer forces tiles but changes nothing anywhere.
    p1 = fuse(spec, [coarse, dense], domain=None)
    flat_fine = Layer(spec=spec, name="flat", module="acoustic", fill=0.0)
    ii, jj = np.meshgrid(np.arange(TILE - 30, TILE + 30), np.arange(TILE - 30, TILE + 30))
    flat_fine.add_cells(6, ii.ravel(), jj.ravel(), np.zeros(ii.size))
    p2 = fuse(spec, [coarse, dense, flat_fine], domain=None)
    assert p2.tiles and not p1.tiles
    np.testing.assert_allclose(p2.value_at_xy(xs, ys), p1.value_at_xy(xs, ys), atol=1e-9)
    np.testing.assert_allclose(p2.base, p1.base, atol=1e-9)
    tile = next(iter(p2.tiles.values()))
    cx, cy = cell_centers_xy(spec, 6, *np.meshgrid(np.arange(tile.ix0, tile.ix0 + tile.nx),
                                                   np.arange(tile.iy0, tile.iy0 + tile.ny)))
    np.testing.assert_allclose(tile.values, p1.value_at_xy(cx, cy), atol=1e-9)


def test_domain_mask_and_bbox():
    spec = small_spec()
    post = fuse(spec, [], domain=SMALL_BBOX)
    assert post.total_mass() == pytest.approx(1.0, abs=1e-9)
    inside = post.value_at_latlon(40.9, -73.5)
    assert np.isfinite(inside) and np.isneginf(post.value_at_latlon(40.70, -73.5))
    # uniform: density = 1 / area of the domain
    assert math.exp(inside) * domain_area_m2(spec, SMALL_BBOX) == pytest.approx(1.0, rel=0.01)


def test_missing_module_equals_flat_layer(tmp_path):
    spec = small_spec()
    coarse, dense, fine = _coarse_and_fine(spec)
    ref = fuse(spec, [coarse, fine], domain=SMALL_BBOX)
    with_flat = fuse(spec, [coarse, fine, Layer.flat(spec, "nothing", "metadata")], domain=SMALL_BBOX)
    np.testing.assert_allclose(with_flat.base, ref.base, atol=1e-12)
    # skipped / failed modules are ignored even when they list layers
    dense_ref = dense.to_ref(tmp_path / "layers" / "dense.npz", relative_to=tmp_path)
    results = [ModuleResult(module="metadata", status="skipped", layers=[dense_ref]),
               ModuleResult(module="visual", status="failed", message="boom")]
    r = run_fusion(spec, layers=[coarse, fine], module_results=results, root=tmp_path, domain=SMALL_BBOX)
    np.testing.assert_allclose(r.posterior.base, ref.base, atol=1e-12)
    assert r.module_status["metadata"] == "skipped"
    # weight 0 also equals absent
    z = fuse(spec, [coarse, fine, dense], domain=SMALL_BBOX, weights={"metadata": 0.0})
    np.testing.assert_allclose(z.base, ref.base, atol=1e-12)


def _gauss_mixture_layer(spec, modes):
    b = spec.base_level
    nxb, nyb = spec.shape(b)
    gi, gj = np.meshgrid(np.arange(nxb), np.arange(nyb))
    x, y = cell_centers_xy(spec, b, gi, gj)
    terms = [math.log(w / (2 * math.pi * s**2)) - ((x - mx) ** 2 + (y - my) ** 2) / (2 * s**2)
             for w, mx, my, s in modes]
    return Layer.from_dense(spec, np.logaddexp.reduce(terms, axis=0), "mix", "visual")


def test_bimodal_candidates_and_hpd_areas():
    spec = small_spec(250_000, MID_BBOX)
    assert GridSpec.cell_size(spec.base_level) == 100
    cx, cy = latlon_to_xy(spec, 40.80, -73.535)
    sigma = 1000.0
    modes = [(0.65, float(cx) - 6000, float(cy) + 1000, sigma),
             (0.35, float(cx) + 6000, float(cy) - 2000, sigma)]
    post = fuse(spec, [_gauss_mixture_layer(spec, modes)], domain=MID_BBOX, weights={"visual": 1.0})
    cands = find_candidates(post, k=10, min_prob_mass=1e-6)
    assert len(cands) == 2
    for c, (_, mx, my, _) in zip(cands, modes, strict=True):
        assert math.hypot(c.x_m - mx, c.y_m - my) <= 100
    assert cands[0].rank == 1 and cands[0].prob_mass > cands[1].prob_mass
    exp_disc = [w * (1 - math.exp(-(300**2) / (2 * sigma**2))) for w, *_ in modes]
    assert cands[0].prob_mass == pytest.approx(exp_disc[0], rel=0.05)
    assert cands[0].margin_to_runner_up == pytest.approx(math.log(exp_disc[0] / exp_disc[1]), abs=0.05)
    # Analytic HPD for equal-sigma Gaussians: t solves sum(w_i - 2 pi s^2 t) = alpha.
    for region in compute_hpd(post, (0.5, 0.9)):
        t = (1.0 - region.mass) / (2 * 2 * math.pi * sigma**2)
        area = sum(math.pi * 2 * sigma**2 * math.log(w / (2 * math.pi * sigma**2 * t)) for w, *_ in modes)
        assert region.area_m2 == pytest.approx(area, rel=0.05)
        assert region.geometry_xy.area == pytest.approx(area, rel=0.05)
        assert region.geometry_xy.geom_type == "MultiPolygon" and len(region.geometry_xy.geoms) == 2
    # density ranking gives the same order here
    assert [c.rank for c in find_candidates(post, k=2, rank_by="density")] == [1, 2]


def _strong_acoustic(spec, lat, lon, peak=15.0, radius=300.0):
    """Strong acoustic blob: 25 m cells, ell = peak - (r/100 m)^2 within 300 m, floor elsewhere."""
    x, y = latlon_to_xy(spec, lat, lon)
    ix0, iy0 = xy_to_cells(spec, float(x) - radius, float(y) - radius, 6)
    n = int(2 * radius / 25) + 2
    gi, gj = np.meshgrid(np.arange(ix0, ix0 + n), np.arange(iy0, iy0 + n))
    cx, cy = cell_centers_xy(spec, 6, gi, gj)
    r2 = (cx - x) ** 2 + (cy - y) ** 2
    m = r2 <= radius**2
    lam = 0.5
    z_sum = (peak - r2[m] / 1e4) / lam
    t_s = 1788900000.0
    lay = acoustic_layer(spec, 6, gi[m], gj[m], z_sum, lam=lam, p_miss=0.1,
                         attrs={"t_s": np.full(m.sum(), t_s),
                                "flights": [["a1b2c3", "d4e5f6"]] * int(m.sum())})
    return lay


def test_gps_mixture_alone_has_alpha_mass():
    spec = small_spec(250_000, MID_BBOX)
    gps = gps_mixture_layer(spec, 40.70, -73.40, accuracy_m=10, domain=MID_BBOX, alpha=0.6)
    assert gps.finest_level == 6 and gps.meta["sigma_m"] == 30.0
    post = fuse(spec, [gps], domain=MID_BBOX)
    top = find_candidates(post, k=1)[0]
    assert math.hypot(*(np.array(latlon_to_xy(spec, 40.70, -73.40)) - [top.x_m, top.y_m])) < 30
    area = domain_area_m2(spec, MID_BBOX)
    assert top.prob_mass == pytest.approx(0.6 + 0.4 * math.pi * 300**2 / area, rel=0.01)


def test_gps_does_not_overwhelm_strong_acoustic():
    spec = small_spec(250_000, MID_BBOX)
    truth = (40.876722, -73.526444)
    gps_pt = (40.70, -73.40)  # ~20 km away
    ac = _strong_acoustic(spec, *truth)
    gps = gps_mixture_layer(spec, *gps_pt, accuracy_m=5, domain=MID_BBOX)
    alone = fuse(spec, [ac], domain=MID_BBOX)
    assert find_candidates(alone, k=1)[0].prob_mass > 0.95
    r = run_fusion(spec, layers=[ac, gps], domain=MID_BBOX, k=5)
    top = r.candidates[0]
    tx, ty = latlon_to_xy(spec, *truth)
    assert math.hypot(top.x_m - tx, top.y_m - ty) < 50
    assert top.prob_mass > 0.9
    assert top.flights == ["a1b2c3", "d4e5f6"] and top.t_s is not None and top.t_s.year == 2026
    assert top.contributions["metadata"] == pytest.approx(math.log(0.4), abs=1e-6)
    assert top.contributions["acoustic"] > 14
    gx, gy = latlon_to_xy(spec, *gps_pt)
    gps_c = [c for c in r.candidates if math.hypot(c.x_m - gx, c.y_m - gy) < 100]
    assert gps_c and gps_c[0].rank > 1 and gps_c[0].contributions["acoustic"] == pytest.approx(math.log(0.1))
    assert top.margin_to_runner_up > 2


def test_evidence_breakdown_matches_posterior():
    spec = small_spec()
    coarse, dense, fine = _coarse_and_fine(spec)
    post = fuse(spec, [coarse, dense, fine], domain=SMALL_BBOX)
    c = find_candidates(post, k=3)
    for cand in c:
        ev = evidence_at(post, cand.x_m, cand.y_m)
        assert sum(ev["contributions"].values()) - post.log_norm == pytest.approx(cand.log_density, abs=1e-9)
        assert set(cand.contributions) == {"visual", "metadata", "acoustic"}
        assert 0 < cand.hpd_level <= 1


def test_rerank_and_pinned():
    spec = small_spec(250_000, MID_BBOX)
    cx, cy = latlon_to_xy(spec, 40.80, -73.535)
    modes = [(0.65, float(cx) - 6000, float(cy), 1000.0), (0.35, float(cx) + 6000, float(cy), 1000.0)]
    post = fuse(spec, [_gauss_mixture_layer(spec, modes)], domain=MID_BBOX)
    cands = find_candidates(post, k=2)
    total = sum(c.prob_mass for c in cands)
    rr = rerank(cands, "sun", [-5.0, 0.0])
    assert [c.rank for c in rr] == [1, 2]
    assert rr[0].x_m == cands[1].x_m and rr[0].contributions["sun"] == 0.0
    assert rr[1].late["sun"] == -5.0
    assert sum(c.prob_mass for c in rr) == pytest.approx(total)
    rr2 = rerank(cands, "sun", [None, None])
    assert [c.x_m for c in rr2] == [c.x_m for c in cands] and "sun: not available" in rr2[0].notes
    pin = pinned_candidate(post, 40.9, -73.4, "trusted GPS")
    out = rerank([pin, *cands], "sun", [0.0, 0.0, 0.0])
    assert out[0].pinned and out[0].rank == 0 and out[1].rank == 1


def test_clue_layer_helpers():
    spec = small_spec()
    assert np.isfinite(membership_loglik(0.0, 1.0)) and membership_loglik(1.0, 0.7) == pytest.approx(0.0)
    assert membership_loglik(0.0, 0.7) == pytest.approx(math.log(0.3))
    k = kernel_layer(spec, 40.9, -73.5, sigma_m=150, confidence=0.8, name="bayview")
    assert k.finest_level == 5  # 50 m cells (<= sigma / 2) for sigma 150 m
    assert float(k.value_at_latlon(40.9, -73.5)) > -0.05  # point on a cell corner: 35 m from centre
    assert float(k.value_at_latlon(40.8, -73.5)) == pytest.approx(math.log(0.2))
    from shapely.geometry import box

    poly = polygon_layer(spec, box(-73.6, 40.85, -73.45, 40.95), confidence=0.9, name="admin",
                         soft_edge_m=3000)
    assert float(poly.value_at_latlon(40.9, -73.5)) == pytest.approx(0.0, abs=1e-6)
    near = float(poly.value_at_latlon(40.9, -73.6 - 0.006))  # ~500 m outside
    assert math.log(0.1) < near < 0
    assert float(poly.value_at_latlon(40.80, -73.35)) == pytest.approx(math.log(0.1), abs=1e-3)


def test_layer_roundtrip_and_results(tmp_path):
    spec = small_spec()
    ac = _strong_acoustic(spec, 40.9, -73.5)
    ref = ac.to_ref(tmp_path / "layers" / "acoustic.npz", relative_to=tmp_path)
    assert ref.path == "layers/acoustic.npz" and ref.level == 6 and ref.module == "acoustic"
    back = Layer.load(tmp_path / ref.path)
    assert back.fill == ac.fill and back.name == ac.name and back.attrs.equals(ac.attrs)
    x, y = latlon_to_xy(spec, 40.9, -73.5)
    assert back.value_at_xy(x, y) == ac.value_at_xy(x, y)
    ref.weight = 0.5
    mr = ModuleResult(module="acoustic", layers=[ref],
                      provenance={"attribution": "Flight data (c) adsb.lol (ODbL)"})
    r = run_fusion(spec, module_results=[mr], root=tmp_path, domain=SMALL_BBOX)
    assert r.posterior.layers[0][1] == 0.5
    assert r.attribution == ["Flight data (c) adsb.lol (ODbL)"]


def test_exports(tmp_path):
    import rasterio

    spec = small_spec()
    truth = (40.876722, -73.526444)
    r = run_fusion(spec, layers=[_strong_acoustic(spec, *truth)], domain=SMALL_BBOX, k=3,
                   pinned=[(40.95, -73.4, "trusted GPS")])
    arts = write_outputs(r, tmp_path)
    assert set(arts) == {"posterior_tif", "posterior_npz", "candidates_geojson", "fusion_json"}
    with rasterio.open(tmp_path / arts["posterior_tif"]) as ds:
        assert ds.count == 2 and ds.crs is not None and "Lambert" in ds.crs.to_wkt()
        b1, b2 = ds.read(1), ds.read(2)
        j, i = np.unravel_index(np.nanargmax(b1), b1.shape)
        x, y = ds.xy(j, i)
        assert np.nanmin(b2) >= 0 and np.nanmax(b2) <= 1 + 1e-6
    tx, ty = latlon_to_xy(spec, *truth)
    assert math.hypot(x - tx, y - ty) < 300
    gj = json.loads((tmp_path / arts["candidates_geojson"]).read_text())
    kinds = [f["properties"]["kind"] for f in gj["features"]]
    assert kinds.count("hpd") == 2 and kinds.count("candidate") == len(r.candidates)
    first = gj["features"][0]
    assert first["properties"]["rank"] == 0 and first["properties"]["notes"] == ["trusted GPS"]
    lon, lat = gj["features"][1]["geometry"]["coordinates"]
    assert abs(lat - truth[0]) < 0.001 and abs(lon - truth[1]) < 0.001
    summary = json.loads((tmp_path / arts["fusion_json"]).read_text())
    assert summary["candidates"][1]["flights"] == ["a1b2c3", "d4e5f6"]
    assert summary["hpd"][0]["mass"] == 0.5 and summary["grid"]["fine_level"] == 6
    from triangulate.fusion.combine import Posterior

    back = Posterior.load(tmp_path / arts["posterior_npz"])
    np.testing.assert_array_equal(back.base, r.posterior.base)
    assert back.total_mass() == pytest.approx(1.0, abs=1e-6)
    merc = web_mercator_raster(r.posterior, width_px=256)
    a = merc["log_density"]
    assert a.shape[1] == 256 and np.isfinite(a).any()
    s, w, n, e = merc["bounds"]
    jj, ii = np.unravel_index(np.argmax(a), a.shape)
    assert s < truth[0] < n and w < truth[1] < e
    assert abs(w + (ii + 0.5) / 256 * (e - w) - truth[1]) < 0.003
