"""STEP 8: web-map export -- GeoJSON (EPSG:4326), PNG overlays, stats, provenance.

Design decisions worth stating, because they change what the web map shows:

* A minimum mapping unit is applied before vectorising. The RF footprint has
  20,742 8-connected components, almost all of them single-pixel speckle; an
  MMU of 10 px (0.1 ha, the usual floor for a 10 m flood product) cuts that to
  2,105 polygons while keeping 98.2% of the area. The confidence layer is far
  more fragmented (37,151 components) so it uses a 5 px MMU and still loses
  7.7% of its area -- which is why it is ALSO shipped as a lossless PNG overlay.
  The MMU is cartographic generalisation only: every metric in Steps 3-7 was
  computed on the full raster and none of them changes.

* `flood_footprint` and `confidence` are deliberately NOT the same layer.
  The footprint is "where the model says flood" (one dissolved extent, the
  central estimate). The confidence layer splits that same extent into the
  HIGH/MEDIUM/LOW bands plus the 0.50-0.70 remainder that carries no confidence
  class at all -- which is 90% of the footprint's buildings, so a web map that
  hides it would badly mislead.

* The radar overlays are PNG, not the GeoTIFF the original spec named. They are
  reprojected to EPSG:4326 first, because an axis-aligned lat/lon image overlay
  of a UTM raster would be visibly rotated (UTM30N is turned ~1.65 deg here).

* Permanent water (WorldCover 80) is removed from every flood layer, as in
  Step 7. It is standing water, not flood.
"""
import os, sys, json, warnings, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd, pandas as pd, shapely
from scipy import ndimage as ndi
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
from PIL import Image
import common as c

os.makedirs(c.WEBMAP, exist_ok=True)
ACCESS = "2026-11-06"
MMU_FOOT, MMU_CONF = 10, 5
SIMP_POLY, SIMP_LINE = 5.0, 2.0
CONF_NAME = {1: "high", 2: "medium", 3: "low"}
CONF_BAND = {1: (0.90, 1.00), 2: (0.80, 0.90), 3: (0.70, 0.80)}

# ----------------------------------------------------------------- inputs
with rasterio.open(c.PROC + "/flood_rf.tif") as s: rf = s.read(1)
with rasterio.open(c.PROC + "/confidence.tif") as s: conf = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
prob = c.read(c.PROC + "/flood_prob_rf.tif")
pw = wc == 80
central = (rf == 1) & ~pw

bld = gpd.read_file(c.PROC + "/buildings.gpkg", layer="buildings")
rds = gpd.read_file(c.PROC + "/roads.gpkg", layer="roads")
ref_parts = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_parts")


def despeckle(mask, mmu):
    """Drop 8-connected components smaller than `mmu` pixels."""
    lb, n = ndi.label(mask, structure=np.ones((3, 3), int))
    sz = np.bincount(lb.ravel()); sz[0] = 0
    keep = np.flatnonzero(sz >= mmu)
    out = np.isin(lb, keep)
    return out, int(mask.sum() - out.sum()), n, len(keep)


LC_CODES = sorted(c.WC_CLASSES)
LC_IDX = {k: i for i, k in enumerate(LC_CODES)}
_lcmap = np.zeros(256, np.int64)
for k, i in LC_IDX.items():
    _lcmap[k] = i


def polygonize_with_stats(mask):
    """Polygonise `mask` and attach zonal stats, one polygon per component.

    Polygonising the *label* array rather than a 0/1 mask means each returned
    shape carries its component id, so the per-polygon statistics come from a
    couple of bincounts over the whole grid instead of one rasterisation per
    polygon (which would be ~10k full-grid passes).
    """
    lb, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return [], []
    lv = lb[mask].astype(np.int64)
    pv = np.nan_to_num(prob[mask], nan=0.0).astype(np.float64)
    cnt = np.bincount(lv, minlength=n + 1)
    psum = np.bincount(lv, weights=pv, minlength=n + 1)
    pmax = np.zeros(n + 1); np.maximum.at(pmax, lv, pv)
    K = len(LC_CODES)
    lcc = np.bincount(lv * K + _lcmap[wc[mask]], minlength=(n + 1) * K
                      ).reshape(n + 1, K)
    dom_i = lcc.argmax(axis=1)
    dom_n = lcc.max(axis=1)
    cmin = np.full(n + 1, 255, np.int64)
    np.minimum.at(cmin, lv, np.where((conf[mask] >= 1) & (conf[mask] <= 3),
                                     conf[mask], 255).astype(np.int64))
    geoms, rows = [], []
    for g, v in features.shapes(lb.astype("int32"), mask=mask,
                                transform=c.TRANSFORM, connectivity=8):
        i = int(v)
        if i == 0:
            continue
        geoms.append(shape(g))
        dom = LC_CODES[dom_i[i]] if cnt[i] else 0
        rows.append(dict(
            prob_mean=round(float(psum[i] / cnt[i]), 4) if cnt[i] else None,
            prob_max=round(float(pmax[i]), 4) if cnt[i] else None,
            landcover=int(dom), landcover_name=c.WC_CLASSES.get(dom, "n/a"),
            landcover_pct=round(float(dom_n[i] / cnt[i]), 3) if cnt[i] else 0.0,
            area_m2=round(float(cnt[i]) * 100.0, 1),
            conf_best=CONF_NAME.get(int(cmin[i]), "none")))
    return geoms, rows


BUDGET_MB = 10.0     # anything heavier is not a web-map vector layer


def save(gdf, name, simp, precision=5):
    """Simplify (metres, in UTM), reproject to WGS84, write GeoJSON.

    Coordinate precision is 5 decimal degrees (~1.1 m): ten times finer than
    the 10 m pixel the geometry came from, so it is lossless in substance and
    noticeably smaller than the default 6.
    """
    gdf = gdf.copy()
    if simp:
        gdf["geometry"] = gdf.geometry.simplify(simp, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf = gdf.to_crs("EPSG:4326")
    p = os.path.join(c.WEBMAP, name)
    gdf.to_file(p, driver="GeoJSON", coordinate_precision=precision)
    mb = os.path.getsize(p) / 1e6
    nv = int(shapely.get_num_coordinates(gdf.geometry.values).sum())
    print(f"  {name:28s} {len(gdf):>7,} feat  {nv:>9,} vertices  {mb:7.2f} MB")
    return mb


def sweep(gdf, label, tols=(0, 5, 10, 20, 40)):
    """Report the size/vertex trade-off, since 5 m is near-inert on a 10 m grid."""
    print(f"  simplification sweep for {label}:")
    for t in tols:
        g = gdf.geometry.simplify(t, preserve_topology=True) if t else gdf.geometry
        nv = int(shapely.get_num_coordinates(g.values).sum())
        print(f"     tol {t:>3d} m -> {nv:>9,} vertices "
              f"({100*nv/shapely.get_num_coordinates(gdf.geometry.values).sum():5.1f}%)")


print("=" * 86)
print("VECTOR EXPORTS")
print("=" * 86)

# --- 1. flood footprint (central estimate) ------------------------------
fm, dropped, n0, n1 = despeckle(central, MMU_FOOT)
print(f"flood footprint: {n0:,} components -> {n1:,} after {MMU_FOOT} px MMU; "
      f"dropped {dropped:,} px ({100*dropped/central.sum():.2f}% of area)")
geoms, rows = polygonize_with_stats(fm)
foot = gpd.GeoDataFrame(rows, geometry=geoms, crs=c.DST_CRS)
sz_foot = save(foot, "flood_footprint.geojson", SIMP_POLY)
sweep(foot, "flood_footprint")

# --- 2. confidence bands ------------------------------------------------
cm, cdrop, cn0, cn1 = despeckle(((conf >= 1) & (conf <= 3)) & ~pw, MMU_CONF)
print(f"confidence: {cn0:,} components -> {cn1:,} after {MMU_CONF} px MMU; "
      f"dropped {cdrop:,} px "
      f"({100*cdrop/(((conf>=1)&(conf<=3))&~pw).sum():.2f}% of area)")
cgeoms, crows = [], []
band_geoms = {}
for k in (1, 2, 3):
    gs, rs = polygonize_with_stats(cm & (conf == k))
    band_geoms[k] = gs
    lo, hi = CONF_BAND[k]
    for g, a in zip(gs, rs):
        a.pop("conf_best", None)
        a.update(conf_code=k, conf_class=CONF_NAME[k], prob_lo=lo, prob_hi=hi)
        crows.append(a); cgeoms.append(g)
# the remainder of the footprint: RF flood but below 0.70, no confidence class
rem_m, rdrop, _, _ = despeckle(fm & (conf == 0), MMU_CONF)
rem_g, rem_r = polygonize_with_stats(rem_m)
band_geoms[0] = rem_g
for g, a in zip(rem_g, rem_r):
    a.pop("conf_best", None)
    a.update(conf_code=0, conf_class="unclassed", prob_lo=0.50, prob_hi=0.70)
    crows.append(a); cgeoms.append(g)
cgdf = gpd.GeoDataFrame(crows, geometry=cgeoms, crs=c.DST_CRS)
sz_conf = save(cgdf, "confidence.geojson", SIMP_POLY)
conf_vector_shipped = sz_conf <= BUDGET_MB
if not conf_vector_shipped:
    os.remove(os.path.join(c.WEBMAP, "confidence.geojson"))
    print(f"  -> {sz_conf:.1f} MB exceeds the {BUDGET_MB:.0f} MB budget; "
          "confidence.geojson DROPPED, served as confidence.png instead "
          "(per the brief's fallback). The PNG is lossless: no MMU, no "
          "simplification, so it also restores the 7.71% of area the "
          "vectorisation would have dropped.")

# --- 3. EMS reference ---------------------------------------------------
# sensor is kept deliberately: it is the evidence for the non-circularity claim
ref = ref_parts[["aoi", "sensor", "det_method", "geometry"]].copy()
ref["area_m2"] = ref.geometry.area.round(1)
sz_ref = save(ref, "flood_reference.geojson", SIMP_POLY)

# --- 4. affected buildings ---------------------------------------------
bx, by = bld.geometry.x.to_numpy(), bld.geometry.y.to_numpy()
bc = np.floor((bx - c.X0) / c.RES).astype(np.int64)
br = np.floor((c.Y1 - by) / c.RES).astype(np.int64)
ok = (br >= 0) & (br < c.HEIGHT) & (bc >= 0) & (bc < c.WIDTH)
hit = np.zeros(len(bld), bool)
hit[ok] = central[br[ok], bc[ok]]
ba = bld[hit].copy()
r2, c2 = br[hit], bc[hit]
ba["conf_class"] = [CONF_NAME.get(int(v), "unclassed") for v in conf[r2, c2]]
ba["prob"] = [round(float(v), 4) for v in prob[r2, c2]]   # float32 -> clean JSON
ba = ba[["building", "conf_class", "prob", "geometry"]].rename(
    columns={"building": "building_type"})
sz_bld = save(ba, "buildings_affected.geojson", 0.0)

# --- 5. affected roads --------------------------------------------------
# Each road is densified to 10 m, every segment is tagged from the raster, and
# runs of consecutive same-class segments are rebuilt into LineStrings. A
# polygon clip (gpd.clip against ~10k band polygons) is the obvious way to do
# this and is far too slow; this is exact to the 10 m grid, which is the
# resolution the footprint was derived at anyway.
seg = shapely.segmentize(rds.geometry.values, 10.0)
co, sidx = shapely.get_coordinates(seg, return_index=True)
pos = np.flatnonzero(sidx[:-1] == sidx[1:])          # first vertex of each seg
x0, y0 = co[pos].T
x1, y1 = co[pos + 1].T
mcol = np.floor(((x0 + x1) / 2 - c.X0) / c.RES).astype(np.int64)
mrow = np.floor((c.Y1 - (y0 + y1) / 2) / c.RES).astype(np.int64)
inb = (mrow >= 0) & (mrow < c.HEIGHT) & (mcol >= 0) & (mcol < c.WIDTH)
keep = np.zeros(len(pos), bool)
keep[inb] = central[mrow[inb], mcol[inb]]
cls = np.full(len(pos), -1, np.int8)
cls[inb] = conf[mrow[inb], mcol[inb]]
line_id = sidx[pos]

k = np.flatnonzero(keep)
brk = np.ones(len(k), bool)                          # start of a new run
brk[1:] = ((line_id[k][1:] != line_id[k][:-1]) |
           (cls[k][1:] != cls[k][:-1]) |
           (pos[k][1:] != pos[k][:-1] + 1))
starts = np.flatnonzero(brk)
ends = np.r_[starts[1:], len(k)] - 1
NMK = {1: "high", 2: "medium", 3: "low", 0: "unclassed"}
geo, hw = [], rds.highway.to_numpy()
for a, b in zip(starts, ends):
    geo.append(shapely.LineString(co[pos[k[a]]:pos[k[b]] + 2]))
raw = gpd.GeoDataFrame({"osm_row": line_id[k[starts]],
                        "conf_class": [NMK.get(int(v), "unclassed")
                                       for v in cls[k[starts]]]},
                       geometry=geo, crs=c.DST_CRS)
raw["highway"] = hw[raw.osm_row.to_numpy()]
print(f"roads inside footprint: {raw.geometry.length.sum()/1000:,.1f} km "
      f"in {len(raw):,} runs")
print(raw.groupby("conf_class").geometry.apply(
    lambda s: round(s.length.sum() / 1000, 1)).to_string())
# One feature per (road, confidence class). A GeoJSON Feature costs ~200 bytes
# of boilerplate; at 94k runs that boilerplate, not the geometry, is three
# quarters of the file. Dissolving to MultiLineString loses nothing.
ra = raw.dissolve(by=["osm_row", "conf_class"], aggfunc="first").reset_index()
ra["length_m"] = ra.geometry.length.round(1)
ra = ra[ra.length_m > 0][["highway", "length_m", "conf_class", "geometry"]]
print(f"  dissolved to {len(ra):,} features "
      f"({ra.length_m.sum()/1000:,.1f} km preserved)")
sz_rd = save(ra, "roads_affected.geojson", SIMP_LINE)

# ----------------------------------------------------------------- PNGs
print("\n" + "=" * 86)
print("RASTER OVERLAYS (reprojected to EPSG:4326)")
print("=" * 86)
MAXW = 4000


def to_wgs(arr, nodata, resamp):
    """Reproject a grid array to EPSG:4326, capped at MAXW px wide."""
    tr, w, h = calculate_default_transform(
        c.DST_CRS, "EPSG:4326", c.WIDTH, c.HEIGHT, *c.BOUNDS)
    if w > MAXW:
        sc = MAXW / w
        w, h = int(w * sc), int(h * sc)
        tr = rasterio.Affine(tr.a / sc, tr.b, tr.c, tr.d, tr.e / sc, tr.f)
    dst = np.full((h, w), nodata, arr.dtype)
    reproject(arr, dst, src_transform=c.TRANSFORM, src_crs=c.DST_CRS,
              dst_transform=tr, dst_crs="EPSG:4326", src_nodata=nodata,
              dst_nodata=nodata, resampling=resamp)
    b = rasterio.transform.array_bounds(h, w, tr)
    return dst, (b[0], b[1], b[2], b[3]), w, h


def radar_png(src_tif, name):
    a = c.read(src_tif)
    g, bnds, w, h = to_wgs(np.where(np.isfinite(a), a, np.nan).astype("float32"),
                           np.nan, Resampling.average)
    v = g[np.isfinite(g)]
    lo, hi = np.percentile(v, [2, 98])
    s = np.clip((g - lo) / (hi - lo), 0, 1)
    # mode "LA" (grey + alpha), not RGBA: the three colour channels of a
    # greyscale image are identical, so RGBA triples the payload for nothing.
    la = np.zeros((h, w, 2), "uint8")
    la[..., 0] = (np.nan_to_num(s) * 255).astype("uint8")
    la[..., 1] = np.where(np.isfinite(g), 255, 0)
    p = os.path.join(c.WEBMAP, name)
    Image.fromarray(la, mode="LA").save(p, optimize=True)
    print(f"  {name:28s} {w} x {h}  stretch {lo:.2f}..{hi:.2f} dB  "
          f"{os.path.getsize(p)/1e6:7.2f} MB")
    return bnds, (float(lo), float(hi)), os.path.getsize(p) / 1e6


bnds_pre, st_pre, _ = radar_png(c.PROC + "/pre_vv.tif", "pre_radar.png")
bnds_post, st_post, _ = radar_png(c.PROC + "/post_vv.tif", "post_radar.png")

# confidence PNG -- lossless companion to the (despeckled) GeoJSON
cviz = np.where(pw, 255, conf).astype("uint8")
g, bnds_c, w, h = to_wgs(cviz, 255, Resampling.nearest)
PAL = {1: (0, 68, 27), 2: (253, 174, 97), 3: (215, 48, 39), 0: (0, 0, 0)}
rgba = np.zeros((h, w, 4), "uint8")
for k, col in PAL.items():
    m = g == k
    rgba[m, :3] = col
    rgba[m, 3] = 200 if k else 0
p = os.path.join(c.WEBMAP, "confidence.png")
Image.fromarray(rgba).save(p, optimize=True)
print(f"  {'confidence.png':28s} {w} x {h}  "
      f"{os.path.getsize(p)/1e6:7.2f} MB  (lossless, no MMU applied)")

# ----------------------------------------------------------------- stats
val = pd.read_csv(c.PROC + "/validation_results.csv")
cvm = json.load(open(c.PROC + "/_cv_meta.json"))
exp = pd.read_csv(c.PROC + "/exposure_results.csv")
expr = pd.read_csv(c.PROC + "/exposure_vs_reference.csv")
cs = pd.read_csv(c.PROC + "/confidence_summary.csv")
clc = pd.read_csv(c.PROC + "/confidence_by_landcover.csv")
dec = pd.read_csv(c.PROC + "/_decile_errors.csv")
rfm = json.load(open(c.PROC + "/_rf_meta.json"))
thr = np.load(c.PROC + "/_conf_thr.npy")
otsu_thr = float(np.load(c.PROC + "/_otsu_thr.npy")[0])
prov_npy = np.load(c.PROC + "/_prov.npy", allow_pickle=True)

sensor_area = {str(r[1]): float(r[3]) for r in prov_npy}
tot_sa = sum(sensor_area.values())

stats = {
    "generated": ACCESS,
    "event": {"name": "Valencia flash flood (DANA)",
              "date": "2024-10-29",
              "ems_activation": "EMSR773"},
    "study_area": {"bbox_wgs84": list(c.BBOX_WGS), "crs": c.DST_CRS,
                   "resolution_m": c.RES,
                   "grid": [c.WIDTH, c.HEIGHT],
                   "area_km2": round(c.WIDTH * c.HEIGHT * 100 / 1e6, 2)},
    "imagery": {
        "pre": {"scene": c.S1_PRE, "acquired": "2024-10-20T06:02:07Z"},
        "post": {"scene": c.S1_POST, "acquired": "2024-11-01T06:02:07Z"},
        "lag_after_peak_hours": 19.7,
        "note": "The post-event pass is ~3 days after the 29 Oct peak and "
                "19.7 h after the flood crest in the reference imagery; much "
                "of the flash flood had already drained."},
    "model": {"type": "RandomForestClassifier", "n_estimators": 200,
              "max_depth": 15, "class_weight": "balanced",
              "features": cvm["features"],
              "importance_random_cv": dict(zip(cvm["features"],
                                               [round(v, 4) for v in cvm["imp_random_mean"]])),
              "importance_spatial_cv_2km": dict(zip(cvm["features"],
                                                    [round(v, 4) for v in cvm["imp_spatial2_mean"]])),
              "training_pixels": rfm.get("n_train", 300000)},
    "baseline_otsu": {"threshold_db": round(otsu_thr, 3),
                      "literature_manual_threshold_db": -3.0},
    "validation": {
        "headline": {k: round(v, 4) for k, v in cvm["headline"].items()},
        "f1_optimism_absolute": round(cvm["headline"]["random_f1"] -
                                      cvm["headline"]["spatial2_f1"], 4),
        "f1_optimism_percent": round(100 * (cvm["headline"]["random_f1"] -
                                            cvm["headline"]["spatial2_f1"]) /
                                     cvm["headline"]["random_f1"], 1),
        "by_method": [
            {"method": m,
             "block_size_km": None if pd.isna(g.block_size_km.iloc[0])
             else float(g.block_size_km.iloc[0]),
             "mean": {k: round(float(g[k].mean()), 4)
                      for k in ("iou", "f1", "precision", "recall")},
             "sd": {k: round(float(g[k].std()), 4)
                    for k in ("iou", "f1", "precision", "recall")},
             "folds": [{"fold": int(r.fold), "iou": round(float(r.iou), 4),
                        "f1": round(float(r.f1), 4),
                        "precision": round(float(r.precision), 4),
                        "recall": round(float(r.recall), 4),
                        "test_pos_rate": round(float(r.test_pos_rate), 4)}
                       for r in g.itertuples()]}
            for m, g in val.groupby("method", sort=False)],
        "protocol": {"folds": 5, "buffer_m": 500,
                     "block_sizes_km": [1, 2, 4],
                     "note": "Random and spatial CV differ only in fold "
                             "assignment; training budget, model settings and "
                             "test-fold evaluation are identical."}},
    "confidence": {
        "thresholds": {"low": float(thr[0]), "medium": float(thr[1]),
                       "high": float(thr[2])},
        "threshold_basis": "out-of-fold (spatial 2 km) decile error rates; "
                           "replaces the originally specified 0.3/0.5/0.8",
        "error_rates": [
            {"set": r.set, "class": r.confidence, "n_px": int(r.n),
             "area_km2": round(float(r.area_km2), 2),
             "true_flood_rate": round(float(r.true_flood_rate), 4),
             "error_rate": round(float(r.error_rate), 4)}
            for r in cs.itertuples()],
        "by_landcover": [
            {"landcover": r.landcover, "class": r.confidence, "n_px": int(r.n),
             "error_rate": round(float(r.error_rate), 4)}
            for r in clc.itertuples()],
        "calibration_deciles": [
            {"set": r.set, "lo": float(r.lo), "hi": float(r.hi),
             "n_px": int(r.n), "flood_rate": round(float(r.flood_rate), 4)}
            for r in dec.itertuples()]},
    "exposure": {
        "osm_totals": {"buildings": int(len(bld)),
                       "road_km": round(float(rds.geometry.length.sum() / 1000), 1)},
        "estimates": [
            {"estimate": r.estimate, "definition": r.definition,
             "area_km2": float(r.area_km2), "buildings": int(r.n_buildings),
             "road_km": float(r.road_km),
             "pct_of_all_buildings": float(r.pct_of_all_buildings),
             "pct_of_all_road_km": float(r.pct_of_all_road_km)}
            for r in exp.itertuples()],
        "vs_reference": [
            {"estimate": r.estimate,
             "buildings": {"tp": int(r.bld_tp), "fp": int(r.bld_fp),
                           "fn": int(r.bld_fn), "precision": float(r.bld_precision),
                           "recall": float(r.bld_recall), "f1": float(r.bld_f1)},
             "roads": {"tp_km": float(r.road_tp_km), "fp_km": float(r.road_fp_km),
                       "precision": float(r.road_precision),
                       "recall": float(r.road_recall), "f1": float(r.road_f1)}}
            for r in expr.itertuples()],
        "caveats": [
            "'upper' is identical to 'central': the data-driven LOW band "
            "(p>=0.70) lies inside the p>0.5 footprint, so the union adds "
            "nothing.",
            "90% of the central estimate's buildings (21,351 of 23,732) fall "
            "in the 0.50-0.70 probability band, which has ~20% out-of-fold "
            "flood rate against a 15.4% base rate -- i.e. no skill.",
            "HIGH confidence is effectively unreachable on built-up land "
            "(4 of 63,353 HIGH pixels), so the lower bound holds 9 buildings.",
            "3.0% of reference buildings and 7.5% of reference road km lie "
            "outside the Sentinel-1 swath."]},
    "reference": {
        "activation": "EMSR773", "aois": ["AOI01", "AOI02", "AOI03"],
        "note": "AOI02 lies entirely outside the study box.",
        "area_km2_in_grid": 126.98,
        "sensor_area_km2": {k: round(v, 2) for k, v in sensor_area.items()},
        "sentinel1_share_pct": round(100 * sensor_area.get("Sentinel-1", 0)
                                     / tot_sa, 2),
        "circularity": "The delineation this model is validated against is "
                       "overwhelmingly optical (Sentinel-2, Landsat-8, "
                       f"GeoEye-1, WorldView-2). Only "
                       f"{100*sensor_area.get('Sentinel-1',0)/tot_sa:.2f}% of "
                       "the reference area was derived from Sentinel-1, so "
                       "validating a Sentinel-1 model against it is not "
                       "circular."},
    "layers": {
        "flood_footprint.geojson": {"type": "polygon", "crs": "EPSG:4326",
                                    "mmu_px": MMU_FOOT, "simplify_m": SIMP_POLY,
                                    "represents": "central estimate (RF p>0.5)"},
        **({"confidence.geojson": {
            "type": "polygon", "crs": "EPSG:4326", "mmu_px": MMU_CONF,
            "simplify_m": SIMP_POLY,
            "classes": ["high", "medium", "low", "unclassed"]}}
           if conf_vector_shipped else {}),
        "confidence.png": {"type": "image_overlay", "crs": "EPSG:4326",
                           "bounds_wgs84": [round(v, 6) for v in bnds_c],
                           "classes": {"high": "#00441b", "medium": "#fdae61",
                                       "low": "#d73027"},
                           "note": "lossless: no MMU and no simplification. "
                                   "Replaces confidence.geojson, which was "
                                   f"{sz_conf:.1f} MB."
                                   if not conf_vector_shipped else "lossless"},
        "flood_reference.geojson": {"type": "polygon", "crs": "EPSG:4326"},
        "buildings_affected.geojson": {"type": "point", "crs": "EPSG:4326",
                                       "note": "OSM building centroids"},
        "roads_affected.geojson": {"type": "line", "crs": "EPSG:4326",
                                   "simplify_m": SIMP_LINE},
        "pre_radar.png": {"type": "image_overlay", "crs": "EPSG:4326",
                          "bounds_wgs84": [round(v, 6) for v in bnds_pre],
                          "stretch_db": st_pre, "band": "VV gamma0 dB"},
        "post_radar.png": {"type": "image_overlay", "crs": "EPSG:4326",
                           "bounds_wgs84": [round(v, 6) for v in bnds_post],
                           "stretch_db": st_post, "band": "VV gamma0 dB"}},
}
sp = os.path.join(c.WEBMAP, "stats.json")
json.dump(stats, open(sp, "w"), indent=1)
print(f"\nstats.json  {os.path.getsize(sp)/1e3:.1f} kB")

# ------------------------------------------------------------- provenance
OSM_LIC = "Open Database License (ODbL) 1.0"
rows = [
    dict(name="Sentinel-1A RTC, pre-event", provider="ASF HyP3 (GAMMA RTC) / "
         "Copernicus Sentinel-1", sensor="Sentinel-1A C-band SAR (IW, VV+VH)",
         acquired="2024-10-20T06:02:07Z", accessed=ACCESS,
         licence="Copernicus Sentinel data 2024; CC-BY 4.0 (HyP3 products)",
         url="https://search.asf.alaska.edu/",
         identifier=c.S1_PRE, role="pre-event backscatter"),
    dict(name="Sentinel-1A RTC, post-event", provider="ASF HyP3 (GAMMA RTC) / "
         "Copernicus Sentinel-1", sensor="Sentinel-1A C-band SAR (IW, VV+VH)",
         acquired="2024-11-01T06:02:07Z", accessed=ACCESS,
         licence="Copernicus Sentinel data 2024; CC-BY 4.0 (HyP3 products)",
         url="https://search.asf.alaska.edu/",
         identifier=c.S1_POST, role="post-event backscatter"),
    dict(name="Copernicus DEM GLO-30", provider="ESA / Copernicus",
         sensor="TanDEM-X (bistatic InSAR)", acquired="2011-2015",
         accessed=ACCESS,
         licence="Copernicus DEM free, full and open licence",
         url="https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model",
         identifier="GLO-30", role="slope, HAND"),
    dict(name="ESA WorldCover 10 m v200", provider="ESA WorldCover consortium",
         sensor="Sentinel-1 + Sentinel-2 (2021 composite)", acquired="2021",
         accessed=ACCESS, licence="CC-BY 4.0",
         url="https://esa-worldcover.org/", identifier="v200",
         role="land-cover feature; permanent-water mask"),
    dict(name="OpenStreetMap waterways", provider="OpenStreetMap contributors",
         sensor="n/a (crowd-sourced vector)", acquired=ACCESS, accessed=ACCESS,
         licence=OSM_LIC, url="https://overpass-api.de/api/interpreter",
         identifier='way[waterway~"river|stream|canal|drain"]',
         role="dist_river feature"),
    dict(name="OpenStreetMap buildings", provider="OpenStreetMap contributors",
         sensor="n/a (crowd-sourced vector)", acquired=ACCESS, accessed=ACCESS,
         licence=OSM_LIC, url="https://overpass-api.de/api/interpreter",
         identifier="way/relation[building], out center",
         role="exposure: building counts"),
    dict(name="OpenStreetMap roads", provider="OpenStreetMap contributors",
         sensor="n/a (crowd-sourced vector)", acquired=ACCESS, accessed=ACCESS,
         licence=OSM_LIC, url="https://overpass-api.de/api/interpreter",
         identifier="way[highway] (drivable classes only)",
         role="exposure: road length"),
]
for aoi, sens, nparts, area in prov_npy:
    rows.append(dict(
        name=f"Copernicus EMS EMSR773 observed flood, {aoi} ({sens})",
        provider="Copernicus Emergency Management Service, Rapid Mapping",
        sensor=sens, acquired="2024-10-30 / 2024-11-02", accessed=ACCESS,
        licence="Copernicus EMS data policy (free, full and open)",
        url="https://rapidmapping.emergency.copernicus.eu/EMSR773/",
        identifier=f"{aoi} observedEventA, dmg_src_id -> {sens}, "
                   f"{int(nparts)} parts, {float(area):.2f} km2",
        role="validation reference (ground truth)"))
pv = pd.DataFrame(rows)[["name", "provider", "sensor", "acquired", "accessed",
                         "licence", "url", "identifier", "role"]]
pp = os.path.join(c.BASE, "provenance.csv")
pv.to_csv(pp, index=False)
print(f"provenance.csv  {len(pv)} sources -> {pp}")

print("\n" + "=" * 86)
print("webmap/ CONTENTS")
print("=" * 86)
tot = 0
for f in sorted(os.listdir(c.WEBMAP)):
    s = os.path.getsize(os.path.join(c.WEBMAP, f)); tot += s
    print(f"  {f:32s} {s/1e6:9.3f} MB")
print(f"  {'TOTAL':32s} {tot/1e6:9.3f} MB")
print("done")
