"""STEP 8 patch: footprint at 10 m tolerance, radar overlays as JPEG.

Two file swaps only. Nothing else in webmap/ is touched.

The footprint is re-derived from flood_rf.tif rather than re-simplifying the
existing 5 m GeoJSON: Douglas-Peucker is not composable, so simplifying an
already-simplified line at 10 m does not give the same result as simplifying
the original at 10 m. Same MMU, same attributes, same code path as Step 8 --
only SIMP_POLY changes.

stats.json is edited in place so its `layers` block still names files that
exist (it feeds the web map's info panels).
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd, shapely
from scipy import ndimage as ndi
from rasterio import features
from shapely.geometry import shape
from PIL import Image
import common as c

MMU_FOOT, NEW_TOL, OLD_TOL, JPEG_Q = 10, 10.0, 5.0, 85
CONF_NAME = {1: "high", 2: "medium", 3: "low"}

with rasterio.open(c.PROC + "/flood_rf.tif") as s: rf = s.read(1)
with rasterio.open(c.PROC + "/confidence.tif") as s: conf = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
prob = c.read(c.PROC + "/flood_prob_rf.tif")
central = (rf == 1) & (wc != 80)

LC_CODES = sorted(c.WC_CLASSES)
_lcmap = np.zeros(256, np.int64)
for i, k in enumerate(LC_CODES):
    _lcmap[k] = i

# ------------------------------------------------- 1. footprint at 10 m
lb, n = ndi.label(central, structure=np.ones((3, 3), int))
sz = np.bincount(lb.ravel()); sz[0] = 0
fm = np.isin(lb, np.flatnonzero(sz >= MMU_FOOT))
lb, n = ndi.label(fm, structure=np.ones((3, 3), int))

lv = lb[fm].astype(np.int64)
pv = np.nan_to_num(prob[fm], nan=0.0).astype(np.float64)
cnt = np.bincount(lv, minlength=n + 1)
psum = np.bincount(lv, weights=pv, minlength=n + 1)
pmax = np.zeros(n + 1); np.maximum.at(pmax, lv, pv)
K = len(LC_CODES)
lcc = np.bincount(lv * K + _lcmap[wc[fm]], minlength=(n + 1) * K).reshape(n + 1, K)
dom_i, dom_n = lcc.argmax(axis=1), lcc.max(axis=1)
cmin = np.full(n + 1, 255, np.int64)
np.minimum.at(cmin, lv, np.where((conf[fm] >= 1) & (conf[fm] <= 3),
                                 conf[fm], 255).astype(np.int64))

geoms, rows = [], []
for g, v in features.shapes(lb.astype("int32"), mask=fm,
                            transform=c.TRANSFORM, connectivity=8):
    i = int(v)
    if i == 0:
        continue
    dom = LC_CODES[dom_i[i]] if cnt[i] else 0
    geoms.append(shape(g))
    rows.append(dict(
        prob_mean=round(float(psum[i] / cnt[i]), 4) if cnt[i] else None,
        prob_max=round(float(pmax[i]), 4) if cnt[i] else None,
        landcover=int(dom), landcover_name=c.WC_CLASSES.get(dom, "n/a"),
        landcover_pct=round(float(dom_n[i] / cnt[i]), 3) if cnt[i] else 0.0,
        area_m2=round(float(cnt[i]) * 100.0, 1),
        conf_best=CONF_NAME.get(int(cmin[i]), "none")))

foot = gpd.GeoDataFrame(rows, geometry=geoms, crs=c.DST_CRS)
raw_v = int(shapely.get_num_coordinates(foot.geometry.values).sum())
p = os.path.join(c.WEBMAP, "flood_footprint.geojson")
old_mb = os.path.getsize(p) / 1e6
foot["geometry"] = foot.geometry.simplify(NEW_TOL, preserve_topology=True)
foot = foot[~foot.geometry.is_empty & foot.geometry.notna()]
foot = foot.to_crs("EPSG:4326")
foot.to_file(p, driver="GeoJSON", coordinate_precision=5)

# Validity must be checked AFTER writing. The geometry is valid in UTM; it is
# the writer's rounding to 5 decimal degrees (~1.1 m) that collapses
# near-coincident vertices into ring self-intersections, which break renderers
# and spatial queries. So: write, read back, repair in WGS84, rewrite.
chk = gpd.read_file(p)
nbad = int((~chk.is_valid).sum())
if nbad:
    chk = chk.set_geometry(gpd.GeoSeries(
        shapely.make_valid(chk.geometry.values), crs="EPSG:4326"))
    chk.to_file(p, driver="GeoJSON", coordinate_precision=5)
    chk = gpd.read_file(p)
    print(f"   repaired {nbad} ring self-intersections introduced by "
          f"coordinate rounding; invalid now {int((~chk.is_valid).sum())}, "
          f"{len(chk):,} features kept")
foot = chk
new_v = int(shapely.get_num_coordinates(foot.geometry.values).sum())
new_mb = os.path.getsize(p) / 1e6
print("1. flood_footprint.geojson re-exported at 10 m tolerance")
print(f"   {len(foot):,} polygons (unchanged), area "
      f"{foot.to_crs(c.DST_CRS).area.sum()/1e6:.2f} km2")
print(f"   vertices {raw_v:,} raw -> {new_v:,} at 10 m")
print(f"   {old_mb:.2f} MB (5 m)  ->  {new_mb:.2f} MB (10 m)   "
      f"{100*(new_mb-old_mb)/old_mb:+.1f}%")

# ------------------------------------------------- 2. radar PNG -> JPEG
print(f"\n2. radar overlays -> JPEG quality {JPEG_Q}, white where alpha was 0")
swapped = {}
for stem in ("pre_radar", "post_radar"):
    src = os.path.join(c.WEBMAP, stem + ".png")
    dst = os.path.join(c.WEBMAP, stem + ".jpg")
    im = Image.open(src)
    a = np.array(im.convert("LA"))
    grey, alpha = a[..., 0], a[..., 1]
    flat = np.where(alpha > 0, grey, 255).astype("uint8")   # white outside swath
    Image.fromarray(flat, mode="L").save(dst, quality=JPEG_Q, optimize=True,
                                         progressive=True)
    omb, nmb = os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6
    print(f"   {stem+'.png':16s} {omb:6.2f} MB  ->  {stem+'.jpg':16s} "
          f"{nmb:6.2f} MB   {100*(nmb-omb)/omb:+.1f}%  "
          f"({100*alpha.size and 100*(alpha == 0).mean():.1f}% of frame was "
          "outside the swath)")
    os.remove(src)
    swapped[stem] = nmb

# ------------------------------------------------- 3. stats.json fixups
sp = os.path.join(c.WEBMAP, "stats.json")
st = json.load(open(sp))
L = st["layers"]
L["flood_footprint.geojson"]["simplify_m"] = NEW_TOL
for stem in ("pre_radar", "post_radar"):
    e = L.pop(stem + ".png")
    e["format"] = "JPEG"
    e["quality"] = JPEG_Q
    e["alpha"] = ("none; nodata outside the Sentinel-1 swath is white "
                  "(255). The swath mask is applied in the web map layer "
                  "code, not in the image.")
    L[stem + ".jpg"] = e
json.dump(st, open(sp, "w"), indent=1)
print(f"\n3. stats.json updated: simplify_m -> {NEW_TOL:.0f}, "
      "pre/post_radar.png -> .jpg (+format, quality, alpha note)")

print("\n" + "=" * 70)
print("webmap/ CONTENTS")
print("=" * 70)
tot = 0
for f in sorted(os.listdir(c.WEBMAP)):
    s = os.path.getsize(os.path.join(c.WEBMAP, f)); tot += s
    print(f"  {f:32s} {s/1e6:9.3f} MB")
print(f"  {'TOTAL':32s} {tot/1e6:9.3f} MB")
