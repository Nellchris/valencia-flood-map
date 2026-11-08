"""Swath masks for the web map's radar overlays.

The radar overlays are JPEG, which has no alpha, so everything outside the
Sentinel-1 swath is white. Two ways to get that transparent in the browser both
fail:

  * MapLibre has no `raster-color` / `raster-color-mix` paint property at any
    version (it is a Mapbox GL JS feature), so the brief's approach is out;
  * thresholding on brightness in the browser would punch holes in the image,
    because bright urban double-bounce saturates at the top of the 2-98%
    stretch and is indistinguishable from the white border by value alone.

So the mask is derived geometrically here, from the same nodata the dB rasters
carry, reprojected on exactly the grid step8_export.py used for the JPEGs. A
two-value image compresses to almost nothing, so this costs a few tens of kB
and keeps the 2.3 MB JPEGs.
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
import common as c

MAXW = 4000          # identical cap to step8_export.to_wgs
OUT = os.path.join(c.BASE, "site", "webmap")


def wgs_grid():
    tr, w, h = calculate_default_transform(
        c.DST_CRS, "EPSG:4326", c.WIDTH, c.HEIGHT, *c.BOUNDS)
    if w > MAXW:
        sc = MAXW / w
        w, h = int(w * sc), int(h * sc)
        tr = rasterio.Affine(tr.a / sc, tr.b, tr.c, tr.d, tr.e / sc, tr.f)
    return tr, w, h


tr, W, H = wgs_grid()
print(f"target grid {W} x {H} (EPSG:4326), matching the exported JPEGs")

for stem, src in (("pre_radar", "pre_vv.tif"), ("post_radar", "post_vv.tif")):
    a = c.read(os.path.join(c.PROC, src))
    valid = np.isfinite(a).astype("uint8")          # 1 inside swath, 0 outside
    dst = np.zeros((H, W), "uint8")
    reproject(valid, dst, src_transform=c.TRANSFORM, src_crs=c.DST_CRS,
              dst_transform=tr, dst_crs="EPSG:4326",
              src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
    im = Image.fromarray((dst * 255).astype("uint8"), mode="L").convert("1")
    p = os.path.join(OUT, stem + "_mask.png")
    im.save(p, optimize=True, bits=1)
    print(f"  {stem+'_mask.png':22s} {os.path.getsize(p)/1e3:7.1f} kB   "
          f"inside swath {100*dst.mean():.1f}%")

# record the masks in stats.json so the layer code can find them
sp = os.path.join(OUT, "stats.json")
st = json.load(open(sp))
for stem in ("pre_radar", "post_radar"):
    st["layers"][stem + ".jpg"]["alpha"] = (
        f"none in the JPEG; {stem}_mask.png is a 1-bit swath mask on the same "
        "grid (255 = inside the Sentinel-1 swath). The web map composites the "
        "two onto a canvas to get transparency.")
    st["layers"][stem + "_mask.png"] = {
        "type": "alpha_mask", "crs": "EPSG:4326",
        "bounds_wgs84": st["layers"][stem + ".jpg"]["bounds_wgs84"],
        "for": stem + ".jpg",
        "note": "1-bit; 255 = valid radar, 0 = outside swath"}
json.dump(st, open(sp, "w"), indent=1)
print(f"\nstats.json updated ({os.path.getsize(sp)/1e3:.1f} kB)")
