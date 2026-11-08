"""Re-render confidence.png in the accessible palette, and fold provenance
into stats.json so the web map's Methods drawer has a single source.

The original PNG came out of Step 6 in a dark-green / amber / red ramp, which
is a red-green pair -- exactly what the web map's accessibility rule rules out.
The teal / amber / red-orange ramp used here keeps the same ordering while
staying distinguishable under deuteranopia and protanopia.

Geometry is untouched: same reprojection grid, same dimensions, same bounds.
"""
import os, sys, csv, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
import common as c

MAXW = 4000
ALPHA = 200
PALETTE = {1: ("high", "#0d9488", (13, 148, 136)),
           2: ("medium", "#f59e0b", (245, 158, 11)),
           3: ("low", "#ea580c", (234, 88, 12))}
TARGETS = [os.path.join(c.BASE, "webmap"), os.path.join(c.BASE, "site", "webmap")]

with rasterio.open(c.PROC + "/confidence.tif") as s:
    conf = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s:
    wc = s.read(1)

# identical grid to step8_export.to_wgs
tr, W, H = calculate_default_transform(
    c.DST_CRS, "EPSG:4326", c.WIDTH, c.HEIGHT, *c.BOUNDS)
if W > MAXW:
    sc = MAXW / W
    W, H = int(W * sc), int(H * sc)
    tr = rasterio.Affine(tr.a / sc, tr.b, tr.c, tr.d, tr.e / sc, tr.f)

src = np.where(wc == 80, 255, conf).astype("uint8")      # drop permanent water
dst = np.full((H, W), 255, "uint8")
reproject(src, dst, src_transform=c.TRANSFORM, src_crs=c.DST_CRS,
          dst_transform=tr, dst_crs="EPSG:4326", src_nodata=255, dst_nodata=255,
          resampling=Resampling.nearest)

rgba = np.zeros((H, W, 4), "uint8")
counts = {}
for v, (name, hexv, rgb) in PALETTE.items():
    m = dst == v
    rgba[m, :3] = rgb
    rgba[m, 3] = ALPHA
    counts[name] = int(m.sum())
# class 0 (below 0.70) and 255 (nodata) stay fully transparent

bounds = list(rasterio.transform.array_bounds(H, W, tr))
print(f"grid {W} x {H}, bounds {[round(b, 6) for b in bounds]}")
for k, n in counts.items():
    print(f"  {k:7s} {n:>9,} px  {PALETTE[[a for a in PALETTE if PALETTE[a][0]==k][0]][1]}")
print(f"  transparent {int((rgba[...,3]==0).sum()):,} px")

img = Image.fromarray(rgba, mode="RGBA")
for d in TARGETS:
    p = os.path.join(d, "confidence.png")
    old = os.path.getsize(p) if os.path.exists(p) else 0
    img.save(p, optimize=True)
    print(f"-> {p}   {old/1e3:.1f} kB -> {os.path.getsize(p)/1e3:.1f} kB")

# ------------------------------------------------- stats.json: classes + provenance
prov = []
with open(os.path.join(c.BASE, "provenance.csv"), newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        prov.append({k: r[k] for k in
                     ("name", "provider", "sensor", "acquired", "accessed",
                      "licence", "url", "role")})

for d in TARGETS:
    sp = os.path.join(d, "stats.json")
    st = json.load(open(sp, encoding="utf-8"))
    st["layers"]["confidence.png"]["classes"] = {
        name: hexv for _, (name, hexv, _) in sorted(PALETTE.items())}
    st["layers"]["confidence.png"]["note"] = (
        "lossless: no MMU and no simplification. Re-rendered in a "
        "colour-vision-safe teal/amber/red-orange ramp; the earlier "
        "dark-green/red ramp was a red-green pair. Below 0.70 probability is "
        "transparent (no confidence class).")
    st["provenance"] = prov
    json.dump(st, open(sp, "w", encoding="utf-8"), indent=1)
    print(f"-> {sp}   classes updated, {len(prov)} provenance rows")
