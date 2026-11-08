"""STEP 2c: distance to river + labels raster."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import common as c

# ------------------------------------------------- distance to nearest waterway
wa = gpd.read_file(c.PROC + "/waterways.gpkg", layer="waterways")
riv = rasterize([(g, 1) for g in wa.geometry], out_shape=(c.HEIGHT, c.WIDTH),
                transform=c.TRANSFORM, fill=0, dtype="uint8",
                all_touched=True).astype(bool)
print(f"OSM waterways: {len(wa)} lines, {wa.geometry.length.sum()/1000:.1f} km, "
      f"{riv.sum():,} px on grid")
dist = distance_transform_edt(~riv, sampling=c.RES).astype("float32")
c.write(c.PROC + "/dist_river.tif", dist, "float32", c.NODATA_F)
print(f"  dist_river.tif  min {dist.min():.0f}  max {dist.max():.0f}  "
      f"mean {dist.mean():.0f}  median {np.median(dist):.0f} m")

# ------------------------------------------------------------------ labels
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
with rasterio.open(c.PROC + "/post_vv.tif") as s: s1 = s.read(1) != c.NODATA_F
with rasterio.open(c.PROC + "/hand.tif") as s: hand_ok = s.read(1) != c.NODATA_F
f = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
cov = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="reference_coverage")

R = lambda g: rasterize([(x, 1) for x in g.geometry], out_shape=(c.HEIGHT, c.WIDTH),
                        transform=c.TRANSFORM, fill=0, dtype="uint8").astype(bool)
flood, covr = R(f), R(cov)
permwater = wc == 80

# Valid label domain: inside EMS reference coverage AND valid SAR AND not
# permanent water. Everything else -> nodata (255), NOT 0.
valid = covr & s1 & ~permwater
labels = np.full((c.HEIGHT, c.WIDTH), 255, dtype="uint8")
labels[valid & flood] = 1
labels[valid & ~flood] = 0
c.write(c.PROC + "/labels.tif", labels, "uint8", 255)

px = 100 / 1e6
tot = c.HEIGHT * c.WIDTH
print("\n" + "=" * 70)
print("LABEL RASTER  (1 = flooded, 0 = not flooded, 255 = nodata)")
print("=" * 70)
print("Exclusion cascade:")
print(f"  grid                                  {tot:10,} px  {tot*px:8.2f} km2")
print(f"  - outside EMS reference coverage      {(~covr).sum():10,} px  "
      f"{(~covr).sum()*px:8.2f} km2")
print(f"  - no valid SAR (swath gap)            {(covr&~s1).sum():10,} px  "
      f"{(covr&~s1).sum()*px:8.2f} km2")
print(f"  - WorldCover 80 permanent water       {(covr&s1&permwater).sum():10,} px  "
      f"{(covr&s1&permwater).sum()*px:8.2f} km2")
print(f"  = VALID LABEL DOMAIN                  {valid.sum():10,} px  "
      f"{valid.sum()*px:8.2f} km2  ({100*valid.mean():.1f}% of grid)")
nf, nd = int((labels == 1).sum()), int((labels == 0).sum())
print(f"\n  flooded      (1) {nf:10,} px  {nf*px:8.2f} km2   {100*nf/valid.sum():6.2f}%")
print(f"  not flooded  (0) {nd:10,} px  {nd*px:8.2f} km2   {100*nd/valid.sum():6.2f}%")
print(f"  nodata     (255) {int((labels==255).sum()):10,} px")
print(f"  class ratio dry:flooded = {nd/nf:.2f} : 1")
print(f"\n  reference flood lost to swath gap     "
      f"{(flood&covr&~s1&~permwater).sum()*px:8.2f} km2")
print(f"  reference flood on permanent water    {(flood&permwater).sum()*px:8.2f} km2")

# feature completeness over the label domain
print("\nFEATURE COMPLETENESS over valid label domain:")
for nm in ("delta_vv", "delta_vh", "slope", "hand", "dist_river"):
    with rasterio.open(f"{c.PROC}/{nm}.tif") as s: a = s.read(1)
    ok = a != c.NODATA_F
    miss = (valid & ~ok).sum()
    print(f"  {nm:11s} valid in domain {100*(valid&ok).sum()/valid.sum():6.2f}%"
          f"   missing {miss:,} px")

# separability preview
dvv = c.read(c.PROC + "/delta_vv.tif")
for nm, arr in (("delta_vv", dvv), ("delta_vh", c.read(c.PROC+"/delta_vh.tif")),
                ("hand", c.read(c.PROC+"/hand.tif")),
                ("slope", c.read(c.PROC+"/slope.tif")),
                ("dist_river", dist)):
    a = arr[valid & (labels == 1) & np.isfinite(arr)]
    b = arr[valid & (labels == 0) & np.isfinite(arr)]
    print(f"  {nm:11s} flooded median {np.median(a):8.2f} | "
          f"dry median {np.median(b):8.2f}  -> sep {np.median(a)-np.median(b):+8.2f}")
