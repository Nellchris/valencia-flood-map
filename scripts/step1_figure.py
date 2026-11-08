"""STEP 1 figure + coverage diagnostics."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import common as c

dem = c.read(os.path.join(c.PROC, "dem.tif"))
with rasterio.open(os.path.join(c.PROC, "landcover.tif")) as s: wc = s.read(1)
with rasterio.open(os.path.join(c.PROC, "pre_vv.tif")) as s:
    s1 = s.read(1) != c.NODATA_F
flood = gpd.read_file(os.path.join(c.PROC, "flood_reference.gpkg"), layer="flood_reference")
cov = gpd.read_file(os.path.join(c.PROC, "flood_reference.gpkg"), layer="reference_coverage")
ext = [c.X0, c.X1, c.Y0, c.Y1]

sea = wc == 80
print("DIAGNOSTICS")
print(f"  grid area                     : {c.WIDTH*c.HEIGHT*100/1e6:8.2f} km2")
print(f"  WorldCover class 80 (water)   : {sea.sum()*100/1e6:8.2f} km2  {100*sea.mean():5.1f}%")
print(f"  S1 valid                      : {s1.sum()*100/1e6:8.2f} km2  {100*s1.mean():5.1f}%")
print(f"  S1 invalid                    : {(~s1).sum()*100/1e6:8.2f} km2  {100*(~s1).mean():5.1f}%")
print(f"  S1 invalid AND sea            : {(~s1 & sea).sum()*100/1e6:8.2f} km2")
print(f"  S1 invalid AND land           : {(~s1 & ~sea).sum()*100/1e6:8.2f} km2")
print(f"  land (not class 80)           : {(~sea).sum()*100/1e6:8.2f} km2")
print(f"  ANALYSABLE land with S1       : {(s1 & ~sea).sum()*100/1e6:8.2f} km2  "
      f"{100*(s1&~sea).mean():5.1f}% of grid")
# where is S1 missing? column / row profile
colfrac = (~s1).mean(axis=0); rowfrac = (~s1).mean(axis=1)
miss_cols = np.where(colfrac > 0.5)[0]
miss_rows = np.where(rowfrac > 0.5)[0]
if miss_cols.size:
    print(f"  cols >50% missing             : {miss_cols.min()}..{miss_cols.max()} "
          f"(x {c.X0+miss_cols.min()*10:.0f}..{c.X0+miss_cols.max()*10:.0f})")
if miss_rows.size:
    print(f"  rows >50% missing             : {miss_rows.min()}..{miss_rows.max()} "
          f"(y {c.Y1-miss_rows.max()*10:.0f}..{c.Y1-miss_rows.min()*10:.0f})")
ref_in_s1 = gpd.clip(flood, gpd.GeoDataFrame(geometry=[__import__("shapely").geometry.box(*c.BOUNDS)], crs=c.DST_CRS))
print(f"  reference flood area in grid  : {flood.geometry.area.sum()/1e6:8.2f} km2")

fig, axes = plt.subplots(2, 2, figsize=(16, 15))

ax = axes[0, 0]
hs = np.where(np.isfinite(dem), dem, np.nan)
im = ax.imshow(hs, extent=ext, cmap="terrain", vmin=-20, vmax=340)
flood.boundary.plot(ax=ax, color="red", linewidth=0.45)
cov.boundary.plot(ax=ax, color="blue", linewidth=1.4, linestyle="--")
ax.plot([c.X0, c.X1, c.X1, c.X0, c.X0], [c.Y0, c.Y0, c.Y1, c.Y1, c.Y0],
        color="k", lw=2)
plt.colorbar(im, ax=ax, shrink=0.75, label="elevation (m)")
ax.set_title("DEM + EMSR773 flood reference (red) + reference coverage (blue)\n"
             "black = study grid (EPSG:32630, 10 m)")
ax.legend(handles=[Patch(ec="red", fc="none", label="flood reference"),
                   Patch(ec="blue", fc="none", ls="--", label="EMS AOI coverage")],
          loc="upper right", fontsize=8)

ax = axes[0, 1]
with rasterio.open(os.path.join(c.PROC, "post_vv.tif")) as s: pv = s.read(1)
pv = np.where(pv == c.NODATA_F, np.nan, pv)
im = ax.imshow(pv, extent=ext, cmap="gray", vmin=-25, vmax=5)
flood.boundary.plot(ax=ax, color="red", linewidth=0.4)
plt.colorbar(im, ax=ax, shrink=0.75, label="VV $\gamma^0$ (dB)")
ax.set_title("Post-event VV backscatter (1 Nov 2024) + reference outline")

ax = axes[1, 0]
codes = sorted(c.WC_CLASSES)
cmap = ListedColormap(["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000",
                       "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#00cf75",
                       "#fae6a0"])
norm = BoundaryNorm([0] + [k + 1 for k in codes], cmap.N)
ax.imshow(wc, extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
flood.boundary.plot(ax=ax, color="red", linewidth=0.4)
present = [k for k in codes if (wc == k).any()]
ax.legend(handles=[Patch(fc=cmap(norm(k)), label=f"{k} {c.WC_CLASSES[k]}")
                   for k in present], loc="upper right", fontsize=7)
ax.set_title("ESA WorldCover 10 m 2021")

ax = axes[1, 1]
diag = np.zeros(s1.shape, "uint8")
diag[s1 & ~sea] = 1
diag[sea] = 2
diag[~s1 & ~sea] = 3
diag[~s1 & sea] = 4
cm2 = ListedColormap(["#cccccc", "#2ca25f", "#0064c8", "#de2d26", "#756bb1"])
ax.imshow(diag, extent=ext, cmap=cm2, norm=BoundaryNorm([0,1,2,3,4,5], 5),
          interpolation="nearest")
flood.boundary.plot(ax=ax, color="k", linewidth=0.4)
ax.legend(handles=[Patch(fc="#2ca25f", label="land + S1 (analysable)"),
                   Patch(fc="#0064c8", label="WC water (sea/lagoon), S1 ok"),
                   Patch(fc="#de2d26", label="land, S1 MISSING"),
                   Patch(fc="#756bb1", label="water, S1 missing")],
          loc="upper right", fontsize=8)
ax.set_title("Coverage diagnostic: S1 footprint vs permanent water")

for a in axes.ravel():
    a.set_xlabel("Easting (m, EPSG:32630)"); a.set_ylabel("Northing (m)")
plt.tight_layout()
out = os.path.join(c.PROC, "fig_step1_overlay.png")
plt.savefig(out, dpi=115, bbox_inches="tight")
print("\nfigure ->", out)
