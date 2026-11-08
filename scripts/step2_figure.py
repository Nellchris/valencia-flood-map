"""STEP 2 feature maps + label statistics figure."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.patches import Patch
import common as c

ext = [c.X0, c.X1, c.Y0, c.Y1]
f = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
wa = gpd.read_file(c.PROC + "/waterways.gpkg", layer="waterways")
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
G = lambda n: c.read(c.PROC + f"/{n}.tif")
dvv, dvh, hand, slope, dist = (G("delta_vv"), G("delta_vh"), G("hand"),
                               G("slope"), G("dist_river"))
V, F, D = lab != 255, lab == 1, lab == 0

fig = plt.figure(figsize=(19, 19))
gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.18)

def panel(i, j, arr, title, cmap, vmin, vmax, lab_=None, outline=True):
    ax = fig.add_subplot(gs[i, j])
    im = ax.imshow(arr, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
    if outline:
        f.boundary.plot(ax=ax, color="red", linewidth=0.35)
    plt.colorbar(im, ax=ax, shrink=0.78, label=lab_ or "")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return ax

panel(0, 0, dvv, "delta_vv = post − pre (dB)\nred = EMSR773 reference outline",
      "RdBu", -12, 12, "ΔVV (dB)")
panel(0, 1, dvh, "delta_vh = post − pre (dB)", "RdBu", -12, 12, "ΔVH (dB)")

ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(np.clip(hand, 0, 60), extent=ext, cmap="viridis_r", vmin=0, vmax=60)
f.boundary.plot(ax=ax, color="red", linewidth=0.35)
plt.colorbar(im, ax=ax, shrink=0.78, label="HAND (m)")
ax.set_title("HAND (pysheds D8, 0.5 km² channel threshold)", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

panel(1, 0, np.clip(slope, 0, 20), "Slope (degrees)", "magma_r", 0, 20, "slope (°)")

ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(dist, extent=ext, cmap="cividis_r", vmin=0, vmax=3000)
wa.plot(ax=ax, color="cyan", linewidth=0.3)
ax.set_xlim(c.X0, c.X1); ax.set_ylim(c.Y0, c.Y1)
plt.colorbar(im, ax=ax, shrink=0.78, label="distance (m)")
ax.set_title("Distance to OSM waterway (cyan = 1,073 lines, 702 km)", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1, 2])
disp = np.full(lab.shape, 0, "uint8")
disp[D] = 1; disp[F] = 2
cm = ListedColormap(["#ffffff", "#9ecae1", "#08306b"])
ax.imshow(disp, extent=ext, cmap=cm, norm=BoundaryNorm([0,1,2,3], 3),
          interpolation="nearest")
ax.legend(handles=[Patch(fc="#08306b", label="1 flooded (17.36%)"),
                   Patch(fc="#9ecae1", label="0 not flooded (82.64%)"),
                   Patch(fc="#ffffff", ec="k", label="255 nodata")],
          loc="upper right", fontsize=9)
ax.set_title("labels.tif — valid domain 610.2 km²", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# histograms
ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(-20, 15, 120)
ax.hist(dvv[D][np.isfinite(dvv[D])], bins=bins, density=True, alpha=.6,
        label="not flooded", color="#9ecae1")
ax.hist(dvv[F][np.isfinite(dvv[F])], bins=bins, density=True, alpha=.6,
        label="flooded", color="#08306b")
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("delta_vv (dB)"); ax.set_ylabel("density"); ax.legend(fontsize=9)
ax.set_title("ΔVV distribution — heavily overlapping\n(medians differ by only 0.44 dB)",
             fontsize=11)

ax = fig.add_subplot(gs[2, 1])
bins = np.linspace(0, 40, 90)
ax.hist(hand[D], bins=bins, density=True, alpha=.6, label="not flooded", color="#9ecae1")
ax.hist(hand[F], bins=bins, density=True, alpha=.6, label="flooded", color="#08306b")
ax.set_xlabel("HAND (m)"); ax.set_ylabel("density"); ax.legend(fontsize=9)
ax.set_title("HAND distribution — strongly separating\n(medians 0.96 m vs 5.79 m)",
             fontsize=11)

# separability by land cover
ax = fig.add_subplot(gs[2, 2])
ks, seps, ns = [], [], []
for k in sorted(np.unique(wc[V])):
    m = wc == k
    fl, dr = F & m, D & m
    if fl.sum() < 500: continue
    ks.append(f"{k} {c.WC_CLASSES[k][:14]}")
    seps.append(np.nanmedian(dvv[fl]) - np.nanmedian(dvv[dr]))
    ns.append(fl.sum())
y = np.arange(len(ks))
ax.barh(y, seps, color=["#d7301f" if s < -2 else "#bdbdbd" for s in seps])
ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=9)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("median ΔVV (flooded) − median ΔVV (dry), dB")
ax.set_title("Radar separability by land cover\nonly wetland & cropland carry signal",
             fontsize=11)
for i, (s_, n_) in enumerate(zip(seps, ns)):
    ax.text(s_ + (-0.4 if s_ < 0 else 0.15), i, f"n={n_:,}", va="center",
            ha="right" if s_ < 0 else "left", fontsize=7.5)
ax.set_xlim(-14, 4)

out = c.PROC + "/fig_step2_features.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print("figure ->", out)
