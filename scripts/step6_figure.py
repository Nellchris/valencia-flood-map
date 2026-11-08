"""STEP 6 figure: confidence map + calibration + error tables."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import common as c

ext = [c.X0, c.X1, c.Y0, c.Y1]
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/confidence.tif") as s: conf = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
prob_oof = c.read(c.PROC + "/oof_prob_spatial2km.tif")
prob_in = c.read(c.PROC + "/flood_prob_rf.tif")
f = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
dec = pd.read_csv(c.PROC + "/_decile_errors.csv")
Y = lab == 1; VALID = lab != 255
oof_dom = VALID & np.isfinite(prob_oof)

# recall share per confidence class (in-sample deliverable raster)
tot_pos = int(Y[VALID].sum())
print("RECALL SHARE OF REFERENCE FLOOD BY CONFIDENCE (in-sample domain)")
rec = {}
for k, nm in ((1, "HIGH"), (2, "MEDIUM"), (3, "LOW"), (0, "not flooded")):
    n = int((VALID & (conf == k) & Y).sum())
    rec[nm] = 100 * n / tot_pos
    print(f"  {nm:12s} {n:>9,} true-flood px = {100*n/tot_pos:5.2f}% of reference")
cum = rec["HIGH"] + rec["MEDIUM"] + rec["LOW"]
print(f"  flood footprint (H+M+L) captures {cum:.2f}% of reference flood")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.24, wspace=0.2)

# --- confidence map
ax = fig.add_subplot(gs[0, 0])
cm = ListedColormap(["#f7f7f7", "#00441b", "#fdae61", "#d73027"])
ax.imshow(np.where(conf == 255, np.nan, conf), extent=ext, cmap=cm,
          norm=BoundaryNorm([0, 1, 2, 3, 4], 4), interpolation="nearest")
f.boundary.plot(ax=ax, color="#2166ac", linewidth=0.35)
ax.legend(handles=[Patch(fc="#00441b", label="1 HIGH"),
                   Patch(fc="#fdae61", label="2 MEDIUM"),
                   Patch(fc="#d73027", label="3 LOW"),
                   Patch(fc="#f7f7f7", ec="k", label="0 not flooded"),
                   Patch(ec="#2166ac", fc="none", label="reference")],
          loc="upper right", fontsize=8)
ax.set_title("confidence.tif  (thresholds 0.7 / 0.8 / 0.9)", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# --- calibration curve
ax = fig.add_subplot(gs[0, 1])
for nm, col, mk in (("OUT-OF-FOLD (spatial 2 km)", "#4575b4", "o"),
                    ("IN-SAMPLE (Step 4 model)", "#d73027", "s")):
    d = dec[dec.set == nm]
    ax.plot(d.lo + .05, 100 * d.flood_rate, mk + "-", color=col, lw=2, ms=7,
            label=nm.split(" (")[0].title())
ax.axhline(100 * float(Y[oof_dom].mean()), color="k", ls=":", lw=1.5,
           label=f"base rate {100*Y[oof_dom].mean():.1f}%")
for t in (0.7, 0.8, 0.9):
    ax.axvline(t, color="g", ls="--", lw=1.2, alpha=.7)
ax.text(0.72, 5, "LOW", color="g", fontsize=9)
ax.text(0.82, 5, "MED", color="g", fontsize=9)
ax.text(0.92, 5, "HIGH", color="g", fontsize=9)
ax.set_xlabel("RF probability bin"); ax.set_ylabel("true flooded share (%)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3)
ax.set_title("Calibration — flat to 0.7, then breaks sharply", fontsize=11)

# --- error by confidence, in-sample vs oof
ax = fig.add_subplot(gs[0, 2])
summ = pd.read_csv(c.PROC + "/confidence_summary.csv")
cls = ["HIGH", "MEDIUM", "LOW"]
ins = [float(summ[(summ.set.str.startswith("IN")) &
                  (summ.confidence == k)].error_rate.iloc[0]) for k in cls]
oo = [float(summ[(summ.set.str.startswith("OUT")) &
                 (summ.confidence == k)].error_rate.iloc[0]) for k in cls]
x = np.arange(3)
ax.bar(x - .2, ins, .4, label="in-sample (optimistic)", color="#d73027",
       edgecolor="k")
ax.bar(x + .2, oo, .4, label="out-of-fold (honest)", color="#4575b4",
       edgecolor="k")
for i, (a, b) in enumerate(zip(ins, oo)):
    ax.text(i - .2, a + .015, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + .2, b + .015, f"{b:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cls); ax.set_ylabel("error rate")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y"); ax.set_ylim(0, .85)
ax.set_title("HIGH survives honest validation;\nMEDIUM and LOW degrade",
             fontsize=11)

# --- error heatmap by confidence x land cover
ax = fig.add_subplot(gs[1, :2])
lcr = pd.read_csv(c.PROC + "/confidence_by_landcover.csv")
codes = sorted({int(s.split()[0]) for s in lcr.landcover})
M = np.full((len(codes), 3), np.nan)
N = np.zeros((len(codes), 3), int)
for i, code in enumerate(codes):
    for j, k in enumerate(cls):
        r = lcr[(lcr.landcover.str.startswith(f"{code} ")) &
                (lcr.confidence == k)]
        if len(r):
            M[i, j] = r.error_rate.iloc[0]; N[i, j] = r.n.iloc[0]
im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(cls)
ax.set_yticks(range(len(codes)))
ax.set_yticklabels([f"{k} {c.WC_CLASSES[k]}" for k in codes], fontsize=9)
for i in range(len(codes)):
    for j in range(3):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i,j]:.3f}\nn={N[i,j]:,}", ha="center",
                    va="center", fontsize=8.5,
                    color="white" if M[i, j] > .6 else "black")
        else:
            ax.text(j, i, "too few\npixels", ha="center", va="center",
                    fontsize=8, color="#666666")
plt.colorbar(im, ax=ax, shrink=.8, label="error rate (out-of-fold)")
ax.set_title("Error rate by confidence x land cover (out-of-fold, honest)\n"
             "HIGH confidence is reachable almost only on cropland and wetland",
             fontsize=11)

# --- summary
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
txt = ("STEP 6 SUMMARY\n"
       "----------------------------------------\n"
       "Data-driven thresholds from the\n"
       "out-of-fold calibration curve:\n"
       "   LOW    0.70 - 0.80\n"
       "   MEDIUM 0.80 - 0.90\n"
       "   HIGH   >= 0.90  (+ Otsu agreement)\n"
       "   below 0.70 -> not flooded\n\n"
       "(original fixed 0.3/0.5/0.8 replaced;\n"
       " 0.5-0.7 carries ~20% flood rate vs a\n"
       " 15.4% base rate, i.e. near-zero skill,\n"
       " so it is NOT called flood)\n\n"
       "OUT-OF-FOLD precision\n"
       "   HIGH     99.03%   err 0.010\n"
       "   MEDIUM   53.78%   err 0.462\n"
       "   LOW      33.51%   err 0.665\n\n"
       "Monotonic: yes (both in-sample\n"
       "and out-of-fold)\n\n"
       "RECALL share of reference flood\n"
       f"   HIGH     {rec['HIGH']:5.2f}%\n"
       f"   MEDIUM   {rec['MEDIUM']:5.2f}%\n"
       f"   LOW      {rec['LOW']:5.2f}%\n"
       f"   missed   {rec['not flooded']:5.2f}%\n\n"
       "-> The trustworthy core of the map is\n"
       "   small. Most of the reference flood\n"
       "   sits below 0.7 probability.")
ax.text(0, 1, txt, va="top", family="monospace", fontsize=9.5)

out = c.PROC + "/fig_step6_confidence.png"
plt.savefig(out, dpi=105, bbox_inches="tight")
print("figure ->", out)
