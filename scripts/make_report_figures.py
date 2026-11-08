"""Report figures, rendered in the web map's palette.

The Step 5-7 diagnostic figures in processed/ were working plots: nine panels,
matplotlib defaults, and a dark-green/red confidence ramp that is a red-green
pair. These are purpose-built for the report instead -- same numbers, same
colours as the interactive map, sized for a printed page.
"""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import common as c

OUT = os.path.join(c.BASE, "report", "figures")
os.makedirs(OUT, exist_ok=True)

FLOOD = "#3b82f6"; HIGH = "#0d9488"; MED = "#f59e0b"; LOW = "#ea580c"
GREY = "#9ca3af"; REF = "#c026d3"; INK = "#111827"; INK2 = "#4b5563"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

v = pd.read_csv(c.PROC + "/validation_results.csv")
meta = json.load(open(c.PROC + "/_cv_meta.json"))

# ---------------------------------------------------- Fig 1: validation
ORDER = [("random", "Random\n5-fold"), ("spatial_1km", "Spatial\n1 km"),
         ("spatial_2km", "Spatial\n2 km"), ("spatial_4km", "Spatial\n4 km")]
COLS = [LOW, "#93c5fd", FLOOD, "#1e40af"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9),
                              gridspec_kw={"width_ratios": [1.25, 1]})
xs = np.arange(len(ORDER))
means = [v[v.method == m].f1.mean() for m, _ in ORDER]
sds = [v[v.method == m].f1.std() for m, _ in ORDER]
ax.bar(xs, means, 0.62, yerr=sds, capsize=5, color=COLS, edgecolor=INK,
       linewidth=.6, error_kw=dict(ecolor=INK, lw=1.3))
for i, (m, _) in enumerate(ORDER):
    pts = v[v.method == m].f1.dropna().values
    ax.scatter(np.full(len(pts), xs[i]), pts, s=16, c=INK, zorder=5, alpha=.65)
    ax.text(xs[i], means[i] + sds[i] + .028, f"{means[i]:.3f}", ha="center",
            fontsize=9, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in ORDER], fontsize=8.5)
ax.set_ylabel("F1 score"); ax.set_ylim(0, 0.72)
ax.grid(axis="y", alpha=.25, lw=.6)
ax.set_title("(a)  F1 by validation scheme\nbars = mean of 5 folds, "
             "whiskers = sd, dots = folds", fontsize=9.5, loc="left")

F = meta["features"]
d_r = dict(zip(F, meta["imp_random_mean"]))
d_s = dict(zip(F, meta["imp_spatial2_mean"]))
grp = lambda d: [d["delta_vv"] + d["delta_vh"], d["hand"] + d["slope"],
                 d["dist_river"], sum(x for k, x in d.items() if k.startswith("lc_"))]
labels = ["Radar\n(ΔVV+ΔVH)", "Terrain\n(HAND+slope)", "Distance\nto river", "Land\ncover"]
gr, gs = grp(d_r), grp(d_s)
w = 0.38; yy = np.arange(len(labels))
ax2.barh(yy - w / 2, [100 * x for x in gr], w, label="Random CV",
         color=LOW, edgecolor=INK, linewidth=.6)
ax2.barh(yy + w / 2, [100 * x for x in gs], w, label="Spatial CV 2 km",
         color=FLOOD, edgecolor=INK, linewidth=.6)
for i, (a, b) in enumerate(zip(gr, gs)):
    ax2.text(100 * a + 1, yy[i] - w / 2, f"{100*a:.1f}", va="center", fontsize=8)
    ax2.text(100 * b + 1, yy[i] + w / 2, f"{100*b:.1f}", va="center", fontsize=8)
ax2.set_yticks(yy); ax2.set_yticklabels(labels, fontsize=8.5)
ax2.invert_yaxis(); ax2.set_xlabel("Gini importance (%)"); ax2.set_xlim(0, 62)
ax2.legend(fontsize=8, frameon=False); ax2.grid(axis="x", alpha=.25, lw=.6)
ax2.set_title("(b)  Feature importance\nterrain dominates under both schemes",
              fontsize=9.5, loc="left")
plt.tight_layout()
p1 = os.path.join(OUT, "fig1_validation.png")
plt.savefig(p1, dpi=160, bbox_inches="tight"); plt.close()
print("->", p1, f"{os.path.getsize(p1)/1e3:.0f} kB")

# --------------------------------------------------- Fig 2: confidence
with rasterio.open(c.PROC + "/confidence.tif") as s: conf = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
ref = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
dec = pd.read_csv(c.PROC + "/_decile_errors.csv")
cviz = np.where(wc == 80, 0, conf)
ext = [c.X0, c.X1, c.Y0, c.Y1]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2),
                              gridspec_kw={"width_ratios": [1.05, 1]})
ax.imshow(np.where(np.isin(cviz, [1, 2, 3]), cviz, np.nan), extent=ext,
          cmap=ListedColormap([HIGH, MED, LOW]),
          norm=BoundaryNorm([1, 2, 3, 4], 3), interpolation="nearest")
ref.boundary.plot(ax=ax, color=REF, linewidth=0.22, alpha=.42)
ax.set_xlim(c.X0, c.X1); ax.set_ylim(c.Y0, c.Y1)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(True)
ax.legend(handles=[Patch(fc=HIGH, label="High (99% correct)"),
                   Patch(fc=MED, label="Medium (54%)"),
                   Patch(fc=LOW, label="Low (34%)"),
                   Patch(ec=REF, fc="none", label="EMS reference")],
          loc="upper right", fontsize=7.5, framealpha=.95)
ax.set_title("(a)  Confidence layer\nhigh confidence clusters in the Albufera "
             "wetland and cropland", fontsize=9.5, loc="left")

oof = dec[dec.set.str.startswith("OUT")]
ins = dec[dec.set.str.startswith("IN")]
ax2.plot(ins.lo + .05, 100 * ins.flood_rate, "s--", color=GREY, lw=1.6, ms=5,
         label="In-sample (optimistic)")
ax2.plot(oof.lo + .05, 100 * oof.flood_rate, "o-", color=FLOOD, lw=2, ms=6,
         label="Out-of-fold (honest)")
base = 15.4
ax2.axhline(base, color=INK2, ls=":", lw=1.2)
ax2.text(0.02, base + 2.5, f"base rate {base}%", fontsize=7.5, color=INK2)
for t, col, nm in ((0.7, LOW, "Low"), (0.8, MED, "Med"), (0.9, HIGH, "High")):
    ax2.axvline(t, color=col, lw=1.3, alpha=.85)
    ax2.text(t + .008, 92, nm, fontsize=7.5, color=col, fontweight="bold")
ax2.set_xlabel("Random Forest probability"); ax2.set_ylabel("Pixels truly flooded (%)")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 105)
ax2.legend(fontsize=8, frameon=False, loc="upper left")
ax2.grid(alpha=.25, lw=.6)
ax2.set_title("(b)  Calibration\nflat to 0.70, then three sharp breaks",
              fontsize=9.5, loc="left")
plt.tight_layout()
p2 = os.path.join(OUT, "fig2_confidence.png")
plt.savefig(p2, dpi=160, bbox_inches="tight"); plt.close()
print("->", p2, f"{os.path.getsize(p2)/1e3:.0f} kB")

# ----------------------------------------------------- Fig 3: exposure
e = pd.read_csv(c.PROC + "/exposure_results.csv").set_index("estimate")
x = pd.read_csv(c.PROC + "/exposure_vs_reference.csv").set_index("estimate")
ROWS = [("lower (HIGH only)", "High\nconfidence", HIGH),
        ("moderate (HIGH+MEDIUM)", "High +\nMedium", MED),
        ("central (full RF footprint)", "Full model\nfootprint", FLOOD),
        ("EMS reference (EMSR773)", "EMS\nreference", REF)]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
xs = np.arange(len(ROWS))
bld = [e.loc[k, "n_buildings"] for k, _, _ in ROWS]
cols = [cl for _, _, cl in ROWS]
ax.bar(xs, bld, 0.62, color=cols, edgecolor=INK, linewidth=.6)
ax.axhline(e.loc["EMS reference (EMSR773)", "n_buildings"], color=REF, ls="--", lw=1.3)
for i, b in enumerate(bld):
    ax.text(xs[i], b + 700, f"{b:,}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([l for _, l, _ in ROWS], fontsize=8.5)
ax.set_ylabel("Buildings inside estimate"); ax.set_ylim(0, 27500)
ax.grid(axis="y", alpha=.25, lw=.6)
ax.set_title("(a)  Buildings exposed\nthe bounds straddle, they do not bracket",
             fontsize=9.5, loc="left")

prec = [x.loc[k, "bld_precision"] for k, _, _ in ROWS[:3]]
rec = [x.loc[k, "bld_recall"] for k, _, _ in ROWS[:3]]
w = 0.36; xs2 = np.arange(3)
ax2.bar(xs2 - w / 2, [100 * p for p in prec], w, label="Precision",
        color=HIGH, edgecolor=INK, linewidth=.6)
ax2.bar(xs2 + w / 2, [100 * r for r in rec], w, label="Recall",
        color=GREY, edgecolor=INK, linewidth=.6)
for i in range(3):
    ax2.text(xs2[i] - w / 2, 100 * prec[i] + 2, f"{100*prec[i]:.0f}%",
             ha="center", fontsize=8)
    ax2.text(xs2[i] + w / 2, 100 * rec[i] + 2, f"{100*rec[i]:.0f}%",
             ha="center", fontsize=8)
ax2.set_xticks(xs2); ax2.set_xticklabels([l for _, l, _ in ROWS[:3]], fontsize=8.5)
ax2.set_ylabel("Building-level agreement (%)"); ax2.set_ylim(0, 105)
ax2.legend(fontsize=8, frameon=False); ax2.grid(axis="y", alpha=.25, lw=.6)
ax2.set_title("(b)  Precision against recall\nhigh confidence is right but tiny",
              fontsize=9.5, loc="left")
plt.tight_layout()
p3 = os.path.join(OUT, "fig3_exposure.png")
plt.savefig(p3, dpi=160, bbox_inches="tight"); plt.close()
print("->", p3, f"{os.path.getsize(p3)/1e3:.0f} kB")
print("done")
