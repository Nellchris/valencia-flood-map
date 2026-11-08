"""STEP 7 figure: exposure estimates vs the EMS reference benchmark."""
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
df = pd.read_csv(c.PROC + "/exposure_results.csv")
rc = pd.read_csv(c.PROC + "/exposure_by_roadclass.csv")
with rasterio.open(c.PROC + "/confidence.tif") as s: conf = s.read(1)
with rasterio.open(c.PROC + "/flood_rf.tif") as s: rf = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
prob = c.read(c.PROC + "/flood_prob_rf.tif")
ref = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
bld = gpd.read_file(c.PROC + "/buildings.gpkg", layer="buildings")
pw = wc == 80

SHORT = {"lower (HIGH only)": "lower\n(HIGH)",
         "moderate (HIGH+MEDIUM)": "moderate\n(HIGH+MED)",
         "central (full RF footprint)": "central\n(RF p>0.5)",
         "upper (footprint + LOW)": "upper\n(= central)",
         "upper_extended (p>0.40)": "upper ext\n(p>0.4)",
         "EMS reference (EMSR773)": "EMS\nREFERENCE"}
COL = {"lower (HIGH only)": "#00441b", "moderate (HIGH+MEDIUM)": "#41ab5d",
       "central (full RF footprint)": "#fdae61",
       "upper (footprint + LOW)": "#f46d43",
       "upper_extended (p>0.40)": "#d73027",
       "EMS reference (EMSR773)": "#2166ac"}
ORD = list(SHORT)

fig = plt.figure(figsize=(19, 12.5))
gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.22)

# --- nested footprint map
ax = fig.add_subplot(gs[0, 0])
nest = np.zeros(conf.shape, "uint8")
nest[np.isfinite(prob) & (prob > 0.40)] = 1
nest[rf == 1] = 2
nest[(conf >= 1) & (conf <= 2)] = 3
nest[conf == 1] = 4
nest[pw] = 0
ax.imshow(np.where(nest == 0, np.nan, nest), extent=ext,
          cmap=ListedColormap(["#fee0d2", "#fdae61", "#41ab5d", "#00441b"]),
          norm=BoundaryNorm([1, 2, 3, 4, 5], 4), interpolation="nearest")
ref.boundary.plot(ax=ax, color="#2166ac", linewidth=0.4)
ax.legend(handles=[Patch(fc="#00441b", label="lower (HIGH)"),
                   Patch(fc="#41ab5d", label="moderate (+MEDIUM)"),
                   Patch(fc="#fdae61", label="central (RF p>0.5) = upper"),
                   Patch(fc="#fee0d2", label="upper ext (p>0.4)"),
                   Patch(ec="#2166ac", fc="none", label="EMS reference")],
          loc="upper right", fontsize=8)
ax.set_title("Nested exposure footprints (permanent water removed)", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# --- buildings
ax = fig.add_subplot(gs[0, 1])
v = [float(df[df.estimate == e].n_buildings.iloc[0]) for e in ORD]
b = ax.bar(range(len(ORD)), v, color=[COL[e] for e in ORD], edgecolor="k")
rb = float(df[df.estimate.str.startswith("EMS")].n_buildings.iloc[0])
ax.axhline(rb, color="#2166ac", ls="--", lw=1.6,
           label=f"EMS reference = {rb:,.0f}")
for i, x in enumerate(v):
    ax.text(i, x + max(v) * .015, f"{x:,.0f}", ha="center", fontsize=9)
ax.set_xticks(range(len(ORD)))
ax.set_xticklabels([SHORT[e] for e in ORD], fontsize=8)
ax.set_ylabel("buildings inside footprint"); ax.legend(fontsize=8)
ax.grid(alpha=.3, axis="y")
ax.set_title("Buildings exposed", fontsize=11)

# --- roads
ax = fig.add_subplot(gs[0, 2])
v = [float(df[df.estimate == e].road_km.iloc[0]) for e in ORD]
ax.bar(range(len(ORD)), v, color=[COL[e] for e in ORD], edgecolor="k")
rk = float(df[df.estimate.str.startswith("EMS")].road_km.iloc[0])
ax.axhline(rk, color="#2166ac", ls="--", lw=1.6,
           label=f"EMS reference = {rk:,.0f} km")
for i, x in enumerate(v):
    ax.text(i, x + max(v) * .015, f"{x:,.0f}", ha="center", fontsize=9)
ax.set_xticks(range(len(ORD)))
ax.set_xticklabels([SHORT[e] for e in ORD], fontsize=8)
ax.set_ylabel("road length inside footprint (km)"); ax.legend(fontsize=8)
ax.grid(alpha=.3, axis="y")
ax.set_title("Roads exposed", fontsize=11)

# --- ratio to reference
ax = fig.add_subplot(gs[1, 0])
est = [e for e in ORD if not e.startswith("EMS")]
xs = np.arange(len(est))
for off, key, lb, cl in ((-.25, "n_buildings", "buildings", "#762a83"),
                         (0.0, "road_km", "road km", "#1b7837"),
                         (0.25, "area_km2", "area", "#999999")):
    r = [float(df[df.estimate == e][key].iloc[0]) /
         float(df[df.estimate.str.startswith("EMS")][key].iloc[0]) for e in est]
    ax.bar(xs + off, r, .25, label=lb, color=cl, edgecolor="k")
    for i, q in enumerate(r):
        ax.text(xs[i] + off, q + .02, f"{q:.2f}", ha="center", fontsize=7.5,
                rotation=90)
ax.axhline(1, color="#2166ac", ls="--", lw=1.6, label="reference = 1.0")
ax.set_xticks(xs); ax.set_xticklabels([SHORT[e] for e in est], fontsize=8)
ax.set_ylabel("ratio to EMS reference"); ax.legend(fontsize=8)
ax.grid(alpha=.3, axis="y")
ax.set_title("Each estimate / reference\n"
             "area tracks far better than buildings", fontsize=11)

# --- road km by class, central vs reference
ax = fig.add_subplot(gs[1, 1])
piv = rc.pivot_table(index="highway", columns="estimate", values="road_km",
                     aggfunc="sum").fillna(0)
keep = piv.sum(axis=1).sort_values(ascending=False).head(8).index
piv = piv.loc[keep]
cols = [e for e in ORD if e in piv.columns]
yy = np.arange(len(keep)); w = 0.8 / len(cols)
for i, e in enumerate(cols):
    ax.barh(yy + (i - (len(cols) - 1) / 2) * w, piv[e].values, w,
            color=COL[e], edgecolor="k", linewidth=.3,
            label=SHORT[e].replace("\n", " "))
ax.set_yticks(yy); ax.set_yticklabels(keep, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("road km inside footprint"); ax.legend(fontsize=7.5)
ax.grid(alpha=.3, axis="x")
ax.set_title("Exposed road length by OSM class", fontsize=11)

# --- summary text
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
g = lambda e, k: float(df[df.estimate == e][k].iloc[0])
txt = ("STEP 7 SUMMARY\n"
       "------------------------------------------\n"
       f"OSM assets on the 33.5 x 32.3 km grid:\n"
       f"   {len(bld):,} buildings\n"
       f"   {rc.groupby('estimate').total_km.sum().iloc[0]:,.0f} km road "
       "(drivable net)\n\n"
       "BUILDINGS EXPOSED\n")
for e in ORD:
    txt += f"   {SHORT[e].replace(chr(10),' '):22s} {g(e,'n_buildings'):>9,.0f}\n"
txt += "\nROAD KM EXPOSED\n"
for e in ORD:
    txt += f"   {SHORT[e].replace(chr(10),' '):22s} {g(e,'road_km'):>9,.1f}\n"
txt += ("\n'upper' is IDENTICAL to 'central':\n"
        "the data-driven LOW band (p>=0.70) lies\n"
        "inside the p>0.5 footprint, so the union\n"
        "adds nothing. 'upper ext' (p>0.4) is the\n"
        "nearest thing to the original intent.\n\n"
        "The lower bound carries almost no\n"
        "buildings -- HIGH confidence is\n"
        "essentially unreachable on built-up land\n"
        "(Step 6: 4 of 63,353 HIGH px).")
ax.text(0, 1, txt, va="top", family="monospace", fontsize=9)

out = c.PROC + "/fig_step7_exposure.png"
plt.savefig(out, dpi=105, bbox_inches="tight")
print("figure ->", out)
