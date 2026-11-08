"""STEP 4 figure: RF prediction, probability surface, importances."""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import common as c

ext = [c.X0, c.X1, c.Y0, c.Y1]
meta = json.load(open(c.PROC + "/_rf_meta.json"))
f = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/flood_rf.tif") as s: rfb = s.read(1)
with rasterio.open(c.PROC + "/flood_otsu.tif") as s: ot = s.read(1)
prob = c.read(c.PROC + "/flood_prob_rf.tif")
V = lab != 255

fig = plt.figure(figsize=(18, 17))
gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.15)

ax = fig.add_subplot(gs[0, 0])
d = np.full(lab.shape, 0, "uint8"); d[rfb != 255] = 1; d[rfb == 1] = 2
ax.imshow(d, extent=ext, cmap=ListedColormap(["#ffffff", "#eeeeee", "#08519c"]),
          norm=BoundaryNorm([0, 1, 2, 3], 3), interpolation="nearest")
f.boundary.plot(ax=ax, color="red", linewidth=0.35)
ax.set_title("RF binary flood map (p > 0.5)\nred = EMSR773 reference", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(prob, extent=ext, cmap="turbo", vmin=0, vmax=1)
f.boundary.plot(ax=ax, color="k", linewidth=0.3)
plt.colorbar(im, ax=ax, shrink=0.78, label="P(flood)")
ax.set_title("RF probability surface", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[0, 2])
d = np.full(lab.shape, 0, "uint8"); d[V] = 1
d[V & (lab == 1) & (rfb == 1)] = 2
d[V & (lab == 0) & (rfb == 1)] = 3
d[V & (lab == 1) & (rfb == 0)] = 4
ax.imshow(d, extent=ext, interpolation="nearest",
          cmap=ListedColormap(["#ffffff", "#f0f0f0", "#1a9850", "#d73027", "#4575b4"]),
          norm=BoundaryNorm([0, 1, 2, 3, 4, 5], 5))
ax.legend(handles=[Patch(fc="#1a9850", label="TP"), Patch(fc="#d73027", label="FP"),
                   Patch(fc="#4575b4", label="FN")], loc="upper right", fontsize=9)
ax.set_title("RF error map", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[1, 0])
names, imp = meta["features"], np.array(meta["importance"])
o = np.argsort(imp)
cols = ["#d73027" if names[i] in ("delta_vv", "delta_vh") else
        "#4575b4" if names[i] in ("hand", "slope", "dist_river") else "#bdbdbd"
        for i in o]
ax.barh(np.arange(len(o)), imp[o], color=cols)
ax.set_yticks(np.arange(len(o))); ax.set_yticklabels([names[i] for i in o], fontsize=9)
ax.set_xlabel("Gini importance")
for i, v in enumerate(imp[o]):
    ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=8)
ax.legend(handles=[Patch(fc="#4575b4", label="terrain / position"),
                   Patch(fc="#d73027", label="radar change"),
                   Patch(fc="#bdbdbd", label="land cover")], fontsize=8,
          loc="lower right")
ax.set_title("Feature importance - terrain dominates", fontsize=11)
ax.set_xlim(0, imp.max() * 1.2)

ax = fig.add_subplot(gs[1, 1])
g = {"terrain\n(HAND+slope)": imp[2] + imp[3], "radar\n(dVV+dVH)": imp[0] + imp[1],
     "dist_river": imp[4], "land cover": imp[5:].sum()}
k = list(g); v = [g[x] for x in k]
ax.bar(k, v, color=["#4575b4", "#d73027", "#4575b4", "#bdbdbd"])
for i, vv in enumerate(v):
    ax.text(i, vv + .008, f"{vv:.3f}", ha="center", fontsize=10)
ax.set_ylabel("summed Gini importance"); ax.set_ylim(0, .6)
ax.set_title("Grouped: 68.5% of the model is terrain + channel position\n"
             "only 24.0% is the actual radar observation", fontsize=11)
ax.tick_params(axis="x", labelsize=9)

ax = fig.add_subplot(gs[1, 2])
bins = np.linspace(0, 1, 60)
ax.hist(prob[V & (lab == 0)], bins=bins, density=True, alpha=.6,
        color="#9ecae1", label="not flooded")
ax.hist(prob[V & (lab == 1)], bins=bins, density=True, alpha=.6,
        color="#08306b", label="flooded")
ax.axvline(.5, color="k", lw=1.5, ls="--", label="0.5 cut")
ax.set_xlabel("RF P(flood)"); ax.set_ylabel("density"); ax.legend(fontsize=9)
ax.set_title("Probability separation by true class", fontsize=11)

ho = np.load(c.PROC + "/_rf_heldout.npy")
yho, pho = ho[:, 1].astype(bool), ho[:, 2].astype("float32")
ax = fig.add_subplot(gs[2, 0])
ts = np.linspace(0.05, 0.95, 60); F1, IOU, P, R = [], [], [], []
for t in ts:
    p = pho > t
    tp = (p & yho).sum(); fp = (p & ~yho).sum(); fn = (~p & yho).sum()
    pr = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    F1.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    IOU.append(tp / (tp + fp + fn) if tp + fp + fn else 0)
    P.append(pr); R.append(rc)
ax.plot(ts, F1, lw=2, label="F1"); ax.plot(ts, IOU, lw=2, label="IoU")
ax.plot(ts, P, ls=":", label="precision"); ax.plot(ts, R, ls=":", label="recall")
ax.axvline(.5, color="k", ls="--", lw=1.2, label="0.5 default")
bi = int(np.argmax(F1))
ax.axvline(ts[bi], color="#e6550d", lw=1.5, ls="-.",
           label=f"best F1 {max(F1):.3f} @ {ts[bi]:.2f}")
ax.set_xlabel("probability cut"); ax.set_ylabel("score")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Held-out threshold sweep", fontsize=11)
print(f"best held-out F1 {max(F1):.4f} at cut {ts[bi]:.2f} "
      f"(default 0.5 -> {F1[int(np.argmin(abs(ts-0.5)))]:.4f})")

ax = fig.add_subplot(gs[2, 1])
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
ks, rf1, ot1 = [], [], []
for kk in sorted(np.unique(wc[V])):
    m = (wc == kk) & V
    if (lab[m] == 1).sum() < 500: continue
    def sc(pred, m=m):
        tp = (pred & (lab[m] == 1)).sum(); fp = (pred & (lab[m] == 0)).sum()
        fn = ((~pred) & (lab[m] == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0
    ks.append(f"{kk} {c.WC_CLASSES[kk][:14]}")
    rf1.append(sc(rfb[m] == 1)); ot1.append(sc(ot[m] == 1))
y = np.arange(len(ks))
ax.barh(y - 0.2, rf1, 0.4, label="Random Forest", color="#4575b4")
ax.barh(y + 0.2, ot1, 0.4, label="Otsu baseline", color="#d73027")
ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=9); ax.set_xlabel("F1")
ax.legend(fontsize=9); ax.grid(alpha=.3, axis="x"); ax.set_xlim(0, 1)
ax.set_title("RF vs Otsu by land cover", fontsize=11)

ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
h = meta["heldout"]
txt = ("STEP 4 SUMMARY\n"
       "--------------------------------------\n"
       f"train sample     {meta['n_train']:>10,} px\n"
       f"held-out         {meta['n_heldout']:>10,} px\n"
       "RF: 200 trees, max_depth=15,\n"
       "    class_weight=balanced, seed=42\n\n"
       "HELD-OUT (random, interleaved)\n"
       f"   IoU        {h['iou']:.4f}\n"
       f"   F1         {h['f1']:.4f}\n"
       f"   precision  {h['precision']:.4f}\n"
       f"   recall     {h['recall']:.4f}\n\n"
       f"train F1 {meta['train_f1']:.4f} -> held-out {h['f1']:.4f}\n"
       f"   apparent overfit gap {meta['train_f1']-h['f1']:+.4f}\n\n"
       "vs Otsu baseline F1 0.3033\n"
       f"   improvement {h['f1']-0.3033:+.4f}\n\n"
       "IMPORTANCE\n"
       "   hand        0.4006\n"
       "   dist_river  0.1952\n"
       "   delta_vv    0.1384\n"
       "   delta_vh    0.1019\n"
       "   slope       0.0890\n\n"
       "Areas\n"
       "   RF         234.20 km2\n"
       "   Otsu       178.04 km2\n"
       "   reference  105.95 km2\n\n"
       "! Held-out pixels are spatially\n"
       "  interleaved with training pixels.\n"
       "  Step 5 tests this properly.")
ax.text(0, 1, txt, va="top", family="monospace", fontsize=10)

out = c.PROC + "/fig_step4_rf.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print("figure ->", out)
