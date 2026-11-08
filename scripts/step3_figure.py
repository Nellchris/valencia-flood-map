"""STEP 3 figure: Otsu flood map vs reference."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import common as c

ext = [c.X0, c.X1, c.Y0, c.Y1]
thr = float(np.load(c.PROC + "/_otsu_thr.npy")[0])
f = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/flood_otsu.tif") as s: ot = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
dvv = c.read(c.PROC + "/delta_vv.tif")
V = lab != 255

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.14)

ax = fig.add_subplot(gs[0, 0])
d = np.full(lab.shape, 0, "uint8"); d[V] = 1; d[V & (ot == 1)] = 2
ax.imshow(d, extent=ext, cmap=ListedColormap(["#ffffff", "#eeeeee", "#08519c"]),
          norm=BoundaryNorm([0,1,2,3],3), interpolation="nearest")
f.boundary.plot(ax=ax, color="red", linewidth=0.35)
ax.set_title(f"Otsu flood map (ΔVV < {thr:+.2f} dB) + screening\n"
             f"red = EMSR773 reference outline", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(handles=[Patch(fc="#08519c", label="Otsu flood"),
                   Patch(ec="red", fc="none", label="reference")],
          loc="upper right", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
d = np.full(lab.shape, 0, "uint8")
d[V] = 1
d[V & (lab == 1) & (ot == 1)] = 2   # TP
d[V & (lab == 0) & (ot == 1)] = 3   # FP
d[V & (lab == 1) & (ot == 0)] = 4   # FN
ax.imshow(d, extent=ext, interpolation="nearest",
          cmap=ListedColormap(["#ffffff","#f0f0f0","#1a9850","#d73027","#4575b4"]),
          norm=BoundaryNorm([0,1,2,3,4,5],5))
ax.set_title("Error map", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
ax.legend(handles=[Patch(fc="#1a9850", label="TP 430,633"),
                   Patch(fc="#d73027", label="FP 1,349,755"),
                   Patch(fc="#4575b4", label="FN 628,817")],
          loc="upper right", fontsize=9)

ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(-20, 15, 200)
ax.hist(dvv[V & (lab == 0)], bins=bins, density=True, alpha=.55,
        color="#9ecae1", label="not flooded")
ax.hist(dvv[V & (lab == 1)], bins=bins, density=True, alpha=.55,
        color="#08306b", label="flooded")
ax.axvline(thr, color="#e6550d", lw=2.2, label=f"Otsu {thr:+.2f} dB")
ax.axvline(-3, color="#31a354", lw=2.2, ls="--", label="literature −3 dB")
ax.set_xlabel("delta_vv (dB)"); ax.set_ylabel("density"); ax.legend(fontsize=9)
ax.set_title("Otsu lands inside the no-change peak,\nnot at the water lobe", fontsize=11)

# threshold sweep
ax = fig.add_subplot(gs[1, 0])
hand = c.read(c.PROC+"/hand.tif"); slope = c.read(c.PROC+"/slope.tif")
scr = (slope <= 15) & (hand <= 50)
y = (lab == 1)[V]; x = dvv[V]; s_ = scr[V]
ts = np.arange(-14, 4.01, 0.25); F1s, IoUs, Ps, Rs = [], [], [], []
for t in ts:
    p = (x < t) & s_
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    F1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    IoUs.append(tp/(tp+fp+fn) if tp+fp+fn else 0); Ps.append(pr); Rs.append(rc)
ax.plot(ts, F1s, lw=2, label="F1"); ax.plot(ts, IoUs, lw=2, label="IoU")
ax.plot(ts, Ps, lw=1.2, ls=":", label="precision")
ax.plot(ts, Rs, lw=1.2, ls=":", label="recall")
ax.axvline(thr, color="#e6550d", lw=2, label=f"Otsu {thr:+.2f}")
ax.axvline(-3, color="#31a354", lw=2, ls="--", label="−3 dB")
bt = ts[int(np.argmax(F1s))]
ax.axvline(bt, color="k", lw=1.2, ls="-.", label=f"best F1 @ {bt:+.2f}")
ax.set_xlabel("ΔVV threshold (dB)"); ax.set_ylabel("score")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title(f"Threshold sweep — best achievable F1 = {max(F1s):.3f}", fontsize=11)
print(f"best-F1 threshold {bt:+.2f} dB  F1={max(F1s):.4f}  IoU={max(IoUs):.4f}")

# per land cover bars
ax = fig.add_subplot(gs[1, 1])
pred = (dvv < thr) & scr
ks, f1s, nn = [], [], []
for k in sorted(np.unique(wc[V])):
    m = (wc == k) & V
    if (lab[m] == 1).sum() < 500: continue
    tp = (pred[m] & (lab[m]==1)).sum(); fp = (pred[m] & (lab[m]==0)).sum()
    fn = (~pred[m] & (lab[m]==1)).sum()
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    ks.append(f"{k} {c.WC_CLASSES[k][:15]}"); f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    nn.append(int((lab[m]==1).sum()))
yy = np.arange(len(ks))
ax.barh(yy, f1s, color=["#1a9850" if v > .5 else "#d73027" for v in f1s])
ax.set_yticks(yy); ax.set_yticklabels(ks, fontsize=9)
ax.set_xlabel("F1"); ax.set_xlim(0, 1); ax.grid(alpha=.3, axis="x")
for i, (v, n) in enumerate(zip(f1s, nn)):
    ax.text(v+.02, i, f"{v:.2f} (n={n:,})", va="center", fontsize=8)
ax.set_title("Otsu F1 by land cover\nworks only on open water surfaces", fontsize=11)

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
txt = (
 "STEP 3 SUMMARY\n"
 "─────────────────────────────────────────\n"
 f"Otsu threshold          {thr:+.3f} dB\n"
 "Literature threshold     −3.000 dB\n"
 f"Difference              {thr+3:+.3f} dB\n\n"
 "Otsu + screening\n"
 "   IoU        0.1787\n"
 "   F1         0.3033\n"
 "   precision  0.2419\n"
 "   recall     0.4065\n\n"
 "−3 dB + screening\n"
 "   IoU        0.2029   (better)\n"
 "   F1         0.3374   (better)\n"
 "   precision  0.5149\n"
 "   recall     0.2509\n\n"
 "Accuracy        67.57 %\n"
 "Majority class  82.64 %\n"
 "  → Otsu is WORSE than predicting\n"
 "    'nothing flooded' everywhere.\n\n"
 "Predicted  178.04 km²\n"
 "Reference  105.95 km²\n"
 "  → 68 % areal over-prediction\n")
ax.text(0, 1, txt, va="top", family="monospace", fontsize=10.5)

out = c.PROC + "/fig_step3_otsu.png"
plt.savefig(out, dpi=105, bbox_inches="tight")
print("figure ->", out)
