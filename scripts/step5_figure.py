"""STEP 5 figure: random vs spatially blocked validation."""
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

df = pd.read_csv(c.PROC + "/validation_results.csv")
meta = json.load(open(c.PROC + "/_cv_meta.json"))
ORDER = ["random", "spatial_1km", "spatial_2km", "spatial_4km"]
LBL = {"random": "Random\n5-fold", "spatial_1km": "Spatial\n1 km",
       "spatial_2km": "Spatial\n2 km", "spatial_4km": "Spatial\n4 km"}
COL = {"random": "#d73027", "spatial_1km": "#fdae61",
       "spatial_2km": "#4575b4", "spatial_4km": "#313695"}

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

# --- main bar chart: all four metrics
ax = fig.add_subplot(gs[0, :2])
METR = ["iou", "f1", "precision", "recall"]
w = 0.2
xs = np.arange(len(METR))
for i, m in enumerate(ORDER):
    g = df[df.method == m]
    means = [g[k].mean() for k in METR]
    sds = [g[k].std() for k in METR]
    pos = xs + (i - 1.5) * w
    ax.bar(pos, means, w, yerr=sds, capsize=3, color=COL[m],
           label=LBL[m].replace("\n", " "), edgecolor="k", linewidth=.5)
    for j, mm in enumerate(means):
        if np.isfinite(mm):
            ax.text(pos[j], mm + sds[j] + .015, f"{mm:.3f}", ha="center",
                    fontsize=8.5, rotation=90)
    # per-fold points
    for j, k in enumerate(METR):
        v = g[k].dropna().values
        ax.scatter(np.full(len(v), pos[j]), v, s=14, c="k", zorder=5, alpha=.7)
ax.set_xticks(xs); ax.set_xticklabels([m.upper() for m in METR])
ax.set_ylabel("score"); ax.legend(fontsize=9, ncol=4)
ax.grid(alpha=.3, axis="y"); ax.set_ylim(0, 1.05)
ax.set_title("Random vs spatially blocked cross-validation\n"
             "bars = mean across 5 folds, whiskers = sd, dots = individual folds",
             fontsize=12)

# --- headline gap
ax = fig.add_subplot(gs[0, 2])
h = meta["headline"]
r, s = h["random_f1"], h["spatial2_f1"]
ax.bar(["Random CV", "Spatial CV\n(2 km)"], [r, s],
       color=["#d73027", "#4575b4"], edgecolor="k")
ax.plot([0, 1], [r, s], "k--", lw=1.5)
ax.annotate("", xy=(1, s), xytext=(1, r),
            arrowprops=dict(arrowstyle="<->", color="k", lw=2))
ax.text(1.06, (r + s) / 2, f"-{r-s:.3f}\n({100*(r-s)/r:.0f}%)",
        fontsize=13, fontweight="bold", va="center")
for i, v in enumerate([r, s]):
    ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("F1"); ax.set_ylim(0, max(r, s) * 1.45)
ax.grid(alpha=.3, axis="y")
ax.set_title("HEADLINE: optimism from\nspatial leakage", fontsize=12)

# --- per-fold F1
ax = fig.add_subplot(gs[1, 0])
for i, m in enumerate(ORDER):
    g = df[df.method == m].sort_values("fold")
    ax.plot(g.fold, g.f1, "o-", color=COL[m], label=LBL[m].replace("\n", " "),
            lw=2, ms=7)
ax.set_xlabel("fold"); ax.set_ylabel("F1"); ax.set_xticks(range(5))
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Per-fold F1 — spatial folds vary far more", fontsize=11)

# --- fold difficulty vs score
ax = fig.add_subplot(gs[1, 1])
for m in ORDER:
    g = df[df.method == m]
    ax.scatter(100 * g.test_pos_rate, g.f1, s=70, color=COL[m],
               label=LBL[m].replace("\n", " "), edgecolor="k", zorder=4)
ax.set_xlabel("flooded share of test fold (%)"); ax.set_ylabel("F1")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Fold composition drives spatial score\n"
             "(folds holding the Albufera score high)", fontsize=11)

# --- importance shift
ax = fig.add_subplot(gs[1, 2])
F = meta["features"]
ir = np.array(meta["imp_random_mean"]); isp = np.array(meta["imp_spatial2_mean"])
keep = np.argsort(ir)[::-1][:6]
yy = np.arange(len(keep))
ax.barh(yy - .2, ir[keep], .4, label="random CV", color="#d73027", edgecolor="k")
ax.barh(yy + .2, isp[keep], .4, label="spatial CV 2 km", color="#4575b4",
        edgecolor="k")
ax.set_yticks(yy); ax.set_yticklabels([F[i] for i in keep], fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("Gini importance"); ax.legend(fontsize=9)
ax.grid(alpha=.3, axis="x")
for i, k in enumerate(keep):
    ax.text(max(ir[k], isp[k]) + .006, i, f"{isp[k]-ir[k]:+.3f}", va="center",
            fontsize=8)
ax.set_title("Feature importance under spatial CV", fontsize=11)

out = c.PROC + "/fig_step5_validation.png"
plt.savefig(out, dpi=105, bbox_inches="tight")
print("figure ->", out)
