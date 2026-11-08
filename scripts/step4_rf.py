"""STEP 4: Random Forest flood model."""
import os, sys, time, warnings, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio
from sklearn.ensemble import RandomForestClassifier
import common as c

SEED, N_SAMPLE = 42, 300_000
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------ feature stack
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
cont = {n: c.read(c.PROC + f"/{n}.tif")
        for n in ("delta_vv", "delta_vh", "slope", "hand", "dist_river")}
CONT = list(cont)
WC_CODES = [10, 20, 30, 40, 50, 60, 90]
FEATS = CONT + [f"lc_{k}_{c.WC_CLASSES[k].split('/')[0].replace(' ','')}"
                for k in WC_CODES]
print(f"{len(FEATS)} features: {FEATS}")

feat_ok = np.ones(lab.shape, bool)
for a in cont.values(): feat_ok &= np.isfinite(a)

def build(mask):
    """Stack features for the pixels in `mask` (one row per pixel)."""
    cols = [cont[n][mask].astype("float32") for n in CONT]
    w = wc[mask]
    cols += [(w == k).astype("float32") for k in WC_CODES]
    return np.column_stack(cols)

# ---------------------------------------------- stratified 300k sample
V = (lab != 255) & feat_ok
idx = np.flatnonzero(V)
yall = (lab.ravel()[idx] == 1)
pos, neg = idx[yall], idx[~yall]
frac = len(pos) / len(idx)
n_pos = int(round(N_SAMPLE * frac)); n_neg = N_SAMPLE - n_pos
sp = rng.choice(pos, n_pos, replace=False)
sn = rng.choice(neg, n_neg, replace=False)
train_idx = np.concatenate([sp, sn]); rng.shuffle(train_idx)
held_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

print("\n" + "=" * 72)
print("SAMPLING")
print("=" * 72)
print(f"  valid label domain    {len(idx):10,} px   flooded {100*frac:5.2f}%")
print(f"  TRAIN sample          {len(train_idx):10,} px   "
      f"flooded {100*n_pos/N_SAMPLE:5.2f}%  ({n_pos:,} pos / {n_neg:,} neg)")
print(f"  HELD-OUT              {len(held_idx):10,} px   "
      f"flooded {100*(lab.ravel()[held_idx]==1).mean():5.2f}%")
print(f"  class ratio preserved: train {n_neg/n_pos:.2f}:1  "
      f"vs domain {(len(idx)-len(pos))/len(pos):.2f}:1")

mask_tr = np.zeros(lab.shape, bool); mask_tr.ravel()[train_idx] = True
Xtr = build(mask_tr); ytr = (lab[mask_tr] == 1)

# ------------------------------------------------------------------- train
t = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            class_weight="balanced", random_state=SEED,
                            n_jobs=-1)
rf.fit(Xtr, ytr)
print(f"\ntrained in {time.time()-t:.0f}s  "
      f"(200 trees, max_depth=15, class_weight='balanced', seed={SEED})")

# -------------------------------------------------------- feature importance
imp = rf.feature_importances_
order = np.argsort(imp)[::-1]
print("\n" + "=" * 72)
print("FEATURE IMPORTANCE (Gini)")
print("=" * 72)
for r, i in enumerate(order, 1):
    bar = "#" * int(round(imp[i] * 120))
    print(f"  {r:2d}. {FEATS[i]:26s} {imp[i]:7.4f}  {bar}")
grp = {"radar (ΔVV+ΔVH)": imp[0] + imp[1], "terrain (slope+HAND)": imp[2] + imp[3],
       "dist_river": imp[4], "landcover (one-hot sum)": imp[5:].sum()}
print("\n  GROUPED:")
for k, v in sorted(grp.items(), key=lambda x: -x[1]):
    print(f"    {k:26s} {v:7.4f}  {'#'*int(round(v*120))}")

def metrics(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc,
                f1=2*pr*rc/(pr+rc) if pr+rc else 0.0,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 0.0)

# --------------------------------------------------- held-out evaluation
mask_ho = np.zeros(lab.shape, bool); mask_ho.ravel()[held_idx] = True
print("\n" + "=" * 72)
print("HELD-OUT EVALUATION  (random split -- spatially interleaved)")
print("=" * 72)
ho_pred = np.zeros(len(held_idx), bool); ho_prob = np.zeros(len(held_idx), "float32")
rows_ho = np.unravel_index(held_idx, lab.shape)
CH = 1_000_000
for a in range(0, len(held_idx), CH):
    b = min(a + CH, len(held_idx))
    m = np.zeros(lab.shape, bool); m[rows_ho[0][a:b], rows_ho[1][a:b]] = True
    p = rf.predict_proba(build(m))[:, 1]
    ho_prob[a:b] = p; ho_pred[a:b] = p > 0.5
yho = lab.ravel()[held_idx] == 1
mh = metrics(ho_pred, yho)
print(f"  IoU {mh['iou']:.4f}   F1 {mh['f1']:.4f}   "
      f"precision {mh['precision']:.4f}   recall {mh['recall']:.4f}")
print(f"  accuracy {100*(mh['tp']+mh['tn'])/len(yho):.2f}%  "
      f"(majority baseline {100*(1-yho.mean()):.2f}%)")
print(f"  TP {mh['tp']:,}  FP {mh['fp']:,}  FN {mh['fn']:,}  TN {mh['tn']:,}")
mtr = metrics(rf.predict(Xtr).astype(bool), ytr)
print(f"\n  training-set F1 {mtr['f1']:.4f} vs held-out F1 {mh['f1']:.4f}  "
      f"(gap {mtr['f1']-mh['f1']:+.4f})")
print("  NOTE: held-out pixels are randomly interleaved with training pixels,")
print("        so this is the OPTIMISTIC random-CV number. Step 5 tests spatially.")

# --------------------------------------- predict full study area (chunked)
print("\npredicting full study area...", flush=True)
t = time.time()
prob = np.full(lab.shape, np.nan, "float32")
rows_all = np.flatnonzero(feat_ok)
rr = np.unravel_index(rows_all, lab.shape)
for a in range(0, len(rows_all), CH):
    b = min(a + CH, len(rows_all))
    m = np.zeros(lab.shape, bool); m[rr[0][a:b], rr[1][a:b]] = True
    prob[m] = rf.predict_proba(build(m))[:, 1]
print(f"  {len(rows_all):,} px in {time.time()-t:.0f}s")

c.write(c.PROC + "/flood_prob_rf.tif",
        np.where(np.isfinite(prob), prob, c.NODATA_F), "float32", c.NODATA_F)
binr = np.full(lab.shape, 255, "uint8")
binr[feat_ok] = (prob[feat_ok] > 0.5).astype("uint8")
c.write(c.PROC + "/flood_rf.tif", binr, "uint8", 255)
px = 100 / 1e6
print(f"\n-> flood_prob_rf.tif   -> flood_rf.tif")
print(f"   RF flood extent over predicted area : {(binr==1).sum()*px:8.2f} km2")
print(f"   (within label domain)               : "
      f"{((binr==1)&(lab!=255)).sum()*px:8.2f} km2")
print(f"   reference                           : {(lab==1).sum()*px:8.2f} km2")
print(f"   Otsu baseline                       :   178.04 km2")

json.dump({"features": FEATS, "importance": imp.tolist(),
           "heldout": {k: float(v) for k, v in mh.items()},
           "train_f1": mtr["f1"], "n_train": int(len(train_idx)),
           "n_heldout": int(len(held_idx))},
          open(c.PROC + "/_rf_meta.json", "w"), indent=1)
np.save(c.PROC + "/_rf_heldout.npy",
        np.column_stack([held_idx, yho, ho_prob]))
