"""STEP 5: honest validation -- random vs spatially blocked cross-validation.

Random and spatial CV differ ONLY in how folds are assigned. Everything else
(training budget, model settings, evaluation on the full test fold) is identical,
so the score gap is attributable to spatial leakage alone.
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier
import common as c

SEED = 42
N_FOLDS = 5
N_TRAIN = 300_000          # same budget as Step 4, so scores are comparable
BUFFER_M = 500.0
CHUNK = 1_000_000
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------ feature stack
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
cont = {n: c.read(c.PROC + f"/{n}.tif")
        for n in ("delta_vv", "delta_vh", "slope", "hand", "dist_river")}
CONT = list(cont)
WC_CODES = [10, 20, 30, 40, 50, 60, 90]
FEATS = CONT + [f"lc_{k}" for k in WC_CODES]

feat_ok = np.ones(lab.shape, bool)
for a in cont.values(): feat_ok &= np.isfinite(a)
VALID = (lab != 255) & feat_ok
Y = (lab == 1)
print(f"label domain: {VALID.sum():,} px, flooded {100*Y[VALID].mean():.2f}%")

rr, cc = np.indices(lab.shape)


def build(mask):
    cols = [cont[n][mask].astype("float32") for n in CONT]
    w = wc[mask]
    cols += [(w == k).astype("float32") for k in WC_CODES]
    return np.column_stack(cols)


def metrics(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    pr = tp / (tp + fp) if tp + fp else np.nan
    rc = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * pr * rc / (pr + rc) if (pr and rc and pr + rc) else np.nan
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    return iou, f1, pr, rc, tp, fp, fn, tn


def predict_mask(model, mask):
    idx = np.flatnonzero(mask)
    out = np.zeros(len(idx), "float32")
    ri, ci = np.unravel_index(idx, lab.shape)
    for a in range(0, len(idx), CHUNK):
        b = min(a + CHUNK, len(idx))
        m = np.zeros(lab.shape, bool); m[ri[a:b], ci[a:b]] = True
        out[a:b] = model.predict_proba(build(m))[:, 1]
    return out


def subsample(mask, n, rs):
    """Stratified subsample of `mask` to n pixels, preserving class ratio."""
    idx = np.flatnonzero(mask)
    y = Y.ravel()[idx]
    pos, neg = idx[y], idx[~y]
    if len(idx) <= n:
        return mask.copy()
    frac = len(pos) / len(idx)
    np_, nn_ = int(round(n * frac)), n - int(round(n * frac))
    np_ = min(np_, len(pos)); nn_ = min(nn_, len(neg))
    sel = np.concatenate([rs.choice(pos, np_, replace=False),
                          rs.choice(neg, nn_, replace=False)])
    m = np.zeros(lab.shape, bool); m.ravel()[sel] = True
    return m


def run_cv(name, block_km, fold_arr, buffer_mask):
    """fold_arr: int fold id per pixel (-1 = unassigned). buffer_mask excluded."""
    rows, imps = [], []
    usable = VALID & ~buffer_mask
    print(f"\n{'='*78}\n{name}  (block {block_km if block_km else 'n/a'})")
    print(f"  buffer excludes {int((VALID & buffer_mask).sum()):,} px "
          f"({100*(VALID & buffer_mask).sum()/VALID.sum():.1f}% of domain); "
          f"usable {int(usable.sum()):,}")
    for f in range(N_FOLDS):
        t0 = time.time()
        te = usable & (fold_arr == f)
        tr = usable & (fold_arr != f) & (fold_arr >= 0)
        if te.sum() == 0 or Y[te].sum() == 0 or tr.sum() == 0:
            print(f"   fold {f}: SKIPPED (test px {int(te.sum()):,}, "
                  f"pos {int(Y[te].sum()):,})")
            rows.append(dict(method=name, block_size_km=block_km, fold=f,
                             iou=np.nan, f1=np.nan, precision=np.nan,
                             recall=np.nan, n_test=int(te.sum()),
                             n_train=int(tr.sum()), test_pos_rate=np.nan))
            continue
        rs = np.random.default_rng(SEED + f)
        trs = subsample(tr, N_TRAIN, rs)
        model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                       class_weight="balanced",
                                       random_state=SEED, n_jobs=-1)
        model.fit(build(trs), Y[trs])
        imps.append(model.feature_importances_)
        p = predict_mask(model, te) > 0.5
        iou, f1, pr, rc, tp, fp, fn, tn = metrics(p, Y[te])
        rows.append(dict(method=name, block_size_km=block_km, fold=f,
                         iou=iou, f1=f1, precision=pr, recall=rc,
                         n_test=int(te.sum()), n_train=int(trs.sum()),
                         test_pos_rate=float(Y[te].mean())))
        print(f"   fold {f}: IoU {iou:.4f}  F1 {f1:.4f}  P {pr:.4f}  R {rc:.4f}"
              f"   test {int(te.sum()):>9,} px  pos {100*Y[te].mean():5.2f}%"
              f"   ({time.time()-t0:.0f}s)")
    return rows, (np.array(imps) if imps else None)


# ------------------------------------------------------- RANDOM 5-fold folds
fold_rand = np.full(lab.shape, -1, "int8")
idx = np.flatnonzero(VALID)
y = Y.ravel()[idx]
for cls in (False, True):                      # stratify
    sub = idx[y == cls]
    perm = rng.permutation(len(sub))
    fold_rand.ravel()[sub[perm]] = np.arange(len(sub)) % N_FOLDS
no_buf = np.zeros(lab.shape, bool)

all_rows, imp_random, imp_spatial2 = [], None, None
r, imp_random = run_cv("random", np.nan, fold_rand, no_buf)
all_rows += r

# ------------------------------------------------------- SPATIAL block folds
for block_km in (1, 2, 4):
    bpx = int(block_km * 1000 / c.RES)
    nbx = int(np.ceil(lab.shape[1] / bpx)); nby = int(np.ceil(lab.shape[0] / bpx))
    nblocks = nbx * nby
    bid = (rr // bpx) * nbx + (cc // bpx)
    brs = np.random.default_rng(SEED + 1000 * block_km)
    perm = brs.permutation(nblocks)
    bfold = np.empty(nblocks, "int8")
    bfold[perm] = np.arange(nblocks) % N_FOLDS
    fold_sp = bfold[bid]

    # buffer: any pixel within BUFFER_M of a pixel belonging to a DIFFERENT fold
    dists = np.stack([distance_transform_edt(fold_sp != f,
                                             sampling=c.RES).astype("float32")
                      for f in range(N_FOLDS)])
    dists[fold_sp[None].astype(np.intp),
          rr[None], cc[None]] = np.inf          # ignore own-fold distance
    buffer_mask = dists.min(axis=0) < BUFFER_M
    del dists

    print(f"\n[{block_km} km blocks] {nblocks} blocks "
          f"({nby} x {nbx}), ~{nblocks//N_FOLDS} per fold")
    r, imps = run_cv(f"spatial_{block_km}km", block_km, fold_sp, buffer_mask)
    all_rows += r
    if block_km == 2:
        imp_spatial2 = imps

df = pd.DataFrame(all_rows)
df.to_csv(c.PROC + "/validation_results.csv", index=False)
print(f"\n-> processed/validation_results.csv  ({len(df)} rows)")

# ------------------------------------------------------------------ headline
print("\n" + "=" * 78)
print("SUMMARY  (mean +/- sd across folds)")
print("=" * 78)
print(f"{'method':16s} {'blk':>5s} {'IoU':>16s} {'F1':>16s} "
      f"{'prec':>16s} {'recall':>16s}")
print("-" * 78)
summ = {}
for (meth, blk), g in df.groupby(["method", "block_size_km"], dropna=False):
    s = {m: (g[m].mean(), g[m].std()) for m in
         ("iou", "f1", "precision", "recall")}
    summ[meth] = s
    bl = "-" if np.isnan(blk) else f"{blk:g}km"
    print(f"{meth:16s} {bl:>5s} " + " ".join(
        f"{s[m][0]:7.4f}+/-{s[m][1]:5.4f}" for m in
        ("iou", "f1", "precision", "recall")))

rf1 = summ["random"]["f1"][0]; sf1 = summ["spatial_2km"]["f1"][0]
riou = summ["random"]["iou"][0]; siou = summ["spatial_2km"]["iou"][0]
print("\n" + "*" * 78)
print("HEADLINE")
print("*" * 78)
print(f"  mean random-CV      F1 = {rf1:.4f}   IoU = {riou:.4f}")
print(f"  mean spatial-CV 2km F1 = {sf1:.4f}   IoU = {siou:.4f}")
print(f"  ABSOLUTE GAP           = {rf1-sf1:+.4f} F1   ({riou-siou:+.4f} IoU)")
print(f"  RELATIVE DROP          = {100*(rf1-sf1)/rf1:.1f}% of F1 lost")
print("*" * 78)

# -------------------------------------------------- importance comparison
print("\n" + "=" * 78)
print("FEATURE IMPORTANCE: random CV vs spatial CV (2 km)")
print("=" * 78)
ir, isp = imp_random.mean(0), imp_spatial2.mean(0)
irs, isps = imp_random.std(0), imp_spatial2.std(0)
print(f"{'feature':14s} {'random':>16s} {'spatial2km':>16s} {'delta':>9s} {'%chg':>8s}")
print("-" * 68)
for i in np.argsort(ir)[::-1]:
    d = isp[i] - ir[i]
    pc = 100 * d / ir[i] if ir[i] > 1e-9 else np.nan
    print(f"{FEATS[i]:14s} {ir[i]:8.4f}+/-{irs[i]:5.4f} "
          f"{isp[i]:8.4f}+/-{isps[i]:5.4f} {d:+9.4f} {pc:+7.1f}%")
grp = lambda v: {"radar": v[0] + v[1], "terrain(HAND+slope)": v[2] + v[3],
                 "dist_river": v[4], "landcover": v[5:].sum()}
gr, gs = grp(ir), grp(isp)
print(f"\n{'group':22s} {'random':>9s} {'spatial':>9s} {'delta':>9s}")
for k in gr:
    print(f"{k:22s} {gr[k]:9.4f} {gs[k]:9.4f} {gs[k]-gr[k]:+9.4f}")

json.dump({"features": FEATS,
           "imp_random_mean": ir.tolist(), "imp_random_sd": irs.tolist(),
           "imp_spatial2_mean": isp.tolist(), "imp_spatial2_sd": isps.tolist(),
           "headline": {"random_f1": rf1, "spatial2_f1": sf1,
                        "random_iou": riou, "spatial2_iou": siou}},
          open(c.PROC + "/_cv_meta.json", "w"), indent=1)
print("\ndone")
