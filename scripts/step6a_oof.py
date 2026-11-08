"""STEP 6a: out-of-fold spatial-2km probabilities + decile error-rate table.

The Step 4 probability surface is in-sample (that model saw 300k pixels drawn
from this same domain), so calibrating a confidence layer on it would inherit
exactly the optimism Step 5 measured. Here we regenerate predictions where every
pixel is scored by the spatial-2km fold model that did NOT see its block, giving
honest out-of-fold probabilities to calibrate against.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier
import common as c

SEED, N_FOLDS, N_TRAIN, BUFFER_M, CHUNK = 42, 5, 300_000, 500.0, 1_000_000
BLOCK_KM = 2

with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
cont = {n: c.read(c.PROC + f"/{n}.tif")
        for n in ("delta_vv", "delta_vh", "slope", "hand", "dist_river")}
CONT = list(cont); WC_CODES = [10, 20, 30, 40, 50, 60, 90]
feat_ok = np.ones(lab.shape, bool)
for a in cont.values(): feat_ok &= np.isfinite(a)
VALID = (lab != 255) & feat_ok
Y = (lab == 1)
rr, cc = np.indices(lab.shape)


def build(mask):
    cols = [cont[n][mask].astype("float32") for n in CONT]
    w = wc[mask]
    cols += [(w == k).astype("float32") for k in WC_CODES]
    return np.column_stack(cols)


def subsample(mask, n, rs):
    idx = np.flatnonzero(mask); y = Y.ravel()[idx]
    pos, neg = idx[y], idx[~y]
    if len(idx) <= n: return mask.copy()
    frac = len(pos) / len(idx)
    np_ = min(int(round(n * frac)), len(pos)); nn_ = min(n - np_, len(neg))
    sel = np.concatenate([rs.choice(pos, np_, replace=False),
                          rs.choice(neg, nn_, replace=False)])
    m = np.zeros(lab.shape, bool); m.ravel()[sel] = True
    return m


# -- identical fold assignment to Step 5 (same seed, same block grid)
bpx = int(BLOCK_KM * 1000 / c.RES)
nbx = int(np.ceil(lab.shape[1] / bpx)); nby = int(np.ceil(lab.shape[0] / bpx))
nblocks = nbx * nby
bid = (rr // bpx) * nbx + (cc // bpx)
brs = np.random.default_rng(SEED + 1000 * BLOCK_KM)
perm = brs.permutation(nblocks)
bfold = np.empty(nblocks, "int8"); bfold[perm] = np.arange(nblocks) % N_FOLDS
fold_sp = bfold[bid]
dists = np.stack([distance_transform_edt(fold_sp != f,
                                         sampling=c.RES).astype("float32")
                  for f in range(N_FOLDS)])
dists[fold_sp[None].astype(np.intp), rr[None], cc[None]] = np.inf
buffer_mask = dists.min(axis=0) < BUFFER_M
del dists
usable = VALID & ~buffer_mask
print(f"usable (non-buffer) label px: {int(usable.sum()):,}")

oof = np.full(lab.shape, np.nan, "float32")
for f in range(N_FOLDS):
    t = time.time()
    te = usable & (fold_sp == f)
    tr = usable & (fold_sp != f)
    rs = np.random.default_rng(SEED + f)
    trs = subsample(tr, N_TRAIN, rs)
    model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                   class_weight="balanced", random_state=SEED,
                                   n_jobs=-1)
    model.fit(build(trs), Y[trs])
    idx = np.flatnonzero(te); ri, ci = np.unravel_index(idx, lab.shape)
    for a in range(0, len(idx), CHUNK):
        b = min(a + CHUNK, len(idx))
        m = np.zeros(lab.shape, bool); m[ri[a:b], ci[a:b]] = True
        oof[m] = model.predict_proba(build(m))[:, 1]
    print(f"  fold {f}: scored {len(idx):,} px  ({time.time()-t:.0f}s)", flush=True)

c.write(c.PROC + "/oof_prob_spatial2km.tif",
        np.where(np.isfinite(oof), oof, c.NODATA_F), "float32", c.NODATA_F)
print("-> processed/oof_prob_spatial2km.tif")

# ---------------------------------------------------- decile error rates
insample = c.read(c.PROC + "/flood_prob_rf.tif")
edges = np.round(np.arange(0, 1.01, 0.1), 2)
rows = []
for nm, P, M in (("OUT-OF-FOLD (spatial 2 km)", oof, usable & np.isfinite(oof)),
                 ("IN-SAMPLE (Step 4 model)", insample, VALID)):
    print("\n" + "=" * 86)
    print(f"RF PROBABILITY DECILES -- {nm}")
    print("=" * 86)
    print(f"{'bin':>12s} {'n_px':>11s} {'%dom':>6s} {'flood_rate':>11s} "
          f"{'pred':>6s} {'ERROR':>8s} {'jump':>7s}")
    print("-" * 86)
    prev = None
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = M & (P >= lo) & (P < hi if hi < 1 else P <= 1.0)
        n = int(m.sum())
        if n == 0:
            print(f"  [{lo:.1f},{hi:.1f}) {0:>11,}"); continue
        fr = float(Y[m].mean())
        pred = 1 if (lo + hi) / 2 > 0.5 else 0
        err = fr if pred == 0 else 1 - fr
        jump = "" if prev is None else f"{err-prev:+.4f}"
        print(f"  [{lo:.1f},{hi:.1f}) {n:>11,} {100*n/M.sum():5.1f}% "
              f"{fr:11.4f} {pred:>6d} {err:8.4f} {jump:>7s}")
        rows.append(dict(set=nm, lo=lo, hi=hi, n=n, flood_rate=fr,
                         pred=pred, error=err))
        prev = err
pd.DataFrame(rows).to_csv(c.PROC + "/_decile_errors.csv", index=False)
print("\n-> processed/_decile_errors.csv\ndone")
