"""STEP 3: Otsu threshold baseline on delta_vv."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio
from skimage.filters import threshold_otsu
import common as c

with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
dvv = c.read(c.PROC + "/delta_vv.tif")
hand = c.read(c.PROC + "/hand.tif")
slope = c.read(c.PROC + "/slope.tif")

V = (lab != 255) & np.isfinite(dvv)
y = (lab == 1)[V]
x = dvv[V]
print(f"valid label pixels with delta_vv: {V.sum():,}   flooded {y.mean()*100:.2f}%")

# ------------------------------------------------------------- Otsu threshold
thr = float(threshold_otsu(x, nbins=512))
print("\n" + "=" * 72)
print("OTSU THRESHOLD")
print("=" * 72)
print(f"  Otsu threshold on delta_vv        : {thr:+.3f} dB")
print(f"  Literature manual threshold        :  -3.000 dB")
print(f"  Difference                         : {thr-(-3.0):+.3f} dB")
print(f"  delta_vv distribution: mean {x.mean():+.2f}  median {np.median(x):+.2f}  "
      f"sd {x.std():.2f}")
print(f"  pixels below Otsu threshold        : {100*(x<thr).mean():5.2f}%")
print(f"  pixels below -3 dB                 : {100*(x<-3).mean():5.2f}%")

# -------------------------------------------------- metrics helper
def metrics(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, iou=iou)

# ------------------------------------------- physical screening + predictions
screen_ok = (slope <= 15.0) & (hand <= 50.0)          # flooding plausible
print(f"\nPHYSICAL SCREENING (slope <= 15 deg AND HAND <= 50 m)")
print(f"  plausible over valid domain       : {100*screen_ok[V].mean():5.2f}%")
print(f"  screened OUT                      : {100*(~screen_ok[V]).mean():5.2f}%")
print(f"  of screened-out px, truly flooded : "
      f"{100*(lab[V & ~screen_ok]==1).mean():5.2f}%")

rows = []
for name, t, scr in (("Otsu, no screening", thr, False),
                     ("Otsu + screening", thr, True),
                     ("-3 dB literature, no screening", -3.0, False),
                     ("-3 dB literature + screening", -3.0, True),
                     ("-6 dB strict + screening", -6.0, True)):
    p = (dvv < t)
    if scr: p = p & screen_ok
    m = metrics(p[V], y)
    m["name"] = name; m["thr"] = t
    rows.append(m)

print("\n" + "=" * 72)
print("METRICS vs EMSR773 REFERENCE")
print("=" * 72)
print(f"{'variant':32s} {'thr':>7s} {'IoU':>7s} {'F1':>7s} {'prec':>7s} {'recall':>7s}")
print("-" * 72)
for m in rows:
    print(f"{m['name']:32s} {m['thr']:+7.2f} {m['iou']:7.4f} {m['f1']:7.4f} "
          f"{m['precision']:7.4f} {m['recall']:7.4f}")

best = rows[1]   # Otsu + screening is the saved baseline
print("\n" + "=" * 72)
print(f"CONFUSION MATRIX -- {best['name']} (threshold {best['thr']:+.3f} dB)")
print("=" * 72)
tot = best['tp']+best['fp']+best['fn']+best['tn']
print(f"{'':22s}{'pred FLOOD':>14s}{'pred DRY':>14s}{'total':>14s}")
print(f"{'actual FLOOD':22s}{best['tp']:14,}{best['fn']:14,}{best['tp']+best['fn']:14,}")
print(f"{'actual DRY':22s}{best['fp']:14,}{best['tn']:14,}{best['fp']+best['tn']:14,}")
print(f"{'total':22s}{best['tp']+best['fp']:14,}{best['fn']+best['tn']:14,}{tot:14,}")
print(f"\n  accuracy  {100*(best['tp']+best['tn'])/tot:.2f}%   "
      f"(majority-class baseline {100*(1-y.mean()):.2f}%)")
px = 100/1e6
print(f"  predicted flood area {(best['tp']+best['fp'])*px:.2f} km2  vs "
      f"reference {(best['tp']+best['fn'])*px:.2f} km2")

# ------------------------------------------------------------------ save map
out = np.full(lab.shape, 255, "uint8")
pred = (dvv < thr) & screen_ok
out[(lab != 255)] = 0
out[(lab != 255) & pred] = 1
c.write(c.PROC + "/flood_otsu.tif", out, "uint8", 255)
print(f"\n-> processed/flood_otsu.tif  "
      f"({int((out==1).sum()):,} flood px = {(out==1).sum()*px:.2f} km2)")

# per-land-cover recall, to show where the threshold works
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
print("\nPER-LAND-COVER PERFORMANCE (Otsu + screening)")
print(f"{'class':26s} {'n_flood':>9s} {'recall':>8s} {'prec':>8s} {'F1':>8s}")
for k in sorted(np.unique(wc[lab != 255])):
    m_ = (wc == k) & (lab != 255) & np.isfinite(dvv)
    if (lab[m_] == 1).sum() < 500: continue
    mm = metrics(pred[m_], (lab == 1)[m_])
    print(f"{str(k)+' '+c.WC_CLASSES.get(k,'?'):26s} {mm['tp']+mm['fn']:9,} "
          f"{mm['recall']:8.4f} {mm['precision']:8.4f} {mm['f1']:8.4f}")
np.save(c.PROC + "/_otsu_thr.npy", np.array([thr]))
