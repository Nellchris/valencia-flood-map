"""STEP 6b: confidence layer with data-driven thresholds.

Usage: step6b_confidence.py T_LOW T_MED T_HIGH

Confidence expresses trust in the FLOOD call:
    1 = high, 2 = medium, 3 = low, 0 = confidently not flooded, 255 = nodata.

Classes are assigned from the probability band, then demoted:
    - Otsu disagrees with the RF binary call -> at best medium
    - slope > 5 deg on an RF flood pixel    -> low
Error rates are reported both in-sample (optimistic) and out-of-fold
(spatial 2 km, honest).
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, pandas as pd
import common as c

T_LOW, T_MED, T_HIGH = (float(sys.argv[1]), float(sys.argv[2]),
                        float(sys.argv[3]))
print(f"thresholds: low>={T_LOW}  med>={T_MED}  high>={T_HIGH}")

with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
with rasterio.open(c.PROC + "/flood_otsu.tif") as s: otsu = s.read(1)
prob_in = c.read(c.PROC + "/flood_prob_rf.tif")
prob_oof = c.read(c.PROC + "/oof_prob_spatial2km.tif")
slope = c.read(c.PROC + "/slope.tif")
Y = lab == 1
VALID = lab != 255


def classify(prob, domain):
    """Return confidence class raster (0/1/2/3, 255 nodata) for `domain`."""
    conf = np.full(lab.shape, 255, "uint8")
    rf_flood = prob > 0.5
    otsu_flood = otsu == 1
    agree = rf_flood == otsu_flood
    conf[domain] = 0                                   # confidently dry
    band_low = domain & (prob >= T_LOW) & (prob < T_MED)
    band_med = domain & (prob >= T_MED) & (prob < T_HIGH)
    band_high = domain & (prob >= T_HIGH)
    conf[band_low] = 3
    conf[band_med] = 2
    conf[band_high] = 1
    # demotions
    conf[(conf == 1) & ~agree] = 2                     # Otsu disagrees
    conf[(conf <= 2) & (conf >= 1) & (slope > 5) & rf_flood] = 3
    conf[band_low & (slope > 5) & rf_flood] = 3
    conf[~domain] = 255
    return conf


conf_in = classify(prob_in, VALID)
oof_dom = VALID & np.isfinite(prob_oof)
conf_oof = classify(prob_oof, oof_dom)

# the deliverable raster is the full-coverage in-sample one
full_dom = np.isfinite(prob_in)
c.write(c.PROC + "/confidence.tif", classify(prob_in, full_dom), "uint8", 255)
print("-> processed/confidence.tif")

NAMES = {1: "HIGH", 2: "MEDIUM", 3: "LOW", 0: "not flooded"}
px = 100 / 1e6


def err_table(conf, dom, title):
    print("\n" + "=" * 92)
    print(f"ERROR RATE BY CONFIDENCE CLASS -- {title}")
    print("=" * 92)
    print(f"{'class':14s} {'n_px':>11s} {'area km2':>10s} {'%pred':>7s} "
          f"{'true flood%':>12s} {'ERROR':>8s} {'precision':>10s}")
    print("-" * 92)
    out = []
    pred_area = int(((conf >= 1) & (conf <= 3)).sum())
    for k in (1, 2, 3, 0):
        m = dom & (conf == k)
        n = int(m.sum())
        if not n: continue
        tf = float(Y[m].mean())
        # error = fraction misclassified if this class is taken as flood
        # (for class 0 the call is 'dry', so error is the flooded fraction)
        err = (1 - tf) if k in (1, 2, 3) else tf
        print(f"{NAMES[k]:14s} {n:>11,} {n*px:10.2f} "
              f"{100*n/pred_area if k else float('nan'):6.1f}% "
              f"{100*tf:11.2f}% {err:8.4f} {tf if k else float('nan'):10.4f}")
        out.append(dict(set=title, confidence=NAMES[k], n=n, area_km2=n * px,
                        true_flood_rate=tf, error_rate=err))
    return out


rows = []
rows += err_table(conf_in, VALID, "IN-SAMPLE (optimistic)")
rows += err_table(conf_oof, oof_dom, "OUT-OF-FOLD spatial 2 km (honest)")

# ------------------------------------------- by confidence x land cover
print("\n" + "=" * 100)
print("ERROR RATE BY CONFIDENCE x WORLDCOVER CLASS  (out-of-fold, honest)")
print("=" * 100)
hdr = f"{'land cover':26s}" + "".join(f"{NAMES[k]:>22s}" for k in (1, 2, 3))
print(hdr); print("-" * len(hdr))
print(f"{'':26s}" + "".join(f"{'n':>9s}{'err':>7s}{'flood%':>6s}"
                            for _ in (1, 2, 3)))
lc_rows = []
for code in sorted(np.unique(wc[oof_dom])):
    line = f"{str(code)+' '+c.WC_CLASSES.get(code,'?'):26s}"
    for k in (1, 2, 3):
        m = oof_dom & (conf_oof == k) & (wc == code)
        n = int(m.sum())
        if n < 200:
            line += f"{n:>9,}{'-':>7s}{'-':>6s}"
            continue
        tf = float(Y[m].mean()); err = 1 - tf
        line += f"{n:>9,}{err:7.3f}{100*tf:6.1f}"
        lc_rows.append(dict(landcover=f"{code} {c.WC_CLASSES.get(code,'?')}",
                            confidence=NAMES[k], n=n, error_rate=err,
                            true_flood_rate=tf))
    print(line)

pd.DataFrame(rows).to_csv(c.PROC + "/confidence_summary.csv", index=False)
pd.DataFrame(lc_rows).to_csv(c.PROC + "/confidence_by_landcover.csv",
                             index=False)
print("\n-> processed/confidence_summary.csv")
print("-> processed/confidence_by_landcover.csv")

# monotonicity check
print("\nMONOTONICITY CHECK (error should rise HIGH -> MEDIUM -> LOW)")
for title, conf, dom in (("in-sample", conf_in, VALID),
                         ("out-of-fold", conf_oof, oof_dom)):
    e = [1 - float(Y[dom & (conf == k)].mean()) for k in (1, 2, 3)]
    ok = e[0] <= e[1] <= e[2]
    print(f"  {title:12s} HIGH {e[0]:.3f} -> MED {e[1]:.3f} -> LOW {e[2]:.3f}"
          f"   {'OK' if ok else '** NOT MONOTONIC **'}")
np.save(c.PROC + "/_conf_thr.npy", np.array([T_LOW, T_MED, T_HIGH]))
print("done")
