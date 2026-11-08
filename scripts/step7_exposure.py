"""STEP 7: exposure of OSM buildings and roads to each flood footprint.

Five footprints, all evaluated on the same 10 m grid:

  lower      confidence == HIGH                      (p >= 0.90 + Otsu agreement)
  moderate   confidence in {HIGH, MEDIUM}            (p >= 0.80)
  central    flood_rf.tif == 1                       (p >  0.50)
  upper      central  UNION  confidence in {H,M,L}   (p >= 0.70 is inside central)
  reference  EMSR773 observed flood, rasterised to the same grid

IMPORTANT -- `upper` is degenerate. The original specification defined it as
"footprint + LOW confidence areas", which assumed the LOW band sat *below* the
p > 0.5 decision boundary (the fixed 0.3/0.5/0.8 thresholds). Step 6 replaced
those with data-driven thresholds of 0.70/0.80/0.90, so the entire LOW band now
lies *inside* the RF footprint and the union adds nothing: upper == central,
pixel for pixel. It is computed and reported as specified rather than silently
redefined, and an `upper_extended` row (p > 0.40, the next decile down) is added
as the nearest thing to the original intent, with the caveat that that band has
a 16.6% out-of-fold flood rate against a 15.4% base rate -- i.e. no skill.

Permanent water (WorldCover 80: the sea and the Albufera lagoon) is removed from
every footprint. It is standing water, not flood, and it was already excluded
from the Step 2 label domain; leaving it in would also admit the offshore
concentric banding artefact of the dist_river feature.

Road length is measured by densifying each line to 10 m segments and testing the
segment midpoint against the raster -- the same discretisation for all five
footprints, so the comparison is like for like. For the EMS reference the exact
vector-intersection length is also reported, as a check on how much the 10 m
discretisation costs.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio, geopandas as gpd, pandas as pd, shapely
from rasterio.features import rasterize
import common as c

PXKM2 = 100 / 1e6                      # km2 per 10 m pixel
SEG = 10.0                             # road densification step (m)

# ------------------------------------------------------------------- rasters
with rasterio.open(c.PROC + "/confidence.tif") as s: conf = s.read(1)
with rasterio.open(c.PROC + "/flood_rf.tif") as s: rf = s.read(1)
with rasterio.open(c.PROC + "/labels.tif") as s: lab = s.read(1)
with rasterio.open(c.PROC + "/landcover.tif") as s: wc = s.read(1)
prob = c.read(c.PROC + "/flood_prob_rf.tif")

ref = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="flood_reference")
cov = gpd.read_file(c.PROC + "/flood_reference.gpkg", layer="reference_coverage")
ref_r = rasterize([(g, 1) for g in ref.geometry], out_shape=(c.HEIGHT, c.WIDTH),
                  transform=c.TRANSFORM, fill=0, dtype="uint8",
                  all_touched=False).astype(bool)
cov_r = rasterize([(g, 1) for g in cov.geometry], out_shape=(c.HEIGHT, c.WIDTH),
                  transform=c.TRANSFORM, fill=0, dtype="uint8",
                  all_touched=False).astype(bool)

permwater = wc == 80
FOOT = {
    "lower (HIGH only)":          (conf == 1),
    "moderate (HIGH+MEDIUM)":     (conf >= 1) & (conf <= 2),
    "central (full RF footprint)": (rf == 1),
    "upper (footprint + LOW)":    (rf == 1) | ((conf >= 1) & (conf <= 3)),
    "upper_extended (p>0.40)":    np.isfinite(prob) & (prob > 0.40),
    "EMS reference (EMSR773)":    ref_r,
}
DEFN = {
    "lower (HIGH only)": "confidence==1  (p>=0.90 + Otsu agreement)",
    "moderate (HIGH+MEDIUM)": "confidence in {1,2}  (p>=0.80)",
    "central (full RF footprint)": "flood_rf==1  (p>0.50)",
    "upper (footprint + LOW)": "central UNION confidence in {1,2,3}; "
                               "IDENTICAL to central (LOW band is p>=0.70)",
    "upper_extended (p>0.40)": "RF probability >0.40 (added; original 'upper' "
                               "is degenerate). 16.6% OOF flood rate = no skill",
    "EMS reference (EMSR773)": "Copernicus EMS observed flood, rasterised to grid",
}
for k in FOOT:                                     # drop permanent water
    FOOT[k] = FOOT[k] & ~permwater

# ------------------------------------------------------------------- vectors
bld = gpd.read_file(c.PROC + "/buildings.gpkg", layer="buildings")
rds = gpd.read_file(c.PROC + "/roads.gpkg", layer="roads")
print(f"{len(bld):,} buildings, {len(rds):,} road parts "
      f"({rds.geometry.length.sum()/1000:,.1f} km) inside the grid")


def rc(x, y):
    """grid row/col for map coords (vectorised, no bounds check)."""
    col = np.floor((x - c.X0) / c.RES).astype(np.int64)
    row = np.floor((c.Y1 - y) / c.RES).astype(np.int64)
    return row, col


# building centroids -> grid index
bx = bld.geometry.x.to_numpy(); by = bld.geometry.y.to_numpy()
brow, bcol = rc(bx, by)
bok = (brow >= 0) & (brow < c.HEIGHT) & (bcol >= 0) & (bcol < c.WIDTH)
brow, bcol = brow[bok], bcol[bok]
print(f"  {bok.sum():,} building centroids land on the grid")

# roads -> 10 m segment midpoints with lengths
seg = shapely.segmentize(rds.geometry.values, SEG)
co, idx = shapely.get_coordinates(seg, return_index=True)
same = idx[:-1] == idx[1:]
x0, y0 = co[:-1][same].T
x1, y1 = co[1:][same].T
mlen = np.hypot(x1 - x0, y1 - y0)
mrow, mcol = rc((x0 + x1) / 2, (y0 + y1) / 2)
mok = (mrow >= 0) & (mrow < c.HEIGHT) & (mcol >= 0) & (mcol < c.WIDTH)
mrow, mcol, mlen = mrow[mok], mcol[mok], mlen[mok]
mcls = rds.highway.to_numpy()[idx[:-1][same]][mok]
print(f"  {len(mlen):,} road segments, {mlen.sum()/1000:,.1f} km on the grid")

# ------------------------------------------------------------------- tally
rows, cls_rows = [], []
tot_b, tot_r = int(bok.sum()), mlen.sum() / 1000
for name, m in FOOT.items():
    hit_b = m[brow, bcol]
    hit_r = m[mrow, mcol]
    nb = int(hit_b.sum())
    km = float(mlen[hit_r].sum() / 1000)
    area = float(m.sum()) * PXKM2
    out_cov = float((m & ~cov_r).sum()) / max(m.sum(), 1)
    rows.append(dict(estimate=name, definition=DEFN[name],
                     area_km2=round(area, 2),
                     n_buildings=nb,
                     pct_of_all_buildings=round(100 * nb / tot_b, 3),
                     road_km=round(km, 2),
                     pct_of_all_road_km=round(100 * km / tot_r, 3),
                     buildings_per_km2=round(nb / area, 1) if area else np.nan,
                     frac_outside_ems_coverage=round(out_cov, 4)))
    for k, g in pd.DataFrame({"cls": mcls, "L": mlen,
                              "hit": hit_r}).groupby("cls"):
        cls_rows.append(dict(estimate=name, highway=k,
                             road_km=round(g.L[g.hit].sum() / 1000, 3),
                             total_km=round(g.L.sum() / 1000, 3)))

df = pd.DataFrame(rows)
df.to_csv(c.PROC + "/exposure_results.csv", index=False)
pd.DataFrame(cls_rows).to_csv(c.PROC + "/exposure_by_roadclass.csv", index=False)

# exact vector length for the reference, as a discretisation check
ref_u = ref.geometry.union_all()
exact = rds.geometry.intersection(ref_u).length.sum() / 1000
exact_b = int(bld.within(ref_u).sum())

print("\n" + "=" * 104)
print("EXPOSURE ESTIMATES -- OSM buildings and roads inside each flood footprint")
print("=" * 104)
print(f"{'estimate':28s} {'area km2':>9s} {'buildings':>10s} {'% all':>7s} "
      f"{'road km':>9s} {'% all':>7s} {'bld/km2':>8s}")
print("-" * 104)
for r in rows:
    print(f"{r['estimate']:28s} {r['area_km2']:9.2f} {r['n_buildings']:10,} "
          f"{r['pct_of_all_buildings']:6.2f}% {r['road_km']:9.2f} "
          f"{r['pct_of_all_road_km']:6.2f}% {r['buildings_per_km2']:8.1f}")
print("-" * 104)
print(f"{'STUDY AREA TOTAL':28s} {c.WIDTH*c.HEIGHT*PXKM2:9.2f} {tot_b:10,} "
      f"{100.0:6.2f}% {tot_r:9.2f} {100.0:6.2f}%")

print("\nREFERENCE discretisation check (10 m raster vs exact vector overlay)")
rr = df[df.estimate.str.startswith("EMS")].iloc[0]
print(f"  buildings : raster {rr.n_buildings:,}   exact {exact_b:,}   "
      f"diff {rr.n_buildings-exact_b:+,}")
print(f"  road km   : raster {rr.road_km:,.2f}   exact {exact:,.2f}   "
      f"diff {rr.road_km-exact:+.2f} km "
      f"({100*(rr.road_km-exact)/exact:+.2f}%)")

# ---------------------------------------- ratio of each estimate to reference
print("\n" + "=" * 86)
print("EACH ESTIMATE AGAINST THE EMS REFERENCE BENCHMARK")
print("=" * 86)
rb, rk, ra = rr.n_buildings, rr.road_km, rr.area_km2
print(f"{'estimate':28s} {'bld ratio':>10s} {'road ratio':>11s} {'area ratio':>11s}")
print("-" * 86)
for r in rows:
    if r["estimate"].startswith("EMS"): continue
    print(f"{r['estimate']:28s} {r['n_buildings']/rb:10.3f} "
          f"{r['road_km']/rk:11.3f} {r['area_km2']/ra:11.3f}")

# --------------------------------------------- affected assets by confidence
print("\n" + "=" * 86)
print("AFFECTED ASSETS BY CONFIDENCE CLASS  (within the RF footprint)")
print("=" * 86)
NM = {1: "HIGH", 2: "MEDIUM", 3: "LOW", 0: "below 0.70 (not called flood)"}
cb = conf[brow, bcol]; cr = conf[mrow, mcol]
pw_b = permwater[brow, bcol]; pw_r = permwater[mrow, mcol]
rfb = rf[brow, bcol] == 1; rfr = rf[mrow, mcol] == 1
print(f"{'class':34s} {'buildings':>10s} {'road km':>10s}")
print("-" * 86)
for k in (1, 2, 3, 0):
    mb = (cb == k) & ~pw_b
    mr = (cr == k) & ~pw_r
    lbl = NM[k]
    if k == 0:                       # only the part still called flood by RF
        mb &= rfb; mr &= rfr
        lbl = "0.50-0.70 (RF flood, no conf class)"
    print(f"{lbl:34s} {int(mb.sum()):10,} {float(mlen[mr].sum()/1000):10.2f}")

# ------------------------------------------------- land cover of affected bld
print("\n" + "=" * 86)
print("BUILDINGS IN EACH FOOTPRINT BY WORLDCOVER CLASS OF THEIR PIXEL")
print("=" * 86)
wcb = wc[brow, bcol]
codes = [10, 20, 30, 40, 50, 60, 90]
hdr = f"{'estimate':28s}" + "".join(f"{str(k):>9s}" for k in codes)
print(hdr); print("-" * len(hdr))
for name, m in FOOT.items():
    h = m[brow, bcol]
    print(f"{name:28s}" + "".join(
        f"{int((h & (wcb == k)).sum()):>9,}" for k in codes))

print("\n-> processed/exposure_results.csv")
print("-> processed/exposure_by_roadclass.csv")
print("done")

# ------------------------------------------------- asset-level confusion
# The four estimates are only comparable to the reference where the reference
# actually looked. Restrict to the unioned EMS areas-of-interest and score each
# asset as TP / FP / FN against the observed-flood delineation.
print("\n" + "=" * 100)
print("ASSET-LEVEL AGREEMENT WITH THE EMS REFERENCE (inside EMS AOI coverage only)")
print("=" * 100)
cb_ok = cov_r[brow, bcol] & ~permwater[brow, bcol]
cr_ok = cov_r[mrow, mcol] & ~permwater[mrow, mcol]
ref_b = ref_r[brow, bcol]; ref_m = ref_r[mrow, mcol]
print(f"  in-coverage assets: {int(cb_ok.sum()):,} buildings, "
      f"{mlen[cr_ok].sum()/1000:,.1f} km road")
print(f"  of which reference-flooded: {int((cb_ok & ref_b).sum()):,} buildings, "
      f"{mlen[cr_ok & ref_m].sum()/1000:,.1f} km road")
print(f"\n{'estimate':28s} {'TP bld':>8s} {'FP bld':>8s} {'FN bld':>8s} "
      f"{'prec':>6s} {'rec':>6s} {'F1':>6s} | "
      f"{'TP km':>8s} {'FP km':>8s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
print("-" * 100)
conf_rows = []
for name, m in FOOT.items():
    if name.startswith("EMS"): continue
    pb = m[brow, bcol] & cb_ok
    pm = m[mrow, mcol] & cr_ok
    tp = int((pb & ref_b).sum()); fp = int((pb & ~ref_b).sum())
    fn = int((~pb & ref_b & cb_ok).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rcl = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rcl / (pr + rcl) if pr + rcl else 0.0
    tk = mlen[pm & ref_m].sum() / 1000
    fk = mlen[pm & ~ref_m].sum() / 1000
    nk = mlen[~pm & ref_m & cr_ok].sum() / 1000
    pr2 = tk / (tk + fk) if tk + fk else 0.0
    rc2 = tk / (tk + nk) if tk + nk else 0.0
    f2 = 2 * pr2 * rc2 / (pr2 + rc2) if pr2 + rc2 else 0.0
    print(f"{name:28s} {tp:8,} {fp:8,} {fn:8,} {pr:6.3f} {rcl:6.3f} {f1:6.3f} | "
          f"{tk:8.1f} {fk:8.1f} {pr2:6.3f} {rc2:6.3f} {f2:6.3f}")
    conf_rows.append(dict(estimate=name, bld_tp=tp, bld_fp=fp, bld_fn=fn,
                          bld_precision=round(pr, 4), bld_recall=round(rcl, 4),
                          bld_f1=round(f1, 4), road_tp_km=round(tk, 2),
                          road_fp_km=round(fk, 2), road_fn_km=round(nk, 2),
                          road_precision=round(pr2, 4),
                          road_recall=round(rc2, 4), road_f1=round(f2, 4)))
pd.DataFrame(conf_rows).to_csv(c.PROC + "/exposure_vs_reference.csv", index=False)
print("\n-> processed/exposure_vs_reference.csv")
