"""STEP 1: Data preparation and validation."""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import geopandas as gpd
import pyogrio
from shapely.geometry import box
from shapely.ops import unary_union

import common as c

os.makedirs(c.PROC, exist_ok=True)
t0 = time.time()


def warp_to_grid(src_path, resampling, src_nodata=None):
    """Read any source onto the canonical grid via a WarpedVRT (windowed reads)."""
    with rasterio.open(src_path) as src:
        vrt_opts = dict(crs=c.DST_CRS, transform=c.TRANSFORM,
                        width=c.WIDTH, height=c.HEIGHT, resampling=resampling)
        if src_nodata is not None:
            vrt_opts["src_nodata"] = src_nodata
            vrt_opts["nodata"] = src_nodata
        with WarpedVRT(src, **vrt_opts) as vrt:
            return vrt.read(1)


# ---------------------------------------------------------------- Sentinel-1
print("=" * 78)
print("SENTINEL-1 RTC  (gamma-0 POWER scale per filename code 'gp..'; EPSG:32631)")
print("=" * 78)
s1_stats = {}
for label, scene in (("pre", c.S1_PRE), ("post", c.S1_POST)):
    for pol in ("VV", "VH"):
        t = time.time()
        pw = warp_to_grid(c.s1_path(scene, pol), Resampling.bilinear, src_nodata=0.0)
        # power -> dB, with floor; source nodata is 0.0 (invalid power)
        valid = np.isfinite(pw) & (pw > 0)
        db = np.full(pw.shape, c.NODATA_F, dtype="float32")
        db[valid] = np.maximum(10.0 * np.log10(pw[valid]), c.DB_FLOOR)
        out = os.path.join(c.PROC, f"{label}_{pol.lower()}.tif")
        c.write(out, db, "float32", c.NODATA_F)
        s1_stats[f"{label}_{pol}"] = (valid.mean() * 100,
                                      float(db[valid].min()), float(db[valid].max()),
                                      float(db[valid].mean()))
        print(f"  {label:4s} {pol}  -> {os.path.basename(out):12s} "
              f"valid {valid.mean()*100:5.1f}%  dB range "
              f"[{db[valid].min():7.2f},{db[valid].max():7.2f}]  mean {db[valid].mean():6.2f}"
              f"   ({time.time()-t:.0f}s)")

# ---------------------------------------------------------------------- DEM
print("\nDEM (Copernicus GLO-30, EPSG:4326 -> EPSG:32630, bilinear)")
dem = warp_to_grid(c.DEM_PATH, Resampling.bilinear).astype("float32")
dem_bad = ~np.isfinite(dem)
dem[dem_bad] = c.NODATA_F
c.write(os.path.join(c.PROC, "dem.tif"), dem, "float32", c.NODATA_F)
dv = dem[dem != c.NODATA_F]
print(f"  dem.tif  elevation {dv.min():.1f} .. {dv.max():.1f} m  "
      f"mean {dv.mean():.1f}  invalid {dem_bad.mean()*100:.3f}%")

# --------------------------------------------------------------- WorldCover
print("\nWORLDCOVER (ESA 10 m 2021 v200, EPSG:4326 -> EPSG:32630, nearest)")
wc = warp_to_grid(c.WC_PATH, Resampling.nearest, src_nodata=0).astype("uint8")
c.write(os.path.join(c.PROC, "landcover.tif"), wc, "uint8", 0)
tot = wc.size
print(f"  landcover.tif  classes present:")
for k, n in sorted([(k, int((wc == k).sum())) for k in np.unique(wc)], key=lambda x: -x[1]):
    print(f"     {k:4d} {c.WC_CLASSES.get(k,'nodata/unknown'):24s} "
          f"{n:10d} px  {100*n/tot:6.2f}%")

# ----------------------------------------------------- EMSR773 flood vectors
print("\n" + "=" * 78)
print("EMSR773 FLOOD DELINEATION  (observedEventA)")
print("=" * 78)
bb_wgs = box(*[c.BBOX_WGS[i] for i in (0, 1, 2, 3)])
grid_box = box(*c.BOUNDS)
parts, aoi_foot, prov = [], [], []
for aoi in ("AOI01", "AOI02", "AOI03"):
    g = gpd.read_file(c.ems_layer(aoi, "observedEventA"))
    smap = c.SENSOR_MAP[aoi]
    g["aoi"] = aoi
    g["sensor"] = g["dmg_src_id"].map(smap).fillna("unknown")
    g = g.to_crs(c.DST_CRS)
    g = g[g.intersects(grid_box)]
    if len(g):
        g = gpd.clip(g, grid_box)
        g = g[~g.geometry.is_empty & g.geometry.notna()]
    ao = gpd.read_file(c.ems_layer(aoi, "areaOfInterestA")).to_crs(c.DST_CRS)
    ao_in = gpd.clip(ao, grid_box)
    print(f"\n  {aoi}: {len(g)} flood polygons inside study grid; "
          f"AOI footprint overlaps grid: {not ao_in.empty}")
    if len(g):
        for sensor, sub in g.groupby("sensor"):
            a = sub.geometry.area.sum() / 1e6
            print(f"       {sensor:28s} {len(sub):5d} polys  {a:8.2f} km2")
            prov.append((aoi, sensor, len(sub), a))
        parts.append(g[["aoi", "sensor", "dmg_src_id", "obj_desc",
                        "det_method", "geometry"]])
    if not ao_in.empty:
        aoi_foot.append(ao_in.assign(aoi=aoi)[["aoi", "geometry"]])

allparts = gpd.GeoDataFrame(__import__("pandas").concat(parts, ignore_index=True),
                            crs=c.DST_CRS)
# Dissolve overlapping AOI01/AOI03 polygons into one non-double-counted footprint
diss = unary_union(allparts.geometry.values)
flood = gpd.GeoDataFrame(geometry=[diss], crs=c.DST_CRS).explode(index_parts=False)
flood = flood.reset_index(drop=True)
flood["area_m2"] = flood.geometry.area

cov = gpd.GeoDataFrame(__import__("pandas").concat(aoi_foot, ignore_index=True),
                       crs=c.DST_CRS)
cov_u = gpd.GeoDataFrame(geometry=[unary_union(cov.geometry.values)], crs=c.DST_CRS)

gpkg = os.path.join(c.PROC, "flood_reference.gpkg")
if os.path.exists(gpkg):
    os.remove(gpkg)
flood.to_file(gpkg, layer="flood_reference", driver="GPKG")
allparts.to_file(gpkg, layer="flood_parts", driver="GPKG")
cov_u.to_file(gpkg, layer="reference_coverage", driver="GPKG")

tot_a = diss.area / 1e6
s1_a = unary_union(allparts[allparts.sensor == "Sentinel-1"].geometry.values).area / 1e6 \
    if (allparts.sensor == "Sentinel-1").any() else 0.0
print(f"\n  MERGED REFERENCE -> {os.path.basename(gpkg)}")
print(f"     dissolved parts      : {len(flood)}")
print(f"     input polygons       : {len(allparts)}")
print(f"     total flooded area   : {tot_a:.2f} km2")
print(f"     reference coverage   : {cov_u.geometry.area.sum()/1e6:.2f} km2 "
      f"({100*cov_u.geometry.area.sum()/grid_box.area:.1f}% of grid)")
print(f"     grid area            : {grid_box.area/1e6:.2f} km2")
print(f"\n  CIRCULARITY CHECK -- area of reference derived from Sentinel-1: "
      f"{s1_a:.2f} km2 ({100*s1_a/tot_a:.1f}% of reference)")

np.save(os.path.join(c.PROC, "_prov.npy"), np.array(prov, dtype=object),
        allow_pickle=True)

# ----------------------------------------------------------- Validation table
print("\n" + "=" * 78)
print("VALIDATION REPORT -- all processed rasters")
print("=" * 78)
hdr = f"{'file':16s} {'shape':12s} {'crs':11s} {'res':6s} {'dtype':8s} " \
      f"{'nodata':8s} {'valid%':7s} {'min':>9s} {'max':>9s} {'mean':>9s}"
print(hdr); print("-" * len(hdr))
for f in ("pre_vv.tif", "pre_vh.tif", "post_vv.tif", "post_vh.tif",
          "dem.tif", "landcover.tif"):
    d = c.describe(os.path.join(c.PROC, f))
    print(f"{d['file']:16s} {d['shape']:12s} {d['crs']:11s} {d['res']:6s} "
          f"{d['dtype']:8s} {d['nodata']:8s} {d['valid_pct']:6.2f}% "
          f"{d['vmin']:9.2f} {d['vmax']:9.2f} {d['vmean']:9.2f}")
print(f"\nAll rasters share CRS {c.DST_CRS}, {c.RES:g} m pixels, "
      f"{c.HEIGHT}x{c.WIDTH} grid, bounds {c.BOUNDS}")
print(f"\nDone in {time.time()-t0:.0f}s")
