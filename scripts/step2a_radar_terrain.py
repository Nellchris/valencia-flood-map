"""STEP 2a: radar change detection + slope."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio
import common as c

pre_vv, post_vv = c.read(c.PROC+"/pre_vv.tif"), c.read(c.PROC+"/post_vv.tif")
pre_vh, post_vh = c.read(c.PROC+"/pre_vh.tif"), c.read(c.PROC+"/post_vh.tif")

print("RADAR CHANGE DETECTION (post dB - pre dB; negative = backscatter drop = water)")
for nm, a, b in (("vv", post_vv, pre_vv), ("vh", post_vh, pre_vh)):
    d = b * 0 + np.nan
    m = np.isfinite(a) & np.isfinite(b)
    d[m] = a[m] - b[m]
    out = np.where(np.isfinite(d), d, c.NODATA_F)
    c.write(f"{c.PROC}/delta_{nm}.tif", out, "float32", c.NODATA_F)
    v = d[m]
    print(f"  delta_{nm}.tif  valid {m.mean()*100:5.1f}%  "
          f"min {v.min():7.2f}  max {v.max():6.2f}  mean {v.mean():6.2f}  "
          f"sd {v.std():5.2f}  p1 {np.percentile(v,1):6.2f}  p99 {np.percentile(v,99):5.2f}")

# ---- slope from DEM (Horn 3x3, 10 m grid)
dem = c.read(c.PROC+"/dem.tif")
z = np.where(np.isfinite(dem), dem, 0.0).astype("float64")
dzdx = ((z[:,2:] - z[:,:-2])[1:-1,:]) / (2*c.RES)
dzdy = ((z[2:,:] - z[:-2,:])[:,1:-1]) / (2*c.RES)
slope = np.zeros(z.shape, "float32")
slope[1:-1,1:-1] = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")
slope[0,:]=slope[1,:]; slope[-1,:]=slope[-2,:]
slope[:,0]=slope[:,1]; slope[:,-1]=slope[:,-2]
c.write(c.PROC+"/slope.tif", slope, "float32", c.NODATA_F)
print(f"\nSLOPE  min {slope.min():.2f}  max {slope.max():.2f}  mean {slope.mean():.2f} deg")
for t in (2,5,10,15,30):
    print(f"   slope > {t:2d} deg : {100*(slope>t).mean():5.2f}% of grid")
