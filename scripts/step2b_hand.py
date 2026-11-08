"""STEP 2b: HAND (Height Above Nearest Drainage) via pysheds D8."""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, rasterio
# pysheds 0.5 predates NumPy 2.0: restore the aliases it still calls.
for _old, _new in (("in1d", "isin"), ("alltrue", "all"), ("product", "prod"),
                   ("sometrue", "any"), ("cumproduct", "cumprod")):
    if not hasattr(np, _old):
        setattr(np, _old, getattr(np, _new))
for _old, _new in (("float_", "float64"), ("bool8", "bool_"), ("unicode_", "str_")):
    if not hasattr(np, _old):
        setattr(np, _old, getattr(np, _new))
from pysheds.grid import Grid
from pysheds.view import Raster
import common as c

SCRATCH = r"C:/Users/USER/AppData/Local/Temp/claude/555e72459c0f9e15/163e5b5e-6183-47c4-b8ba-b6c4f231037c/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)
ACC_THRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

# DEM clamped at 0 m: coastal/negative elevations otherwise create sinks that
# break D8 flow routing.
dem = c.read(c.PROC + "/dem.tif")
z = np.where(np.isfinite(dem), dem, 0.0)
print(f"DEM clamp: {(z < 0).sum():,} px below 0 m -> 0 "
      f"({100*(z<0).mean():.2f}% of grid, min was {z.min():.2f} m)")
z = np.maximum(z, 0.0).astype("float32")
tmp = os.path.join(SCRATCH, "dem_clamped.tif")
c.write(tmp, z, "float32", c.NODATA_F)

t = time.time()
grid = Grid.from_raster(tmp)
dr = grid.read_raster(tmp)
print("fill_pits...", flush=True);        dr = grid.fill_pits(dr)
print("fill_depressions...", flush=True); dr = grid.fill_depressions(dr)
print("resolve_flats...", flush=True);    dr = grid.resolve_flats(dr)
print("flowdir...", flush=True);          fdir = grid.flowdir(dr)
print("accumulation...", flush=True);     acc = grid.accumulation(fdir)
print(f"  conditioning+routing done in {time.time()-t:.0f}s")

accv = np.asarray(acc)
print(f"  accumulation max {accv.max():,.0f} cells "
      f"({accv.max()*100/1e6:.1f} km2 contributing)")
for th in (1000, 2000, 5000, 10000, 20000):
    n = int((accv > th).sum())
    print(f"   acc > {th:6d} cells ({th*100/1e6:5.2f} km2): "
          f"{n:8,} drainage px, {n*10/1000:7.1f} km of network")

t = time.time()
# Drainage mask = channel network OR permanent water (the coast/lagoon is the
# ultimate drainage base level here) OR the grid border (cells routing off-grid).
with rasterio.open(c.PROC + "/landcover.tif") as _s:
    _wc = _s.read(1)
border = np.zeros(accv.shape, bool)
border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
drain = (accv > ACC_THRESH) | (_wc == 80) | border
print(f"  drainage mask: {drain.sum():,} px "
      f"(network {int((accv>ACC_THRESH).sum()):,}, "
      f"permanent water {int((_wc==80).sum()):,}, border {int(border.sum()):,})")
mask = Raster(drain.astype(bool), viewfinder=fdir.viewfinder)
print(f"\ncompute_hand (threshold {ACC_THRESH} cells = "
      f"{ACC_THRESH*100/1e6:.2f} km2)...", flush=True)
hand = grid.compute_hand(fdir=fdir, dem=dr, mask=mask)
hv = np.asarray(hand).astype("float32")
print(f"  done in {time.time()-t:.0f}s")

finite = np.isfinite(hv)
print(f"  HAND valid {100*finite.mean():.2f}%  "
      f"min {hv[finite].min():.2f}  max {hv[finite].max():.2f}  "
      f"median {np.median(hv[finite]):.2f} m")
out = np.where(finite, np.maximum(hv, 0.0), np.nan).astype("float32")
# Residual gaps (cells whose D8 path still terminates without reaching the mask):
# fill from the nearest valid HAND cell so the feature is complete.
if (~np.isfinite(out)).any():
    from scipy.ndimage import distance_transform_edt
    gap = ~np.isfinite(out)
    idx = distance_transform_edt(gap, return_distances=False, return_indices=True)
    print(f"  nearest-neighbour fill of {gap.sum():,} residual gap px "
          f"({100*gap.mean():.2f}% of grid)")
    out = out[tuple(idx)]
out = np.where(np.isfinite(out), out, 0.0).astype("float32")
c.write(c.PROC + "/hand.tif", out, "float32", c.NODATA_F)
finite = np.ones(out.shape, bool)
np.save(os.path.join(SCRATCH, "acc.npy"), accv)
for t_ in (1, 2, 5, 10, 25, 50):
    print(f"   HAND < {t_:3d} m : {100*(out[finite] < t_).mean():5.2f}% of valid")
print("\n-> processed/hand.tif")
