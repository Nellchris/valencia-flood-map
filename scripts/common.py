"""Shared grid definition and helpers for the Valencia flood mapping pipeline."""
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

BASE = r"C:/Users/USER/Documents/Flood Mapping"
PROC = os.path.join(BASE, "processed")
WEBMAP = os.path.join(BASE, "webmap")

# Study area bounding box (geographic, as specified)
BBOX_WGS = (-0.6, 39.3, -0.2, 39.6)  # W, S, E, N

# Canonical processing grid: EPSG:32630, 10 m.
# Largest axis-aligned UTM30N rectangle fully INSCRIBED in the geographic bbox,
# snapped to a 10 m grid. Inscribed (not the outer envelope) so that every input
# layer -- in particular the DEM, which is clipped exactly to the geographic
# bbox -- has valid data across the entire grid.
DST_CRS = "EPSG:32630"
RES = 10.0
X0, Y1 = 706960.0, 4386110.0   # top-left (min x, max y)
WIDTH, HEIGHT = 3345, 3230
TRANSFORM = from_origin(X0, Y1, RES, RES)
X1 = X0 + WIDTH * RES
Y0 = Y1 - HEIGHT * RES
BOUNDS = (X0, Y0, X1, Y1)

NODATA_F = -9999.0
DB_FLOOR = -40.0

# ESA WorldCover 10 m v200 class codes
WC_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse veg", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss and lichen",
}

S1_PRE = "S1A_IW_20241020T060207_DVP_RTC10_G_gpunem_6499"
S1_POST = "S1A_IW_20241101T060207_DVP_RTC10_G_gpunem_758C"


def s1_path(scene, pol):
    return f"zip://{BASE}/{scene}.zip!/{scene}/{scene}_{pol}.tif"


WC_PATH = (f"zip://{BASE}/terrascope_download_20261104_140228.zip!/WORLDCOVER/"
           "ESA_WORLDCOVER_10M_2021_V200/MAP/"
           "ESA_WorldCover_10m_2021_v200_N39W003_Map/"
           "ESA_WorldCover_10m_2021_v200_N39W003_Map.tif")
DEM_PATH = f"{BASE}/GLO30_DEM_N393_396_W06_02.tif"

EMS = {
    "AOI01": ("EMSR773_AOI01_DEL_PRODUCT_v1.zip", "EMSR773_AOI01_DEL_PRODUCT", "v1"),
    "AOI02": ("EMSR773_AOI02_DEL_PRODUCT_v2.zip", "EMSR773_AOI02_DEL_PRODUCT", "v1"),
    "AOI03": ("EMSR773_AOI03_DEL_PRODUCT_v1.zip", "EMSR773_AOI03_DEL_PRODUCT", "v1"),
}

# src_id -> sensor, read from each AOI's source_v1.dbf (see Step 1 report)
SENSOR_MAP = {
    "AOI01": {1: "Sentinel-2 (pre)", 2: "Landsat-8", 3: "Sentinel-2", 4: "Sentinel-1"},
    "AOI02": {1: "WorldView-2 (pre)", 2: "GeoEye-1", 3: "GeoEye-1"},
    "AOI03": {1: "ESRI World Imagery (pre)", 2: "GeoEye-1", 3: "WorldView-2"},
}


def ems_layer(aoi, layer):
    z, stem, v = EMS[aoi]
    return f"zip://{BASE}/{z}!/{stem}_{layer}_{v}.shp"


def profile(dtype, nodata, count=1):
    return dict(driver="GTiff", height=HEIGHT, width=WIDTH, count=count,
                dtype=dtype, crs=DST_CRS, transform=TRANSFORM, nodata=nodata,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                predictor=2 if dtype != "uint8" else 1)


def write(path, arr, dtype, nodata):
    with rasterio.open(path, "w", **profile(dtype, nodata)) as d:
        d.write(arr.astype(dtype), 1)
    return path


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
        nd = s.nodata
    if nd is not None:
        a = np.where(a == nd, np.nan, a)
    return a


def describe(path):
    with rasterio.open(path) as s:
        a = s.read(1)
        nd = s.nodata
        if np.issubdtype(a.dtype, np.floating):
            m = np.isfinite(a) & ((a != nd) if nd is not None else True)
        else:
            m = (a != nd) if nd is not None else np.ones(a.shape, bool)
        v = a[m]
        return dict(
            file=os.path.basename(path), shape=f"{s.height}x{s.width}",
            crs=str(s.crs), res=f"{s.res[0]:g} m", dtype=str(a.dtype),
            nodata=str(nd), valid_pct=100.0 * m.sum() / m.size,
            vmin=float(v.min()) if v.size else float("nan"),
            vmax=float(v.max()) if v.size else float("nan"),
            vmean=float(v.mean()) if v.size else float("nan"),
        )
