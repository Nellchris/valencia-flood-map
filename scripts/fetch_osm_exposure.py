"""STEP 7a: fetch OSM buildings and roads for the study area via Overpass.

The study box covers the Valencia metropolitan area, so a single Overpass call
for every building footprint would return hundreds of MB and time out. Two
mitigations:

  * the bbox is tiled into 0.1 x 0.1 deg cells and each cell is fetched (and
    cached to the scratchpad) separately, so a failure costs one tile, not the
    whole download;
  * buildings are requested with `out center`, i.e. one centroid per footprint
    instead of the full polygon ring. Exposure is a *count* of buildings, and a
    centroid is the standard primitive for point-in-flood counting, so the ring
    geometry would be ~10x the payload for no analytical gain.

Roads need real geometry (length in km), so they use `out geom`, restricted to
the drivable/trafficable network (footways, cycleways, paths and steps are
excluded -- they are not what "roads affected" means in an exposure report).
Ways crossing tile edges come back in every tile they touch and are
de-duplicated by OSM id.
"""
import os, sys, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import requests, numpy as np, geopandas as gpd
from shapely.geometry import Point, LineString
import common as c

CACHE = os.environ.get("OSM_CACHE", os.path.join(
    os.path.expanduser("~"), "AppData", "Local", "Temp", "osm_valencia"))
os.makedirs(CACHE, exist_ok=True)

W, S, E, N = c.BBOX_WGS[0], c.BBOX_WGS[1], c.BBOX_WGS[2], c.BBOX_WGS[3]
STEP = 0.1
ROAD_RE = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
           "service|living_street|track|road|"
           "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]


def tiles():
    out = []
    ys = np.arange(S, N - 1e-9, STEP)
    xs = np.arange(W, E - 1e-9, STEP)
    for y in ys:
        for x in xs:
            out.append((round(y, 3), round(x, 3),
                        round(min(y + STEP, N), 3), round(min(x + STEP, E), 3)))
    return out


def overpass(query, tag):
    """POST `query`, caching the raw JSON under `tag`."""
    path = os.path.join(CACHE, tag + ".json")
    if os.path.exists(path) and os.path.getsize(path) > 2:
        with open(path) as f:
            return json.load(f)
    last = None
    for url in ENDPOINTS:
        for attempt in range(3):
            try:
                r = requests.post(
                    url, data={"data": query},
                    headers={"User-Agent": "flood-mapping-research/1.0"},
                    timeout=300)
                r.raise_for_status()
                d = r.json()
                with open(path, "w") as f:
                    json.dump(d, f)
                return d
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
                print(f"      retry ({last})", flush=True)
                time.sleep(10)
    raise RuntimeError(f"Overpass failed for {tag}: {last}")


TL = tiles()
print(f"{len(TL)} tiles of {STEP} deg over bbox {c.BBOX_WGS}")

b_rows, r_rows = {}, {}
for i, (s, w, n, e) in enumerate(TL):
    bb = f"{s},{w},{n},{e}"
    print(f"[{i+1:2d}/{len(TL)}] tile {bb}", flush=True)

    q = (f'[out:json][timeout:600];\n'
         f'(way["building"]({bb});relation["building"]({bb}););\nout center;\n')
    d = overpass(q, f"bld_{s}_{w}")
    nb = 0
    for el in d.get("elements", []):
        ctr = el.get("center")
        if not ctr:
            continue
        key = (el["type"], el["id"])
        if key in b_rows:
            continue
        t = el.get("tags", {})
        b_rows[key] = {"osm_id": el["id"], "osm_type": el["type"],
                       "building": t.get("building"), "name": t.get("name"),
                       "geometry": Point(ctr["lon"], ctr["lat"])}
        nb += 1

    q = (f'[out:json][timeout:600];\n'
         f'way["highway"~"^({ROAD_RE})$"]({bb});\nout geom;\n')
    d = overpass(q, f"rd_{s}_{w}")
    nr = 0
    for el in d.get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        if el["id"] in r_rows:
            continue
        t = el.get("tags", {})
        pts = [(p["lon"], p["lat"]) for p in g if p]
        if len(pts) < 2:
            continue
        r_rows[el["id"]] = {"osm_id": el["id"], "highway": t.get("highway"),
                            "name": t.get("name"),
                            "bridge": t.get("bridge"),
                            "geometry": LineString(pts)}
        nr += 1
    print(f"          +{nb:,} new buildings, +{nr:,} new roads "
          f"(running {len(b_rows):,} / {len(r_rows):,})", flush=True)

# ------------------------------------------------------------------- write out
bld = gpd.GeoDataFrame(list(b_rows.values()), crs="EPSG:4326").to_crs(c.DST_CRS)
rds = gpd.GeoDataFrame(list(r_rows.values()), crs="EPSG:4326").to_crs(c.DST_CRS)

# clip to the processing grid so counts/lengths match the raster extent exactly
from shapely.geometry import box
grid = box(*c.BOUNDS)
nb0, nr0 = len(bld), len(rds)
bld = bld[bld.within(grid)].reset_index(drop=True)
rds["geometry"] = rds.geometry.intersection(grid)
rds = rds[~rds.geometry.is_empty & rds.geometry.notna()].explode(
    index_parts=False).reset_index(drop=True)
rds = rds[rds.geometry.geom_type == "LineString"].reset_index(drop=True)

bld.to_file(os.path.join(c.PROC, "buildings.gpkg"), layer="buildings",
            driver="GPKG")
rds.to_file(os.path.join(c.PROC, "roads.gpkg"), layer="roads", driver="GPKG")

print("\n" + "=" * 78)
print("OSM EXPOSURE LAYERS")
print("=" * 78)
print(f"buildings: {nb0:,} fetched -> {len(bld):,} inside the UTM30N grid")
print(f"roads    : {nr0:,} ways fetched -> {len(rds):,} line parts after "
      f"clipping to the grid")
print(f"road length inside grid: {rds.geometry.length.sum()/1000:,.1f} km")
print("\ntop building tags")
print(bld.building.value_counts().head(10).to_string())
print("\nroad length by class (km)")
g = rds.groupby("highway").geometry.apply(lambda s: s.length.sum() / 1000)
print(g.sort_values(ascending=False).to_string(float_format=lambda v: f"{v:,.1f}"))
print(f"\n-> processed/buildings.gpkg\n-> processed/roads.gpkg")
