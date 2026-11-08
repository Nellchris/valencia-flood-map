"""Fetch OSM waterways for the study area via Overpass API."""
import os, sys, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import requests, geopandas as gpd
from shapely.geometry import LineString
import common as c

Q = """[out:json][timeout:180];
way["waterway"~"river|stream|canal|drain"](39.3,-0.6,39.6,-0.2);
out geom;
"""
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
data = None
for url in ENDPOINTS:
    for attempt in range(3):
        try:
            print(f"POST {url} (attempt {attempt+1})", flush=True)
            r = requests.post(url, data={"data": Q},
                              headers={"User-Agent": "flood-mapping-research/1.0"},
                              timeout=240)
            r.raise_for_status()
            data = r.json(); break
        except Exception as e:
            print("   failed:", type(e).__name__, str(e)[:160], flush=True)
            time.sleep(8)
    if data: break
if not data:
    sys.exit("Overpass unreachable from all endpoints")

rows = []
for el in data.get("elements", []):
    g = el.get("geometry")
    if not g or len(g) < 2: continue
    rows.append({"osm_id": el["id"],
                 "waterway": el.get("tags", {}).get("waterway"),
                 "name": el.get("tags", {}).get("name"),
                 "geometry": LineString([(p["lon"], p["lat"]) for p in g])})
gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(c.DST_CRS)
out = os.path.join(c.PROC, "waterways.gpkg")
gdf.to_file(out, layer="waterways", driver="GPKG")
print(f"\n{len(gdf)} waterway lines -> {out}")
print(gdf.waterway.value_counts().to_string())
print(f"total length: {gdf.geometry.length.sum()/1000:.1f} km")
