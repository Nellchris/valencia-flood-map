# Valencia Flood Map — Sentinel-1, October 2024

An interactive radar flood map of the 29 October 2024 Valencia flash flood, built
from a Sentinel-1 image pair and a Random Forest classifier, with an explicit
confidence layer and an honest account of how accurate the map actually is. The
headline finding is methodological: under conventional random cross-validation the
classifier scores F1 = 0.505, but under spatially blocked cross-validation it
scores 0.419 — a 17% loss of apparent skill and a 62× increase in variance between
folds. Feature importance shows why, with terrain and distance-to-river carrying
68% of the weight and the radar change bands only 24%. Rather than publish one
footprint and one accuracy number, the map separates a small, genuinely reliable
core (99% correct, 9.5% of the footprint) from a much larger band that performs no
better than the base rate, and carries that distinction through into building and
road exposure estimates.

- **Live web map:** https://nellchris.github.io/valencia-flood-map/
- **Technical report:** https://nellchris.github.io/valencia-flood-map/report/

## Data sources

| Source | Provider | Licence |
|---|---|---|
| Sentinel-1A RTC (20 Oct, 1 Nov 2024) | ASF HyP3 / Copernicus | CC-BY 4.0 |
| Copernicus EMS EMSR773 observed flood | Copernicus EMS Rapid Mapping | Free, full and open |
| Copernicus DEM GLO-30 | ESA / Airbus | Copernicus DEM licence |
| ESA WorldCover 10 m v200 | ESA WorldCover consortium | CC-BY 4.0 |
| Buildings, roads, waterways | OpenStreetMap contributors | ODbL 1.0 |

Full per-source provenance, including which sensor produced each part of the
validation reference, is in [`provenance.csv`](provenance.csv). That attribution
matters: 99.44% of the reference was delineated from optical imagery and only 0.56%
from Sentinel-1, so validating a radar model against it is not circular.

## Run locally

No build step and no API keys — everything is static.

```bash
cd site
python -m http.server 8777
```

Then open http://localhost:8777. To view the report locally, serve the repository
root instead and open `/report/`.

## Repository layout

```
site/            the web map (index.html, style.css, app.js)
site/webmap/     web-ready exports: GeoJSON, PNG/JPG overlays, stats.json
report/          technical report (index.html + figures/)
scripts/         Python processing pipeline
provenance.csv   every input, with sensor, licence and URL
```

The raw Sentinel-1 archives and the intermediate GeoTIFFs in `processed/` are not
committed — they run to several gigabytes. The derived web exports in
`site/webmap/` are everything the map and report need.

## Author

Nelson Christopher (Geodel), 2026. Built with Sentinel-1 SAR data and open-source
tools. Contains modified Copernicus Sentinel data 2024.
