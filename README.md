# Valencia Flood Map

- **Live web map:** https://nellchris.github.io/valencia-flood-map/
- **Technical report:** https://nellchris.github.io/valencia-flood-map/report/

An interactive web map of the October 2024 Valencia flash flood, built from Sentinel-1 radar satellite imagery.

## What this is

A radar-based flood map covering the Horta Sud region southwest of Valencia, Spain, where flash flooding on 29 October 2024. The map classifies flooded areas using machine learning on satellite radar data, then honestly shows where the classification can and cannot be trusted.

## Method
Radar change detection — Compared Sentinel-1 SAR images from before (20 Oct) and after (1 Nov) the flood to measure backscatter change across the landscape.
Machine learning classification — Trained a Random Forest model on radar change, terrain height above drainage (HAND), slope, distance to rivers, and land cover to classify each 10 m pixel as flooded or dry.
Honest validation — Tested accuracy two ways: standard random cross-validation (F1 = 0.51) and stricter spatially blocked cross-validation (F1 = 0.42). The 17% gap and 62× increase in uncertainty reveal that the model mostly learns terrain, not radar flood signal.
Confidence layer — Classified every pixel as high, medium, or low confidence using data-driven thresholds calibrated on the spatial validation results, so users can see exactly where to trust the map.
Exposure analysis — Estimated affected buildings and roads at each confidence level, compared against the official Copernicus Emergency Management Service (EMS) reference.
Interactive web map — Built with MapLibre GL JS. Features include a before/after radar swipe comparison, toggleable flood footprint, confidence overlay, official EMS reference outline, affected buildings and roads, and panels showing exposure estimates, validation results, and methodology.
Key findings
The model works in wetland and cropland but has no reliable signal in built-up areas
Only 9 buildings fall in the high-confidence zone (89% precision), while 23,732 are in the full footprint (21% precision) — most sit in a probability band with no measurable skill
The 19.7-hour gap between the EMS reference and the post-event radar image limits agreement in fast-draining areas
Random cross-validation overstates accuracy by 17% and understates uncertainty by 62× — spatial validation should be standard practice for any geospatial ML study

## Data sources
Sentinel-1 RTC products from ASF HyP3
Copernicus EMS EMSR773 flood delineation (99.4% optically derived)
Copernicus GLO-30 DEM
ESA WorldCover 10 m land cover
OpenStreetMap buildings, roads, and waterways

## Author
Nelson Christopher · 2026
