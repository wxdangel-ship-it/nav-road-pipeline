# nav-road-pipeline
Windows-native road production pipeline (POC->Prod)

## GEOM NN backend (optional)
- Optional deps: `pip install -r requirements_nn.txt`
- Run with NN backend: `set GEOM_BACKEND=nn` then `scripts\\build_geom.cmd ...`
- Default behavior: `GEOM_BACKEND=auto` prefers NN and falls back to algo on missing deps/errors.

## Strong road polygon smoothing
- Enable strong profile in postopt sweep (quick):
  - `scripts\\sweep_geom_postopt.cmd --config configs\\geom_postopt_strong_smooth.yaml --quick-only --quick-max-frames 400`
- Enable strong profile in postopt sweep (full):
  - `scripts\\sweep_geom_postopt.cmd --config configs\\geom_postopt_strong_smooth.yaml --full-only --full-max-frames 2000`
- Direct single run (ad hoc):
  - `set SMOOTH_PROFILE=strong` then `scripts\\build_geom.cmd --drive <drive_id> --max-frames 400`

## Smoothness metrics
- `roughness` (perimeter^2 / area) and `vertex_count` are logged in `GeomSummary.json`/`qc.json`.
- Lower `roughness` and lower `vertex_count` imply smoother, more regularized boundaries.

## SAT intersections + hybrid
- SAT quick (from sweep index):
  - `scripts\\sat_intersections.cmd --index runs\\<sweep_run>\\postopt_index.jsonl --stage quick --config configs\\sat_intersections_full.yaml --out-dir runs\\sat_intersections_quick --resume --write-back`
- SAT full (Golden8, segmented + resume):
  - `scripts\\sat_intersections.cmd --index runs\\<sweep_run>\\postopt_index.jsonl --stage full --config configs\\sat_intersections_full.yaml --out-dir runs\\sat_intersections_full --resume --write-back`
- SAT finalize only (merge segments):
  - `scripts\\sat_intersections.cmd --index runs\\<sweep_run>\\postopt_index.jsonl --stage full --config configs\\sat_intersections_full.yaml --out-dir runs\\sat_intersections_full --finalize`
- Hybrid final (algo/sat -> final):
  - `scripts\\intersections_hybrid.cmd --index runs\\<sweep_run>\\postopt_index.jsonl --stage full --config configs\\intersections_hybrid.yaml --out-dir runs\\sat_intersections_full`

Missing reason enums (per-drive CSV/JSON):
- `OK`, `no_candidates`, `no_tiles`, `out_of_coverage`, `low_confidence`, `read_error`, `missing_inputs`, `missing_entry`, `tiles_dir_missing`, `dop20_root_unset`, `dop20_root_missing`

QGIS layer suggestion:
- `road_polygon_wgs84.geojson` -> `intersections_algo_wgs84.geojson` -> `intersections_sat_wgs84.geojson` -> `intersections_final_wgs84.geojson`

## Intersection shape refine
- Config: `configs/intersections_shape_refine.yaml`
- Refine uses road polygon + centerlines to clip intersection polygons into road-shaped junctions.
- QC adds: `intersections_circularity`, `intersections_aspect_ratio`, `intersections_overlap_with_road`, `intersections_arm_count`
- Debug layers (build_geom): `outputs/debug/intersections_seed_points.geojson`, `..._local_clip.geojson`, `..._arms.geojson`, `..._refined.geojson`

## Centerlines modes
- Config: `configs/centerlines.yaml`
- Modes:
  - `single`: only base centerline
  - `dual`: split carriageway only (fallback to single if `dual_fallback_single=true`)
  - `both`: single + dual if divider split succeeds
  - `auto`: dual when divider split succeeds, otherwise single
- Dual gating: divider is required; oneway (future) can disable dual
- Divider sources:
  - `geom`: use geometry hints (multi-polygon / holes)
  - `seg`: use divider_median from image feature_store
- QGIS: load `centerlines_single_wgs84.geojson` + `centerlines_dual_wgs84.geojson` + `centerlines_both_wgs84.geojson`

## Image feature store (V1)
- Schema config:
  - `configs/seg_schema.yaml` (class/id/subtype mapping)
  - `configs/feature_schema.yaml` (geometry types + required fields)
- Build feature_store:
  - `scripts\\image_features.cmd --drive <drive_id> --model-out-dir <dir> --out-run-dir runs\\image_feat_v1 --seg-schema configs\\seg_schema.yaml --feature-schema configs\\feature_schema.yaml`
- Output layout:
  - `runs\\<run>\\feature_store\\<drive>\\<frame>\\image_features.gpkg`
  - `runs\\<run>\\feature_store\\<drive>\\<frame>\\traffic_light_dets.json`
  - `runs\\<run>\\feature_store\\index.json`
- Centerlines with seg divider:
  - `set FEATURE_STORE_DIR=runs\\image_feat_v1\\feature_store`
  - `scripts\\centerlines_v2.cmd --drive <drive_id> --max-frames 200`

## DOP20 WMS download (Golden8 AOI)
- Build AOI:
  - `python tools\\build_golden8_aoi.py --index runs\\sweep_geom_postopt_20260119_061421\\postopt_index.jsonl --out-dir runs\\golden8_aoi --margin-m 2000 --crs-epsg 32632`
- Download tiles (WMS -> jpg+jgw):
  - `scripts\\download_dop20.cmd`
  - Or override AOI: `scripts\\download_dop20.cmd runs\\golden8_aoi\\golden8_aoi.json`
