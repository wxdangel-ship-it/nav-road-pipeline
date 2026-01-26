# Evidence Schema (Primitive Evidence + World Candidates)

This document defines the minimum fields for evidence records and world candidates.
Downstream modules rely on these fields; do not break the contract.

## 1) Scope and Outputs
- Primitive Evidence: per-frame, per-provider evidence in pixel/world space.
- World Candidates: aggregated entities in map space, used by fusion/QA/gates.

Outputs (typical):
- `runs/<run_id>/evidence/<provider_id>.jsonl` (primitive evidence)
- `runs/<run_id>/outputs/*_utm32.gpkg` (world candidates)

## 2) Primitive Evidence Schema

### 2.1 Common fields (pixel + world)
- `kind`: `seg_map` | `det` | `poly` | `line`
- `provider_id` / `model_id` / `model_version`
- `source`: `image` | `lidar` | `sat` | `traj`
- `drive_id`, `frame_id`, `timestamp` (timestamp optional)
- `geom_crs`: `pixel` | `utm32` | `wgs84`
- `geometry`: bbox/mask/line/polygon payload (pixel or world)
- `quality`: quality stats (ex: score)
- `support`: support stats (ex: points_support, reproj_iou_bbox)
- `uncertainty`: drift or confidence details
- `provenance`: config/git/time details
- `backend_status`: `real` | `fallback` | `unavailable`
- `fallback_used` / `fallback_from` / `fallback_to` / `backend_reason`
- `score` (if applicable)
- `config_path` / `git_commit` / `generated_at`

### 2.2 Pixel-space evidence (`geom_crs = pixel`)
- `image_path`
- `bbox` (if `kind=det`)
- `mask` (if `kind=seg_map`)
  - `format`: `class_id_png` | `png` | `rle`
  - `path`
- `geometry_frame`: `image_px`

### 2.3 World-space evidence (`geom_crs = utm32 | wgs84`)
- `geometry`: map geometry in EPSG:32632 or EPSG:4326
- `points_support` (optional)
- `points_support_accum` (optional)
- `reproj_iou_bbox` (optional)
- `reproj_iou_dilated` (optional)

## 3) World Candidate Schema (Aggregated Entities)

### 3.1 Required fields
- `candidate_id` (stable id within run)
- `drive_id`, `frame_id`, `timestamp` (timestamp optional)
- `source` / `provider_id` / `version`
- `geom_crs`: `utm32` | `wgs84`
- `geometry` (Polygon/LineString/Point)
- `quality` (ex: score)
- `support` (points_support / points_support_accum / reproj_iou_bbox)
- `uncertainty` (drift_flag + details)
- `provenance` (config_path / git_commit / generated_at)
- `support.points_support`
- `support.points_support_accum`
- `support.reproj_iou_bbox`
- `support.reproj_iou_dilated`
- `rect_w` / `rect_l` / `rectangularity` (meters, minimum rotated rectangle)
- `drift_flag` (Stage2 drift or propagation failure)
- `prop_reason` (Stage2 propagation reason)
- `reject_reasons` (comma-separated or array)

### 3.2 Optional fields
- `candidate_type` (e.g., crosswalk)
- `support_frames` (list or json string)
- `proj_method` (e.g., lidar | plane | bbox_only | stage2_video)
- `qa_flag` (raw | gated | proj_fail | ...)

## 4) CRS Rules
- Any file with `_utm32` must be EPSG:32632.
- Any file with `_wgs84` must be EPSG:4326 (range-checked), otherwise rename or reproject.
