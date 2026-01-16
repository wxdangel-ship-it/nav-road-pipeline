# Geom Compare Report (autotune_20260116_173809)

## Configs
- winner: CFG_BASELINE_0001_B_T000
- compare: CFG_BASELINE_0001_B_T006

## Param Differences
```json
[
  [
    "M0",
    "impl_id",
    "m0_none",
    "m0_align2"
  ],
  [
    "M0",
    "param.align_tol_m",
    null,
    1.4173427018734939
  ],
  [
    "M2",
    "param.dummy_thr",
    0.5489960503508904,
    0.8630527420168641
  ],
  [
    "M4",
    "param.split_eps_m",
    14.534971879393218,
    15.64595181748218
  ],
  [
    "M6a",
    "impl_id",
    "m6a_stub",
    "m6a_stub2"
  ],
  [
    "M6a",
    "param.max_shift_m",
    null,
    1.6214560695373756
  ],
  [
    "M6a",
    "param.smooth_lambda",
    1.1864495772686976,
    1.7150048267104407
  ]
]
```

## Winner RunCard_Arm0 (geom)
### metrics
```json
{
  "C": 0.9564,
  "B_roughness": 0.3846,
  "A_dangling_per_km": 0.0,
  "prior_used": "NONE",
  "prior_confidence_p50": 0.0,
  "alignment_residual_p50": 999.0,
  "conflict_rate": 0.0,
  "image_coverage": 0.0,
  "pose_coverage": 0.0,
  "prior_osm_available": false,
  "prior_sat_available": false
}
```
### qc
```json
{
  "road_bbox_dx_m": 256.0,
  "road_bbox_dy_m": 171.0,
  "road_bbox_diag_m": 307.859,
  "road_component_count_before": 96,
  "road_component_count_after": 1,
  "centerline_1_length_m": 290.097,
  "centerline_2_length_m": 280.876,
  "centerline_total_length_m": 570.973,
  "centerlines_in_polygon_ratio": 1.0,
  "intersections_count": 1,
  "intersections_area_total_m2": 1045.029,
  "intersections_top5_area_m2": [
    1045.029
  ],
  "width_median_m": 24.943,
  "width_p95_m": 39.057,
  "peak_point_count": 3,
  "cluster_count": 1,
  "inter_component_count_before": 1,
  "inter_component_count_after": 1,
  "grid_resolution_m": 0.5,
  "density_threshold": 3,
  "corridor_m": 15.0,
  "simplify_m": 1.2
}
```
### score_terms
```json
{
  "base": {
    "C": 0.9564,
    "B_roughness": 0.3846,
    "A_dangling_per_km": 0.0,
    "prior_used": "NONE",
    "prior_confidence_p50": 0.0,
    "alignment_residual_p50": 999.0,
    "conflict_rate": 0.0,
    "image_coverage": 0.0,
    "pose_coverage": 0.0,
    "prior_osm_available": false,
    "prior_sat_available": false
  },
  "delta": {
    "C": 0.0,
    "B": 0.0,
    "A": 0.0
  },
  "surrogate": {
    "C": 0.0,
    "B": 0.0,
    "A": 0.0
  }
}
```

## Compare RunCard_Arm0 (geom)
### metrics
```json
{
  "C": 0.9559,
  "B_roughness": 0.3951,
  "A_dangling_per_km": 0.0,
  "prior_used": "NONE",
  "prior_confidence_p50": 0.0,
  "alignment_residual_p50": 999.0,
  "conflict_rate": 0.0,
  "image_coverage": 0.0,
  "pose_coverage": 0.0,
  "prior_osm_available": false,
  "prior_sat_available": false
}
```
### qc
```json
{
  "road_bbox_dx_m": 256.0,
  "road_bbox_dy_m": 171.5,
  "road_bbox_diag_m": 308.137,
  "road_component_count_before": 102,
  "road_component_count_after": 1,
  "centerline_1_length_m": 290.097,
  "centerline_2_length_m": 280.876,
  "centerline_total_length_m": 570.973,
  "centerlines_in_polygon_ratio": 1.0,
  "intersections_count": 1,
  "intersections_area_total_m2": 1197.806,
  "intersections_top5_area_m2": [
    1197.806
  ],
  "width_median_m": 25.664,
  "width_p95_m": 38.43,
  "peak_point_count": 3,
  "cluster_count": 1,
  "inter_component_count_before": 1,
  "inter_component_count_after": 1,
  "grid_resolution_m": 0.5,
  "density_threshold": 3,
  "corridor_m": 15.311,
  "simplify_m": 1.2
}
```
### score_terms
```json
{
  "base": {
    "C": 0.9559,
    "B_roughness": 0.3951,
    "A_dangling_per_km": 0.0,
    "prior_used": "NONE",
    "prior_confidence_p50": 0.0,
    "alignment_residual_p50": 999.0,
    "conflict_rate": 0.0,
    "image_coverage": 0.0,
    "pose_coverage": 0.0,
    "prior_osm_available": false,
    "prior_sat_available": false
  },
  "delta": {
    "C": 0.0,
    "B": 0.0,
    "A": 0.0
  },
  "surrogate": {
    "C": 0.0,
    "B": 0.0,
    "A": 0.0
  }
}
```

## Attribution
参数差异与 QC/得分关联说明：
- M2.dummy_thr 差异触发 build_geom 中 `density_thr`：T000 使用 3（dummy_thr≈0.549），T006 仍为 3（dummy_thr≈0.863），因此点云密度阈值相同，road_component_count_before 变化不大。
- M6a.max_shift_m 影响 `corridor_m`：T000 corridor_m=15.0，T006 corridor_m=15.311，road_bbox_diag_m 与 intersections_area_total_m2 略有上升。
- M6a.smooth_lambda 仅影响 `simplify_m`（两者均为 1.2），因此此参数当前未绑定到几何输出。
- M0/M4 的实现与参数当前未绑定到 build_geom，导致 QC 与分数差异有限。
