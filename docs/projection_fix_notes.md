# Projection Fix Notes

## 2026-01-24
- Updated world->image projection to use `r_rect` + `k` after `t_velo_to_cam`.
- This aligns the projection chain with `project_feature_store_to_map` and avoids skipping rectification.
- Applied in:
  - `tools/run_crosswalk_monitor_range.py`
  - `tools/build_road_entities.py`
- Fixed world->ego rotation to use inverse yaw (transpose), matching `load_kitti360_lidar_points_world`.
- Roundtrip reprojection now uses `ground_z` from LiDAR stats when available.
