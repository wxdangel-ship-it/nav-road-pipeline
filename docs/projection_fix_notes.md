# Projection Fix Notes

## 2026-01-24
- Updated world->image projection to use `r_rect` + `k` after `t_velo_to_cam`.
- This aligns the projection chain with `project_feature_store_to_map` and avoids skipping rectification.
- Applied in:
  - `tools/run_crosswalk_monitor_range.py`
  - `tools/build_road_entities.py`
- Fixed world->ego rotation to use inverse yaw (transpose), matching `load_kitti360_lidar_points_world`.
- Roundtrip reprojection now uses `ground_z` from LiDAR stats when available.

## 2026-01-24 (Full-Pose LiDAR Option)
- Added optional full-pose LiDAR world transform (roll/pitch/yaw + cam_to_pose + cam_to_velo).
- Switch: `--lidar-world-mode fullpose` or `USE_FULLPOSE_LIDAR=1` (default remains legacy).
- Full-pose outputs use `lidar_*_utm32_fullpose_<drive>.tif` and QA index fields
  (`lidar_height_fullpose_path`, `lidar_intensity_fullpose_path`).
- Height raster now supports real z-statistics via `--lidar-height-stat` (p10/mean/p95).
